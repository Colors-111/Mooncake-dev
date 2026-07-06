# Guaranteed SSD Offload Priority — Phase 2 Implementation Plan

> **Status: ✅ IMPLEMENTED & VERIFIED (2026-07-06).** All 4 tasks done, 4 tests pass. Client-side only: `BucketMetadata.guaranteed` + `YLT_REFL` (survives restart) + `OffloadObjects` splits into homogeneous buckets + `SelectEvictionCandidate` skips guaranteed buckets (FIFO forward-scan; LRU forward-scan-no-erase). Compile bug fixes applied (DistributedStorageBackend override missed; `return res.error()`→`tl::make_unexpected`; NoopComplete free function). Reference: spec §6.4, §11 Phase 2.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Protect guaranteed objects' SSD replicas from client-side fifo/lru eviction — once written to SSD (Phase 1), a guaranteed bucket is never selected for eviction. This is the second slice of the SSD-managed-lifecycle design (spec §11 Phase 2).

**Architecture:** Client-side only (`storage_backend.cpp` + `file_storage.cpp`). Phase 1 propagates `OffloadTaskItem.guaranteed` master→client, but the client currently **drops it** at `OffloadObjects` (builds `map<string,int64_t>` of key+size only). Phase 2 (a) threads `guaranteed` from `OffloadObjects` through to `BucketMetadata` by splitting guaranteed/normal tasks into **homogeneous buckets** (so a bucket is entirely guaranteed or entirely normal), and (b) makes `SelectEvictionCandidate` skip guaranteed buckets. Eviction has a **single chokepoint** (`PrepareEviction`→`SelectEvictionCandidate`), purely trigger-based inside `BatchOffload` — no background sweep.

**Tech Stack:** C++17, GoogleTest, struct_pack (`YLT_REFL`), Mooncake Store client storage backend.

**Reference spec:** `docs/superpowers/specs/2026-07-02-guaranteed-ssd-offload-priority-design-zh.md` (§6.4, §11 Phase 2).

**Git convention:** `git add` per task (no commit), per user preference. Final review at the end.

---

## Scope

Phase 2 = client-side SSD replica eviction protection. Master side is unchanged (Phase 1 already propagates the flag).

**In scope:**
- `OffloadObjects` splits tasks by `guaranteed` → homogeneous buckets.
- `BucketMetadata.guaranteed` field (persisted via `YLT_REFL` so it survives `BatchLoad` restart).
- `BuildBucket` sets `bucket->guaranteed`.
- `SelectEvictionCandidate` skips guaranteed buckets (FIFO: forward scan; LRU: skip without erasing from `lru_index_`).

**Deferred to Phase 3:** TTL on SSD replicas (guaranteed bucket is permanently protected in Phase 2; Phase 3 TTL expiry makes it evictable again). Accepted failure mode: if guaranteed buckets fill the disk, `WriteBucket` ENOSPC-fails the next offload (backpressure) — Phase 3 TTL mitigates.

**Pre-existing build note:** `mooncake_store` may not compile on the dev machine (missing `libmsgpack-dev` + yalantinglibs include propagation; `dependencies.sh` never run). If `cmake --build` fails on `ylt/util/expected.hpp`, that is environmental — the user verifies in their build environment. Write correct code + self-review by reading.

---

## File Structure

| File | Responsibility | Task |
|------|---------------|------|
| `mooncake-store/include/storage_backend.h` | `BucketMetadata.guaranteed` field + 4 ctor/assignment updates + `YLT_REFL` | 1 |
| `mooncake-store/src/file_storage.cpp` | `OffloadObjects` splits tasks by guaranteed | 2 |
| `mooncake-store/src/storage_backend.cpp` | `BuildBucket` sets guaranteed; `SelectEvictionCandidate` skips (FIFO+LRU) | 3, 4 |
| `mooncake-store/tests/guaranteed_eviction_test.cpp` | new test file | 1, 4 |
| `mooncake-store/tests/CMakeLists.txt` | register test target | 1 |

---

## Task 1: Add `guaranteed` field to `BucketMetadata` (+ test scaffold + CMake)

**Files:**
- Modify: `mooncake-store/include/storage_backend.h:33-91` (BucketMetadata)
- Create: `mooncake-store/tests/guaranteed_eviction_test.cpp`
- Modify: `mooncake-store/tests/CMakeLists.txt:46`

- [ ] **Step 1: Register the new test target in CMake**

In `mooncake-store/tests/CMakeLists.txt`, after the `add_store_test(guaranteed_offload_test guaranteed_offload_test.cpp)` line, add:

```cmake
add_store_test(guaranteed_eviction_test guaranteed_eviction_test.cpp)
```

- [ ] **Step 2: Write the failing test (file scaffold + field-default test)**

Create `mooncake-store/tests/guaranteed_eviction_test.cpp`. This task adds only the scaffold + a field test; eviction tests land in Task 4. Mirror `storage_backend_test.cpp`'s fixture pattern:

```cpp
#include "storage_backend.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "allocator.h"
#include "types.h"
#include "utils.h"

namespace fs = std::filesystem;
namespace mooncake::test {

class GuaranteedEvictionTest : public ::testing::Test {
   protected:
    void SetUp() override {
        google::InitGoogleLogging("GuaranteedEvictionTest");
        FLAGS_logtostderr = true;
        data_path_ = "/tmp/guaranteed_eviction_test_" +
                     std::to_string(getpid());
        fs::remove_all(data_path_);
        fs::create_directories(data_path_);
    }

    void TearDown() override {
        google::ShutdownGoogleLogging();
        std::error_code ec;
        fs::remove_all(data_path_, ec);
    }

    std::string data_path_;
};

// A complete_handler that just accepts the write (records nothing). Free
// function (not a fixture member) so the OffloadBatch helper (also a free
// function, Task 4) can call it — a protected static member would be
// inaccessible from a free function.
ErrorCode NoopComplete(const std::vector<std::string>&,
                      std::vector<StorageObjectMetadata>&) {
    return ErrorCode::OK;
}

// Task 1: BucketMetadata defaults to non-guaranteed.
TEST_F(GuaranteedEvictionTest, BucketMetadataDefaultsNonGuaranteed) {
    BucketMetadata meta;
    EXPECT_FALSE(meta.guaranteed);
    meta.guaranteed = true;
    EXPECT_TRUE(meta.guaranteed);
}

}  // namespace mooncake::test
```

- [ ] **Step 3: Run test to verify it fails**

Run:
```bash
cmake --build build --target guaranteed_eviction_test -j"$(nproc)" 2>&1 | tail -20
```
Expected: **compile failure** — `BucketMetadata` has no `guaranteed` field. (If the build fails on `ylt/util/expected.hpp` instead, that is the pre-existing env issue — note it and proceed; correctness is verified by reading.)

- [ ] **Step 4: Add the `guaranteed` field to `BucketMetadata`**

In `mooncake-store/include/storage_backend.h`, in `struct BucketMetadata` (lines 33-91):

After line 37 (`std::vector<BucketObjectMetadata> metadatas;`), add:

```cpp
    bool guaranteed{false};  // true => bucket is never selected for eviction
                             // (Phase 2: set when all keys are guaranteed).
```

Then update the **copy constructor** (lines 50-56) — add `guaranteed(other.guaranteed),` to the initializer list (before `inflight_reads_(0),`):

```cpp
    BucketMetadata(const BucketMetadata& other)
        : meta_size(other.meta_size),
          data_size(other.data_size),
          keys(other.keys),
          metadatas(other.metadatas),
          guaranteed(other.guaranteed),
          inflight_reads_(0),
          last_access_ns_(0) {}
```

Update the **move constructor** (lines 59-65) — add `guaranteed(other.guaranteed),`:

```cpp
    BucketMetadata(BucketMetadata&& other) noexcept
        : meta_size(other.meta_size),
          data_size(other.data_size),
          keys(std::move(other.keys)),
          metadatas(std::move(other.metadatas)),
          guaranteed(other.guaranteed),
          inflight_reads_(0),
          last_access_ns_(0) {}
```

Update **copy assignment** (lines 68-77) — add `guaranteed = other.guaranteed;` inside the `if`:

```cpp
    BucketMetadata& operator=(const BucketMetadata& other) {
        if (this != &other) {
            meta_size = other.meta_size;
            data_size = other.data_size;
            keys = other.keys;
            metadatas = other.metadatas;
            guaranteed = other.guaranteed;
            // Don't copy runtime state
        }
        return *this;
    }
```

Update **move assignment** (lines 80-89) — add `guaranteed = other.guaranteed;`:

```cpp
    BucketMetadata& operator=(BucketMetadata&& other) noexcept {
        if (this != &other) {
            meta_size = other.meta_size;
            data_size = other.data_size;
            keys = std::move(other.keys));
            metadatas = std::move(other.metadatas));
            guaranteed = other.guaranteed;
            // Don't move runtime state
        }
        return *this;
    }
```

Add `guaranteed` to the `YLT_REFL` macro (line 91) so it persists to the `.meta` file and survives `BatchLoad` restart:

```cpp
YLT_REFL(BucketMetadata, data_size, keys, metadatas, guaranteed);
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cmake --build build --target guaranteed_eviction_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_eviction_test \
    --gtest_filter=GuaranteedEvictionTest.BucketMetadataDefaultsNonGuaranteed
```
Expected: PASS.

- [ ] **Step 6: Stage**

```bash
git add mooncake-store/include/storage_backend.h mooncake-store/tests/guaranteed_eviction_test.cpp mooncake-store/tests/CMakeLists.txt
```

---

## Task 2: `OffloadObjects` splits tasks by `guaranteed` into homogeneous buckets

This is the core threading fix: Phase 1's `guaranteed` flag is dropped at `OffloadObjects` (line 373 builds `map<string,int64_t>`). Split the tasks into guaranteed/normal groups, each independently bucketed + offloaded, so a bucket is homogeneous.

**Files:**
- Modify: `mooncake-store/src/file_storage.cpp:361-491` (OffloadObjects)

- [ ] **Step 1: Read the current `OffloadObjects` to confirm exact text**

Read `mooncake-store/src/file_storage.cpp` lines 361-491. The current flow (verified):
- lines 366-375: build `storage_object_sizes` (`map<string,int64_t>`, drops `guaranteed`) + `task_by_storage_key` (keeps full task).
- lines 377-394: `AllocateOffloadingBuckets(storage_object_sizes, buckets_keys)` (or fallback single bucket).
- lines 429-491: per `buckets_keys` group, build `batch_object`, define `eviction_handler`, call `storage_backend_->BatchOffload(...)`.

The change: instead of one `storage_object_sizes` map + one `AllocateOffloadingBuckets` call, split into two maps (`guaranteed_sizes` / `normal_sizes`), call `AllocateOffloadingBuckets` per group, and pass the `guaranteed` flag to `BatchOffload` (Task 3 adds the param). Since `BatchOffload`'s guaranteed param lands in Task 3, **this task only does the split + flags each group's buckets_keys with a guaranteed bool** — the actual `BatchOffload(..., guaranteed)` call is wired in Task 3 (or this task adds it if Task 3 is merged first; coordinate so the param exists).

- [ ] **Step 2: Refactor `OffloadObjects` to split by guaranteed**

In `mooncake-store/src/file_storage.cpp`, replace the section from line 366 (`std::unordered_map<std::string, int64_t> storage_object_sizes;`) through the `AllocateOffloadingBuckets`/fallback block (end ~line 394) with a split version. Read the exact current code first, then:

```cpp
    std::unordered_map<std::string, int64_t> guaranteed_sizes;
    std::unordered_map<std::string, int64_t> normal_sizes;
    std::unordered_map<std::string, OffloadTaskItem> task_by_storage_key;
    guaranteed_sizes.reserve(offloading_objects.size());
    normal_sizes.reserve(offloading_objects.size());
    task_by_storage_key.reserve(offloading_objects.size());
    for (const auto& task : offloading_objects) {
        const auto storage_key =
            MakeTenantScopedStorageKey(task.tenant_id, task.key);
        task_by_storage_key.emplace(storage_key, task);
        if (task.guaranteed) {
            guaranteed_sizes.emplace(storage_key, task.size);
        } else {
            normal_sizes.emplace(storage_key, task.size);
        }
    }

    // Bucket each guarantee-class separately so buckets are homogeneous
    // (all-guaranteed or all-normal). A guaranteed bucket is never selected
    // for eviction (SelectEvictionCandidate, storage_backend.cpp).
    struct BucketGroup {
        std::vector<std::vector<std::string>> buckets_keys;
        bool guaranteed;
    };
    std::vector<BucketGroup> groups;

    auto allocate_group =
        [this](const std::unordered_map<std::string, int64_t>& sizes,
               bool guaranteed) -> tl::expected<BucketGroup, ErrorCode> {
        BucketGroup group;
        group.guaranteed = guaranteed;
        if (auto bucket_backend = std::dynamic_pointer_cast<BucketStorageBackend>(
                storage_backend_)) {
            auto allocate_res =
                bucket_backend->AllocateOffloadingBuckets(sizes, group.buckets_keys);
            if (!allocate_res) {
                LOG(ERROR) << "AllocateOffloadingBuckets failed with error: "
                           << allocate_res.error();
                return tl::make_unexpected(allocate_res.error());
            }
        } else {
            std::vector<std::string> keys;
            keys.reserve(sizes.size());
            for (const auto& it : sizes) {
                keys.emplace_back(it.first);
            }
            group.buckets_keys.emplace_back(std::move(keys));
        }
        return group;
    };

    if (!guaranteed_sizes.empty()) {
        auto res = allocate_group(guaranteed_sizes, /*guaranteed=*/true);
        if (!res) return tl::make_unexpected(res.error());
        groups.push_back(std::move(res.value()));
    }
    if (!normal_sizes.empty()) {
        auto res = allocate_group(normal_sizes, /*guaranteed=*/false);
        if (!res) return tl::make_unexpected(res.error());
        groups.push_back(std::move(res.value()));
    }
```

Then replace the per-bucket loop (lines 429-491) to iterate `groups` instead of `buckets_keys`, and pass `group.guaranteed` to `BatchOffload`. The loop body (building `batch_object` from keys, defining `eviction_handler`, calling `storage_backend_->BatchOffload`) stays the same; only the outer iteration changes from:

```cpp
    for (const auto& keys : buckets_keys) {
```

to:

```cpp
    for (const auto& group : groups) {
        const bool group_guaranteed = group.guaranteed;
        for (const auto& keys : group.buckets_keys) {
            // ... existing body ...
            auto result = storage_backend_->BatchOffload(
                batch_object, complete_handler, eviction_handler,
                group_guaranteed);  // <-- new 4th arg (Task 3 adds the param)
            // ...
        }
    }
```

**Coordinate with Task 3:** the `BatchOffload(..., group_guaranteed)` 4th argument requires Task 3 to have added the `bool guaranteed = false` param to `BucketStorageBackend::BatchOffload`. Implement Task 3 before/with this task, or temporarily pass it once the param exists. The cleanest order: Task 3 first (adds the param + threads to BuildBucket), then Task 2 (uses it).

- [ ] **Step 3: (SKIP if env broken) build + verify existing file_storage tests still pass**

Run:
```bash
cmake --build build --target file_storage_test -j"$(nproc)" && \
./build/mooncake-store/tests/file_storage_test
```
Expected: existing tests pass (normal offload path unchanged in behavior — just split into a single normal group when no guaranteed tasks).

- [ ] **Step 4: Stage**

```bash
git add mooncake-store/src/file_storage.cpp
```

---

## Task 3: `BucketStorageBackend::BatchOffload` + `BuildBucket` thread `guaranteed` to `BucketMetadata`

Add a trailing `bool guaranteed = false` param to `BatchOffload` (default false → existing callers unchanged), pass it to `BuildBucket`, set `bucket->guaranteed` there.

**Files:**
- Modify: `mooncake-store/include/storage_backend.h` (`BucketStorageBackend::BatchOffload` decl ~626)
- Modify: `mooncake-store/src/storage_backend.cpp` (`BatchOffload` def ~1279, `BuildBucket` def ~1978)

- [ ] **Step 1: Add the `guaranteed` param to `BatchOffload` declarations (base + override)**

The base interface `StorageBackendInterface::BatchOffload` (`mooncake-store/include/storage_backend.h:255`) is **pure virtual**, and `FileStorage::storage_backend_` is typed `std::shared_ptr<StorageBackendInterface>` (`file_storage.h:133`). So `OffloadObjects`'s `storage_backend_->BatchOffload(..., guaranteed)` resolves to the **base** signature — the base MUST gain the param, plus the `BucketStorageBackend` override.

In `mooncake-store/include/storage_backend.h`, base interface (~line 255):

```cpp
    virtual tl::expected<int64_t, ErrorCode> BatchOffload(
        const std::unordered_map<std::string, std::vector<Slice>>& batch_object,
        std::function<ErrorCode(const std::vector<std::string>& keys,
                                std::vector<StorageObjectMetadata>& metadatas)>
            complete_handler,
        std::function<void(const std::vector<std::string>& evicted_keys)>
            eviction_handler = nullptr,
        bool guaranteed = false) = 0;
```

`BucketStorageBackend` override (~line 626) — add the same trailing param:

```cpp
    tl::expected<int64_t, ErrorCode> BatchOffload(
        const std::unordered_map<std::string, std::vector<Slice>>& batch_object,
        std::function<ErrorCode(const std::vector<std::string>& keys,
                                std::vector<StorageObjectMetadata>& metadatas)>
            complete_handler,
        std::function<void(const std::vector<std::string>& evicted_keys)>
            eviction_handler = nullptr,
        bool guaranteed = false) override;
```

Check for OTHER `StorageBackendInterface` subclasses with a `BatchOffload` override (grep `BatchOffload` in `storage_backend.h` — e.g. the `FilePerKey`/legacy backend around line 753/1036). Each override must add the trailing `bool guaranteed = false` param (even if it ignores it — the legacy backend just treats everything as non-guaranteed). Add `bool guaranteed = false` to every `BatchOffload` override in the header; in their `.cpp` definitions, accept-and-ignore (or pass through). This keeps the base contract consistent.

- [ ] **Step 2: Update the `BatchOffload` definition + pass to `BuildBucket`**

In `mooncake-store/src/storage_backend.cpp` (definition ~1279), change the signature to accept `bool guaranteed` (no default in the definition):

```cpp
tl::expected<int64_t, ErrorCode> BucketStorageBackend::BatchOffload(
    const std::unordered_map<std::string, std::vector<Slice>>& batch_object,
    std::function<ErrorCode(const std::vector<std::string>& keys,
                            std::vector<StorageObjectMetadata>& metadatas)>
        complete_handler,
    std::function<void(const std::vector<std::string>& evicted_keys)>
        eviction_handler,
    bool guaranteed) {
```

Then change the `BuildBucket` call (line ~1306-1307):

```cpp
    auto build_bucket_result =
        BuildBucket(bucket_id, batch_object, iovs, metadatas, guaranteed);
```

- [ ] **Step 3: Update `BuildBucket` to set `bucket->guaranteed`**

In `mooncake-store/src/storage_backend.cpp`, `BuildBucket` definition (~1977), add the trailing `bool guaranteed` param and set the field. Change signature:

```cpp
tl::expected<std::shared_ptr<BucketMetadata>, ErrorCode>
BucketStorageBackend::BuildBucket(
    int64_t bucket_id,
    const std::unordered_map<std::string, std::vector<Slice>>& batch_object,
    std::vector<iovec>& iovs, std::vector<StorageObjectMetadata>& metadatas,
    bool guaranteed) {
    auto bucket = std::make_shared<BucketMetadata>();
    bucket->guaranteed = guaranteed;
    int64_t storage_offset = 0;
    // ... rest unchanged ...
```

Also update the `BuildBucket` **declaration** in `mooncake-store/include/storage_backend.h` (grep for `BuildBucket` in the private section) to add `bool guaranteed = false`.

- [ ] **Step 4: (SKIP if env broken) build to verify compilation**

Run:
```bash
cmake --build build --target storage_backend_test -j"$(nproc)" 2>&1 | tail -15
```
Expected: builds cleanly. Existing callers omit `guaranteed` (default false) → non-guaranteed buckets → unchanged behavior.

- [ ] **Step 5: Stage**

```bash
git add mooncake-store/include/storage_backend.h mooncake-store/src/storage_backend.cpp
```

---

## Task 4: `SelectEvictionCandidate` skips guaranteed buckets (FIFO + LRU)

The core protection. FIFO: forward-scan from `begin()` to first non-guaranteed. LRU: skip guaranteed entries in `lru_index_` **without erasing** (erasing would lose them permanently).

**Files:**
- Modify: `mooncake-store/src/storage_backend.cpp:2180-2225` (SelectEvictionCandidate)
- Modify: `mooncake-store/tests/guaranteed_eviction_test.cpp` (eviction tests)

- [ ] **Step 1: Write the failing tests**

Append to `mooncake-store/tests/guaranteed_eviction_test.cpp` (before closing `}  // namespace mooncake::test`):

```cpp
// Helper: offload a batch and return the bucket_id.
tl::expected<int64_t, ErrorCode> OffloadBatch(
    BucketStorageBackend& backend,
    const std::vector<std::pair<std::string, std::string>>& kv,
    bool guaranteed) {
    std::unordered_map<std::string, std::vector<Slice>> batch;
    // buffers must outlive the call; leak is acceptable in a short test, but
    // better to keep them alive until the test ends — return them via a static
    // pool to avoid use-after-free. Simpler: allocate with new[] and don't free
    // (test process exits shortly).
    static std::vector<std::unique_ptr<char[]>> pool;
    for (const auto& [key, value] : kv) {
        auto buf = std::make_unique<char[]>(value.size());
        std::memcpy(buf.get(), value.data(), value.size());
        batch.emplace(key, std::vector<Slice>{Slice{buf.get(), value.size()}});
        pool.push_back(std::move(buf));
    }
    return backend.BatchOffload(batch, NoopComplete,
                                /*eviction_handler=*/nullptr, guaranteed);
}

// Task 4A (FIFO): a guaranteed bucket is NOT evicted even under quota pressure,
// while a normal bucket IS evicted to make room.
TEST_F(GuaranteedEvictionTest, FifoSkipsGuaranteedBucket) {
    FileStorageConfig config;
    config.storage_filepath = data_path_;
    BucketBackendConfig bucket_config;
    bucket_config.eviction_policy = BucketEvictionPolicy::FIFO;
    bucket_config.max_total_size = 1024;  // tiny cap => eviction on 2nd offload
    BucketStorageBackend backend(config, bucket_config);
    ASSERT_TRUE(backend.Init());

    // First offload: a guaranteed bucket (well under cap).
    ASSERT_TRUE(OffloadBatch(backend, {{"g1", std::string(256, 'g')}}, true));
    // Second offload: a normal bucket whose size exceeds the cap => must evict.
    // The guaranteed bucket must NOT be selected; the normal write must either
    // succeed (evicting nothing because only the guaranteed bucket exists, and
    // it's protected) or fail with ENOSPC — but the guaranteed bucket survives.
    auto res = OffloadBatch(backend, {{"n1", std::string(768, 'n')}}, false);

    // The guaranteed key must still be present (not evicted).
    auto g_exist = backend.IsExist("g1");
    ASSERT_TRUE(g_exist);
    EXPECT_TRUE(g_exist.value())
        << "guaranteed bucket must survive eviction pressure";


}

// Task 4B (FIFO): a normal bucket IS evicted when pressure comes from another
// normal offload (regression: normal eviction still works).
TEST_F(GuaranteedEvictionTest, FifoEvictsNormalBucket) {
    FileStorageConfig config;
    config.storage_filepath = data_path_;
    BucketBackendConfig bucket_config;
    bucket_config.eviction_policy = BucketEvictionPolicy::FIFO;
    bucket_config.max_total_size = 1024;
    BucketStorageBackend backend(config, bucket_config);
    ASSERT_TRUE(backend.Init());

    ASSERT_TRUE(OffloadBatch(backend, {{"n1", std::string(256, 'a')}}, false));
    // Force eviction of n1 by writing a second normal batch over the cap.
    ASSERT_TRUE(OffloadBatch(backend, {{"n2", std::string(768, 'b')}}, false));

    auto n1_exist = backend.IsExist("n1");
    ASSERT_TRUE(n1_exist);
    EXPECT_FALSE(n1_exist.value())
        << "normal (non-guaranteed) bucket must be evicted under pressure";


}
```

**Note:** confirm `BucketStorageBackend` has a public `Cleanup()` / `Init()` / `IsExist(key)` method by reading `storage_backend.h` before relying on them. If `Cleanup()` doesn't exist, the destructor handles teardown (omit the call). `IsExist` returns `tl::expected<bool, ErrorCode>` (confirmed in `storage_backend_test.cpp` usage).

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cmake --build build --target guaranteed_eviction_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_eviction_test \
    --gtest_filter='GuaranteedEvictionTest.Fifo*'
```
Expected: 4A FAIL (guaranteed bucket evicted — no skip logic yet); 4B may pass (existing FIFO evicts normal).

- [ ] **Step 3: Implement FIFO skip in `SelectEvictionCandidate`**

In `mooncake-store/src/storage_backend.cpp`, `SelectEvictionCandidate` (~2181), change the FIFO case from:

```cpp
        case BucketEvictionPolicy::FIFO:
            // buckets_ is ordered by bucket_id (monotonically increasing),
            // so begin() is always the oldest bucket.
            return buckets_.begin();
```

to a forward scan skipping guaranteed buckets:

```cpp
        case BucketEvictionPolicy::FIFO:
            // buckets_ is ordered by bucket_id (monotonically increasing),
            // so begin() is always the oldest bucket. Skip guaranteed buckets
            // (they are never eviction candidates).
            for (auto it = buckets_.begin(); it != buckets_.end(); ++it) {
                if (!it->second->guaranteed) {
                    return it;
                }
            }
            return buckets_.end();
```

- [ ] **Step 4: Implement LRU skip (no erase, forward-scan — NOT `++top_it`)**

In the same function, the LRU case (~2200-2220). Guaranteed entries must be **skipped without erasing** (erasing loses them permanently — reads don't re-insert — so the bucket could never be evicted even after Phase 3 TTL expiry).

**IMPORTANT (dead-loop fix):** the naive `if (guaranteed) { ++top_it; continue; }` would **infinite-loop**, because each `while` iteration resets `auto top_it = lru_index_.begin()` — if `begin()` is guaranteed, `continue` restarts at the same guaranteed `begin()` forever. Instead, scan FORWARD from `top_it` for the first non-guaranteed, still-existing entry. Replace the LRU loop body with:

```cpp
        case BucketEvictionPolicy::LRU:
            while (!lru_index_.empty()) {
                auto top_it = lru_index_.begin();
                auto [ts, id] = *top_it;
                auto bucket_it = buckets_.find(id);
                if (bucket_it == buckets_.end()) {
                    lru_index_.erase(top_it);
                    continue;
                }
                // Skip guaranteed buckets WITHOUT erasing from lru_index_:
                // erasing would lose the entry permanently (reads don't
                // re-insert), so the bucket could never be evicted even after
                // its TTL expires (Phase 3). Instead, scan forward through the
                // index (ordered by {ts, id}) for the first non-guaranteed,
                // still-existing entry. We do NOT erase the skipped guaranteed
                // entries. If none is found, return end() so the caller breaks
                // out of its eviction loop instead of spinning forever.
                if (bucket_it->second->guaranteed) {
                    auto scan_it = top_it;
                    ++scan_it;
                    auto chosen_it = lru_index_.end();
                    for (; scan_it != lru_index_.end(); ++scan_it) {
                        auto [s_ts, s_id] = *scan_it;
                        auto s_bucket_it = buckets_.find(s_id);
                        if (s_bucket_it == buckets_.end()) {
                            // Stale/missing bucket: skip (not erased here; the
                            // top-of-loop path lazily discards it if/when it
                            // surfaces to begin()).
                            continue;
                        }
                        if (s_bucket_it->second->guaranteed) {
                            continue;
                        }
                        chosen_it = scan_it;
                        break;
                    }
                    if (chosen_it == lru_index_.end()) {
                        return buckets_.end();  // no evictable candidate remains
                    }
                    // Validate the chosen entry's timestamp (may be stale). If
                    // stale, repair and let the outer while retry from the top.
                    auto [c_ts, c_id] = *chosen_it;
                    auto c_bucket_it = buckets_.find(c_id);
                    int64_t actual_ts = c_bucket_it->second->last_access_ns_.load(
                        std::memory_order_relaxed);
                    if (actual_ts == c_ts) {
                        lru_index_.erase(chosen_it);
                        return c_bucket_it;
                    }
                    lru_index_.erase(chosen_it);
                    lru_index_.emplace(actual_ts, c_id);
                    continue;
                }
                int64_t actual_ts = bucket_it->second->last_access_ns_.load(
                    std::memory_order_relaxed);
                if (actual_ts == ts) {
                    lru_index_.erase(top_it);
                    return bucket_it;
                }
                // Stale: repair and retry to find the true minimum.
                lru_index_.erase(top_it);
                lru_index_.emplace(actual_ts, id);
            }
            return buckets_.end();
```

**Critical:** guaranteed entries are NEVER erased (the `chosen_it` is always non-guaranteed by construction; the guaranteed `top_it` is only read). The forward-scan guarantees termination (each outer-while iteration returns, repairs one stale entry, or returns `end()`). The existing stale-repair (erase+emplace) for non-guaranteed top entries is preserved.

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
cmake --build build --target guaranteed_eviction_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_eviction_test \
    --gtest_filter='GuaranteedEvictionTest.Fifo*'
```
Expected: PASS.

- [ ] **Step 6: Add an LRU skip test + run full suite**

Append to `guaranteed_eviction_test.cpp`:

```cpp
// Task 4C (LRU): a guaranteed bucket is NOT evicted under LRU pressure.
TEST_F(GuaranteedEvictionTest, LruSkipsGuaranteedBucket) {
    FileStorageConfig config;
    config.storage_filepath = data_path_;
    BucketBackendConfig bucket_config;
    bucket_config.eviction_policy = BucketEvictionPolicy::LRU;
    bucket_config.max_total_size = 1024;
    BucketStorageBackend backend(config, bucket_config);
    ASSERT_TRUE(backend.Init());

    ASSERT_TRUE(OffloadBatch(backend, {{"g1", std::string(256, 'g')}}, true));
    // Touch g1 to update its LRU timestamp (so it's "most recently used"), then
    // write a normal batch that exceeds the cap. Under LRU the oldest normal
    // bucket should be evicted, not g1.
    std::unordered_map<std::string, StorageObjectMetadata> meta;
    ASSERT_TRUE(backend.BatchQuery(std::vector<std::string>{"g1"}, meta));
    ASSERT_TRUE(OffloadBatch(backend, {{"n1", std::string(768, 'n')}}, false));

    auto g_exist = backend.IsExist("g1");
    ASSERT_TRUE(g_exist);
    EXPECT_TRUE(g_exist.value()) << "guaranteed bucket must survive LRU eviction";


}
```

Run:
```bash
cmake --build build --target guaranteed_eviction_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_eviction_test
```
Expected: all tests PASS. Confirm `BatchQuery` signature (takes `vector<string>` keys + out-param `map<string,StorageObjectMetadata>`; returns `tl::expected`) by reading `storage_backend.h` before use.

- [ ] **Step 7: Run existing storage_backend/file_storage tests (no regression)**

Run:
```bash
cmake --build build --target storage_backend_test file_storage_test -j"$(nproc)" && \
./build/mooncake-store/tests/storage_backend_test && \
./build/mooncake-store/tests/file_storage_test
```
Expected: PASS — normal eviction behavior unchanged (default `guaranteed=false` → no skip).

- [ ] **Step 8: Stage**

```bash
git add mooncake-store/src/storage_backend.cpp mooncake-store/tests/guaranteed_eviction_test.cpp
```

---

## Final Verification

- [ ] **Step 1: Build all affected test targets**

```bash
cmake --build build --target guaranteed_eviction_test storage_backend_test file_storage_test guaranteed_offload_test -j"$(nproc)"
```

- [ ] **Step 2: Run the full new + regression suite**

```bash
./build/mooncake-store/tests/guaranteed_eviction_test
./build/mooncake-store/tests/storage_backend_test
./build/mooncake-store/tests/file_storage_test
./build/mooncake-store/tests/guaranteed_offload_test
```
Expected: all PASS.

- [ ] **Step 3: Confirm zero regression with Phase 1**

Phase 1's `guaranteed_offload_test` must still pass unchanged (Phase 2 is client-side; master-side Phase 1 behavior untouched).

- [ ] **Step 4: End-to-end (optional, if a real master+client harness is available)**

If a Python e2e test can run master + client with `--enable_guaranteed_cache=true --enable_offload=true --offload_on_evict=true`, verify a guaranteed object survives memory eviction pressure on the SSD. This requires the full stack; defer if only unit tests are available.

---

## Notes

- **`guaranteed` survives restart**: `BucketMetadata.guaranteed` is in `YLT_REFL`, so `BatchLoad` re-marks protected buckets on restart. Without this, restart would lose protection.
- **LRU skip must not erase (and must forward-scan, not `++top_it`)**: erasing a guaranteed entry from `lru_index_` is permanent (reads don't re-insert), so the bucket could never be evicted even after Phase 3 TTL expiry. The skip uses a forward-scan for the first non-guaranteed entry (NOT `++top_it; continue;`, which would infinite-loop because `while` resets `top_it = begin()` each iteration and a guaranteed `begin()` is revisited forever).
- **Accepted failure mode**: guaranteed buckets filling the disk → `WriteBucket` ENOSPC → offload fails (backpressure). Phase 3 TTL mitigates. No backpressure queue in Phase 2 (YAGNI).
- **Phase 1 build env note applies**: if `cmake --build` fails on `ylt/util/expected.hpp`, that is the pre-existing env issue (`dependencies.sh` not run). Verify by reading; the user runs the actual compile/test.

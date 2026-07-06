# Guaranteed SSD Offload Priority — Phase 1 Implementation Plan

> **Status: ✅ IMPLEMENTED & VERIFIED (2026-07-06).** All 9 tasks + supplemental cases 5&9 done, 12 tests pass. Master-side only: independent guaranteed offload queue (no limit) + PutEnd always-offload + NACK retry, behind `enable_guaranteed_cache` flag (default off). Compile bug fixes applied (const-OffloadingTask → erase+emplace). Reference: spec §4–§10.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure a `guaranteed` object is always written to SSD — never rejected for a full offload queue (independent per-client queue, no limit), never abandoned on SSD write failure (retry) — behind an `enable_guaranteed_cache` flag that defaults off.

**Architecture:** Master-side only. A `guaranteed` boolean flows ReplicateConfig → ObjectMetadata → PushOffloadingQueue, which routes guaranteed objects into a separate per-client `guaranteed_offloading_objects` map (parallel to the existing `promotion_objects` precedent) that has no size limit. `PutEnd` always offloads guaranteed objects regardless of `offload_on_evict_`. `OffloadObjectHeartbeat` drains both maps. `NotifyOffloadSuccess` re-enqueues guaranteed objects on NACK. Everything is gated by `enable_guaranteed_cache_` (default false → zero behavior change).

**Tech Stack:** C++17, GoogleTest, glog, struct_pack (`YLT_REFL`), Mooncake Store master service.

**Reference spec:** `docs/superpowers/specs/2026-07-02-guaranteed-ssd-offload-priority-design-zh.md` (§4–§14).

**Git convention for this work:** The user prefers staging over committing per-step. End each task with `git add` (not `git commit`); make one commit at the end of the task (or defer all commits to the user). Adjust per user preference.

---

## Scope

This plan implements **Phase 1 (ensure write to SSD)** — master-side only. It covers spec §4–§10, test cases 1–9.

**Deferred to Phase 2** (flagged to user): client-side `BuildBucket` bucket splitting by `OffloadTaskItem.guaranteed` (spec §6.4, test case 10). Rationale: it has no observable effect without Phase 2's client-side bucket-level pin (eviction protection), and it lives in `storage_backend.cpp` (a different module). The master already propagates the `guaranteed` flag to the client via `OffloadTaskItem` (Task 1) and `OffloadObjectHeartbeat` (Task 8), so the client has the information it needs; Phase 2 just uses it in `BuildBucket` + `SelectEvictionCandidate`.

---

## File Structure

| File | Responsibility | Task |
|------|---------------|------|
| `mooncake-store/include/types.h` | `OffloadTaskItem.guaranteed` wire flag | 1 |
| `mooncake-store/include/replica.h` | `ReplicateConfig.guaranteed_until_ms` marker | 2 |
| `mooncake-store/src/master.cpp` | gflag `enable_guaranteed_cache` + config read + log | 3 |
| `mooncake-store/include/master_config.h` | config structs + copy blocks | 3 |
| `mooncake-store/include/master_service.h` | `enable_guaranteed_cache_` member; `ObjectMetadata.guaranteed_`; `PushOffloadingQueue` decl | 3, 4, 6 |
| `mooncake-store/src/master_service.cpp` | flag init; marking; PutEnd; PushOffloadingQueue; heartbeat drain; NACK retry | 3, 4, 6, 7, 8, 9 |
| `mooncake-store/include/segment.h` | `LocalDiskSegment.guaranteed_offloading_objects` map | 5 |
| `mooncake-store/tests/guaranteed_offload_test.cpp` | new test file | 1, 7, 8, 9 |
| `mooncake-store/tests/CMakeLists.txt` | register new test target | 1 |

---

## Task 1: Add `guaranteed` field to `OffloadTaskItem`

**Files:**
- Modify: `mooncake-store/include/types.h:263-273`
- Create: `mooncake-store/tests/guaranteed_offload_test.cpp`
- Modify: `mooncake-store/tests/CMakeLists.txt:46`

- [ ] **Step 1: Register the new test target in CMake**

In `mooncake-store/tests/CMakeLists.txt`, after line 46 (`add_store_test(offload_on_evict_test offload_on_evict_test.cpp)`), add:

```cmake
add_store_test(guaranteed_offload_test guaranteed_offload_test.cpp)
```

- [ ] **Step 2: Write the failing test (file scaffold + equality/flag test)**

Create `mooncake-store/tests/guaranteed_offload_test.cpp` with the fixture (adapted from `offload_on_evict_test.cpp:1-120`) and the first test:

```cpp
#include "master_service.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "types.h"

namespace mooncake::test {

class GuaranteedOffloadTest : public ::testing::Test {
   protected:
    void SetUp() override {
        google::InitGoogleLogging("GuaranteedOffloadTest");
        FLAGS_logtostderr = true;
    }

    void TearDown() override { google::ShutdownGoogleLogging(); }

    static constexpr size_t kDefaultSegmentBase = 0x300000000;

    Segment MakeSegment(std::string name, size_t base, size_t size) const {
        Segment segment;
        segment.id = generate_uuid();
        segment.name = std::move(name);
        segment.base = base;
        segment.size = size;
        segment.te_endpoint = segment.name;
        return segment;
    }

    struct MountedSegmentContext {
        UUID segment_id;
        UUID client_id;
    };

    MountedSegmentContext PrepareSegment(MasterService& service,
                                         std::string name, size_t base,
                                         size_t size) const {
        Segment segment = MakeSegment(std::move(name), base, size);
        UUID client_id = generate_uuid();
        auto mount_result = service.MountSegment(segment, client_id);
        EXPECT_TRUE(mount_result.has_value());
        return {.segment_id = segment.id, .client_id = client_id};
    }

    // Put an object and complete it. guaranteed_until_ms>0 marks it guaranteed.
    void PutObject(MasterService& service, const UUID& client_id,
                   const std::string& key, int64_t guaranteed_until_ms = 0,
                   size_t size = 1024) {
        ReplicateConfig config;
        config.replica_num = 1;
        config.guaranteed_until_ms = guaranteed_until_ms;
        auto put_start =
            service.PutStart(client_id, key, "default", size, config);
        ASSERT_TRUE(put_start.has_value()) << "PutStart failed for key=" << key;
        auto put_end =
            service.PutEnd(client_id, key, "default", ReplicaType::MEMORY);
        ASSERT_TRUE(put_end.has_value()) << "PutEnd failed for key=" << key;
    }

    // Drain the offload queue via OffloadObjectHeartbeat.
    std::vector<OffloadTaskItem> DrainOffloadQueue(MasterService& service,
                                                    const UUID& client_id) {
        auto res = service.OffloadObjectHeartbeat(client_id, true);
        if (!res) {
            return {};
        }
        return res.value();
    }

    template <typename Predicate>
    void WaitUntil(Predicate&& predicate,
                   std::chrono::milliseconds timeout = std::chrono::milliseconds(4000),
                   std::chrono::milliseconds interval =
                       std::chrono::milliseconds(50)) const {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (predicate()) {
                return;
            }
            std::this_thread::sleep_for(interval);
        }
        EXPECT_TRUE(predicate());
    }
};

// Task 1: OffloadTaskItem carries a guaranteed flag and compares on it.
TEST_F(GuaranteedOffloadTest, OffloadTaskItemCarriesGuaranteedFlag) {
    OffloadTaskItem normal{.tenant_id = "default", .key = "k", .size = 10,
                           .guaranteed = false};
    OffloadTaskItem guar{.tenant_id = "default", .key = "k", .size = 10,
                         .guaranteed = true};
    EXPECT_EQ(normal, (OffloadTaskItem{.tenant_id = "default", .key = "k",
                                        .size = 10, .guaranteed = false}));
    EXPECT_NE(normal, guar)
        << "Items differing only in guaranteed must not be equal";
}

}  // namespace mooncake::test
```

- [ ] **Step 3: Run test to verify it fails**

Run:
```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" 2>&1 | tail -20
```
Expected: **compile failure** — `OffloadTaskItem` has no `.guaranteed` field (designated initializer `guaranteed` does not exist).

- [ ] **Step 4: Add the `guaranteed` field to `OffloadTaskItem`**

In `mooncake-store/include/types.h:263-273`, replace:

```cpp
struct OffloadTaskItem {
    std::string tenant_id;
    std::string key;
    int64_t size;

    bool operator==(const OffloadTaskItem& other) const {
        return tenant_id == other.tenant_id && key == other.key &&
               size == other.size;
    }
};
YLT_REFL(OffloadTaskItem, tenant_id, key, size);
```

with:

```cpp
struct OffloadTaskItem {
    std::string tenant_id;
    std::string key;
    int64_t size;
    bool guaranteed{false};  // set at enqueue from ObjectMetadata.guaranteed_

    bool operator==(const OffloadTaskItem& other) const {
        return tenant_id == other.tenant_id && key == other.key &&
               size == other.size && guaranteed == other.guaranteed;
    }
};
YLT_REFL(OffloadTaskItem, tenant_id, key, size, guaranteed);
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_offload_test \
    --gtest_filter=GuaranteedOffloadTest.OffloadTaskItemCarriesGuaranteedFlag
```
Expected: PASS.

- [ ] **Step 6: Stage**

```bash
git add mooncake-store/include/types.h mooncake-store/tests/guaranteed_offload_test.cpp mooncake-store/tests/CMakeLists.txt
```

---

## Task 2: Add `guaranteed_until_ms` to `ReplicateConfig`

**Files:**
- Modify: `mooncake-store/include/replica.h:81-144`

- [ ] **Step 1: Write the failing test**

Append to `mooncake-store/tests/guaranteed_offload_test.cpp` (before the closing `}  // namespace`):

```cpp
// Task 2: ReplicateConfig carries guaranteed_until_ms (Phase 1: only >0 is checked).
TEST_F(GuaranteedOffloadTest, ReplicateConfigCarriesGuaranteedUntilMs) {
    ReplicateConfig config;
    EXPECT_EQ(config.guaranteed_until_ms, 0);  // default: no guarantee
    config.guaranteed_until_ms = 60000;
    EXPECT_GT(config.guaranteed_until_ms, 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" 2>&1 | tail -10
```
Expected: **compile failure** — `ReplicateConfig` has no member `guaranteed_until_ms`.

- [ ] **Step 3: Add the field**

In `mooncake-store/include/replica.h`, after line 97 (`std::optional<std::vector<std::string>> group_ids{};`) and before line 99 (`ReplicateConfig ForSingleKey...`), insert:

```cpp
    // Guaranteed offload: when >0, this object's SSD offload is mandatory
    // (routed to the independent guaranteed queue, retried on failure).
    // Phase 1 treats this as a boolean marker (>0 => guaranteed).
    int64_t guaranteed_until_ms{0};
```

- [ ] **Step 4: Update the `operator<<` (so logging includes it)**

In the same file, in `operator<<` (around line 132, after the `group_ids` block and before `os << " }";`), add:

```cpp
        os << ", guaranteed_until_ms: " << config.guaranteed_until_ms;
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_offload_test \
    --gtest_filter=GuaranteedOffloadTest.ReplicateConfigCarriesGuaranteedUntilMs
```
Expected: PASS.

- [ ] **Step 6: Stage**

```bash
git add mooncake-store/include/replica.h mooncake-store/tests/guaranteed_offload_test.cpp
```

---

## Task 3: Wire the `enable_guaranteed_cache` config flag (default false)

This is infrastructure (no isolated behavior; gating is verified in Task 7). It mirrors `enable_offload` through the config layers. Tests construct `MasterServiceConfig` directly, so the **test-essential** parts are the `MasterServiceConfig` field + `MasterService` member + initializer (Steps 3-5). The **production CLI** parts (gflag, `MasterConfig`, `WrappedMasterServiceConfig`, copy blocks, log — Steps 6-7) mirror `enable_offload` everywhere it appears.

**Files:**
- Modify: `mooncake-store/include/master_config.h` (MasterServiceConfig ~974, MasterConfig ~53, Wrapped + copy blocks, builder)
- Modify: `mooncake-store/include/master_service.h:1902` (member)
- Modify: `mooncake-store/src/master_service.cpp:205` (initializer)
- Modify: `mooncake-store/src/master.cpp` (gflag, GetBool, override, log)
- Modify: `mooncake-store/tests/guaranteed_offload_test.cpp` (test)

- [ ] **Step 1: Write the failing test (config field default + assignable)**

Append to `guaranteed_offload_test.cpp`:

```cpp
// Task 3: enable_guaranteed_cache defaults false and is settable on MasterServiceConfig.
TEST_F(GuaranteedOffloadTest, EnableGuaranteedCacheConfigField) {
    MasterServiceConfig config;
    EXPECT_FALSE(config.enable_guaranteed_cache);  // default off
    config.enable_guaranteed_cache = true;
    EXPECT_TRUE(config.enable_guaranteed_cache);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" 2>&1 | tail -10
```
Expected: **compile failure** — `MasterServiceConfig` has no `enable_guaranteed_cache`.

- [ ] **Step 3 (test-essential): Add the field to `MasterServiceConfig`**

In `mooncake-store/include/master_config.h`, in `class MasterServiceConfig` after line 974 (`bool enable_offload = false;`), add:

```cpp
    bool enable_guaranteed_cache = false;
```

- [ ] **Step 4 (test-essential): Add the `MasterService` member**

In `mooncake-store/include/master_service.h`, after line 1902 (`const bool enable_offload_;`), add:

```cpp
    const bool enable_guaranteed_cache_{false};
```

- [ ] **Step 5 (test-essential): Initialize the member in the constructor**

In `mooncake-store/src/master_service.cpp`, in the initializer list after line 205 (`enable_offload_(config.enable_offload),`), add:

```cpp
      enable_guaranteed_cache_(config.enable_guaranteed_cache),
```

- [ ] **Step 6 (test-essential): Run test to verify it passes**

Run:
```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_offload_test \
    --gtest_filter=GuaranteedOffloadTest.EnableGuaranteedCacheConfigField
```
Expected: PASS. (The flag is now functional in tests. Steps 7-8 add production CLI parity.)

- [ ] **Step 7 (production CLI): Mirror `enable_offload` everywhere it appears**

The flag must also flow from the `--enable_guaranteed_cache` gflag through `MasterConfig` and the `WrappedMasterServiceConfig`/builder conversion variants into `MasterServiceConfig` for production. Rather than enumerate fragile line numbers, mirror `enable_offload` at every occurrence. First, find them:

```bash
grep -n "enable_offload" mooncake-store/include/master_config.h mooncake-store/src/master.cpp
```

At **each** returned line, add a parallel `enable_guaranteed_cache` line in the same style. The occurrences to handle:

1. **gflag definition** (`master.cpp`, `DEFINE_bool(enable_offload, false, ...)`): after it add
   ```cpp
   DEFINE_bool(enable_guaranteed_cache, false,
               "Enable guaranteed offload: objects put with guaranteed_until_ms>0 "
               "are always written to SSD (independent queue, retried on failure). "
               "Defaults off for zero behavior change.");
   ```
2. **config-file read** (`master.cpp`, `default_config.GetBool("enable_offload", &master_config.enable_offload, FLAGS_enable_offload);`): after it add
   ```cpp
   default_config.GetBool("enable_guaranteed_cache",
                           &master_config.enable_guaranteed_cache,
                           FLAGS_enable_guaranteed_cache);
   ```
3. **explicit-CLI-override block** (`master.cpp`, the `if ((google::GetCommandLineFlagInfo("enable_offload", &info) && !info.is_default) || !conf_set) { master_config.enable_offload = FLAGS_enable_offload; }`): after it add the parallel block for `enable_guaranteed_cache`.
4. **startup log line** (`master.cpp`, `<< ", enable_offload=" << master_config.enable_offload`): after it add
   ```cpp
   << ", enable_guaranteed_cache=" << master_config.enable_guaranteed_cache
   ```
5. **`MasterConfig` struct field** (`master_config.h`, `bool enable_offload;`): add `bool enable_guaranteed_cache = false;` near it.
6. **`WrappedMasterServiceConfig` and other config variants** (`master_config.h`): `enable_offload` appears as `RequiredParam<bool> enable_offload{...}` in WrappedMasterServiceConfig. Since `enable_guaranteed_cache` **defaults off** (not required), add it as a **plain bool** `bool enable_guaranteed_cache = false;` (do **not** use `RequiredParam`) in each config struct that declares `enable_offload`.
7. **copy blocks** (`master_config.h`, lines like `enable_offload = config.enable_offload;`): at each, add
   ```cpp
   enable_guaranteed_cache = config.enable_guaranteed_cache;
   ```
   (match indentation). Find them all with:
   ```bash
   grep -n "enable_offload = config.enable_offload" mooncake-store/include/master_config.h
   ```
8. **builder** (`master_config.h`, `set_enable_offload` + `enable_offload_` member + `config.enable_offload = enable_offload_;` build write-back): mirror with `set_enable_guaranteed_cache` + `enable_guaranteed_cache_` + the write-back line.
9. **`InProcMasterConfig`** (`master_config.h`, `std::optional<bool> enable_offload;` ~1169 + its builder `enable_offload_` ~1186, setter ~1218, write-back ~1272): mirror as `std::optional<bool> enable_guaranteed_cache;`; in the write-back use `config.enable_guaranteed_cache = enable_guaranteed_cache_.value_or(false);` (default off when unset).

If any occurrence is ambiguous, build and let the compiler pinpoint missing spots.

- [ ] **Step 8 (production CLI): Build the master to verify the full wiring compiles**

Run:
```bash
cmake --build build --target mooncake_master -j"$(nproc)" 2>&1 | tail -20
```
Expected: builds cleanly. (If a copy block's source struct lacks the field, the compiler points to it — add the plain-bool field there per Step 6.)

- [ ] **Step 9: Stage**

```bash
git add mooncake-store/src/master.cpp mooncake-store/include/master_config.h \
        mooncake-store/include/master_service.h mooncake-store/src/master_service.cpp \
        mooncake-store/tests/guaranteed_offload_test.cpp
```

---

## Task 4: Mark `ObjectMetadata` guaranteed + thread from config

Adds the `guaranteed_` member (const, immutable after construction) and threads it from `ReplicateConfig.guaranteed_until_ms` under the `enable_guaranteed_cache_` gate. Verified behaviorally in Task 7 (no isolated assertion — `guaranteed_` is private).

**Files:**
- Modify: `mooncake-store/include/master_service.h:862-887` (ctor), `~912` (member)
- Modify: `mooncake-store/src/master_service.cpp:2816-2820` (marking), `~8527` (deserialize — NO change needed, trailing default)

- [ ] **Step 1: Add the `guaranteed_` member**

In `mooncake-store/include/master_service.h`, after line 912 (`const bool hard_pinned{false};          // immutable, set at creation`), add:

```cpp
    const bool guaranteed_{false};        // immutable, set at creation
                                          // (Phase 1: boolean marker; Phase 3
                                          // upgrades to guaranteed_until TTL)
```

- [ ] **Step 2: Add a trailing ctor parameter + initializer**

In `mooncake-store/include/master_service.h`, the constructor signature ends (line 869) with:

```cpp
        std::string tenant_id_ = "default",
        std::string user_key_ = {})
```

Change the closing to add a trailing defaulted param:

```cpp
        std::string tenant_id_ = "default",
        std::string user_key_ = {},
        bool enable_guaranteed = false)
```

Then in the member initializer list (line 879, `hard_pinned(enable_hard_pin),`), after it add:

```cpp
          guaranteed_(enable_guaranteed),
```

- [ ] **Step 3: Thread it in `AllocateAndInsertMetadata`**

In `mooncake-store/src/master_service.cpp:2816-2820`, the `emplace` forwards positional args. The current call:

```cpp
    auto [it, inserted] = tenant_state.metadata.emplace(
        std::piecewise_construct, std::forward_as_tuple(key),
        std::forward_as_tuple(client_id, now, value_length, std::move(replicas),
                              config.with_soft_pin, config.with_hard_pin,
                              config.data_type, group_id, tenant_id, key));
```

Append the guaranteed arg (it maps to the new trailing `enable_guaranteed` param, after `key`):

```cpp
    auto [it, inserted] = tenant_state.metadata.emplace(
        std::piecewise_construct, std::forward_as_tuple(key),
        std::forward_as_tuple(client_id, now, value_length, std::move(replicas),
                              config.with_soft_pin, config.with_hard_pin,
                              config.data_type, group_id, tenant_id, key,
                              enable_guaranteed_cache_ &&
                                  config.guaranteed_until_ms > 0));
```

- [ ] **Step 4: Confirm the HA-deserialize call sites need no change**

Grep for all `ObjectMetadata` construction sites:

```bash
grep -n "make_unique<ObjectMetadata>\|metadata.emplace" mooncake-store/src/master_service.cpp
```

There are **three** construction sites:
- **Line 2816** (`AllocateAndInsertMetadata`) — the live path; updated in Step 3.
- **Line 8330** (HA snapshot `emplace` into shard, ending `..., metadata_ptr->group_id, tenant_id, user_key`) — HA deserialize.
- **Line 8527** (`DeserializeMetadata` `make_unique<ObjectMetadata>(...)`, ending `..., is_hard_pinned, data_type, group_id`) — HA deserialize.

Because `enable_guaranteed` is a **trailing defaulted param** (after `user_key_`), both HA sites (8330 and 8527) pick up the default `false` — correct for HA restart (guaranteed_ resets to false per spec §7.10). **Do not edit them.** Verify they still compile (they will — all new params have defaults).

- [ ] **Step 5: Build to verify compilation**

Run:
```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" 2>&1 | tail -15
```
Expected: builds cleanly. (No new test here — `guaranteed_` is private; behavior is verified in Task 7.)

- [ ] **Step 6: Stage**

```bash
git add mooncake-store/include/master_service.h mooncake-store/src/master_service.cpp
```

---

## Task 5: Add `guaranteed_offloading_objects` map to `LocalDiskSegment`

Infrastructure only (verified by Task 8). `LocalDiskSegment` lives in `segment.h`, **not** `master_service.h`.

**Files:**
- Modify: `mooncake-store/include/segment.h:85-106`

- [ ] **Step 1: Add the map**

In `mooncake-store/include/segment.h`, after line 91 (the `offloading_objects` declaration):

```cpp
    std::unordered_map<std::string, OffloadTaskItem> GUARDED_BY(
        offloading_mutex_) offloading_objects;
```

add:

```cpp
    // Guaranteed offload queue (parallel to offloading_objects). Populated by
    // PushOffloadingQueue when guaranteed=true. No size limit — guaranteed
    // objects must reach SSD. Drained by OffloadObjectHeartbeat alongside
    // offloading_objects. Same locking (offloading_mutex_).
    std::unordered_map<std::string, OffloadTaskItem> GUARDED_BY(
        offloading_mutex_) guaranteed_offloading_objects;
```

- [ ] **Step 2: Build to verify compilation**

Run:
```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" 2>&1 | tail -10
```
Expected: builds cleanly. (The constructor at segment.h:98 needs no change — the map default-constructs.)

- [ ] **Step 3: Stage**

```bash
git add mooncake-store/include/segment.h
```

---

## Task 6: Route guaranteed objects to the independent queue in `PushOffloadingQueue`

Adds a trailing `bool guaranteed = false` param and selects the map; guaranteed skips the size-limit check. Default param keeps existing callers (PutEnd, eviction-path 6821/7066) compiling unchanged. Behavior verified in Task 7.

**Files:**
- Modify: `mooncake-store/include/master_service.h` (declaration)
- Modify: `mooncake-store/src/master_service.cpp:4932-4978` (definition)

- [ ] **Step 1: Update the declaration**

Find the `PushOffloadingQueue` declaration in `mooncake-store/include/master_service.h`:

```bash
grep -n "tl::expected<void, ErrorCode> PushOffloadingQueue" mooncake-store/include/master_service.h
```

Change its signature from:

```cpp
    tl::expected<void, ErrorCode> PushOffloadingQueue(
        const ObjectIdentity& object_id, Replica& replica);
```

to:

```cpp
    tl::expected<void, ErrorCode> PushOffloadingQueue(
        const ObjectIdentity& object_id, Replica& replica,
        bool guaranteed = false);
```

- [ ] **Step 2: Update the definition to select the map + skip limit for guaranteed**

In `mooncake-store/src/master_service.cpp:4932-4978`, replace the whole function body with:

```cpp
tl::expected<void, ErrorCode> MasterService::PushOffloadingQueue(
    const ObjectIdentity& object_id, Replica& replica, bool guaranteed) {
    const auto& segment_names = replica.get_segment_names();
    if (segment_names.empty()) {
        return {};
    }
    for (const auto& segment_name_it : segment_names) {
        if (!segment_name_it.has_value()) {
            continue;
        }
        ScopedLocalDiskSegmentAccess local_disk_segment_access =
            segment_manager_.getLocalDiskSegmentAccess();
        const auto& client_by_name =
            local_disk_segment_access.getClientByName();
        auto client_id_it = client_by_name.find(segment_name_it.value());
        if (client_id_it == client_by_name.end()) {
            return tl::make_unexpected(ErrorCode::SEGMENT_NOT_FOUND);
        }
        auto& client_local_disk_segment =
            local_disk_segment_access.getClientLocalDiskSegment();
        auto local_disk_segment_it =
            client_local_disk_segment.find(client_id_it->second);
        if (local_disk_segment_it == client_local_disk_segment.end()) {
            return tl::make_unexpected(ErrorCode::UNABLE_OFFLOADING);
        }
        MutexLocker locker(&local_disk_segment_it->second->offloading_mutex_);
        if (!local_disk_segment_it->second->enable_offloading) {
            return tl::make_unexpected(ErrorCode::UNABLE_OFFLOADING);
        }
        // Select the queue. Guaranteed objects use a separate per-client map
        // with NO size limit — they must reach SSD. Normal objects keep the
        // existing offloading_queue_limit_ enforcement.
        auto& queue = guaranteed
                          ? local_disk_segment_it->second->guaranteed_offloading_objects
                          : local_disk_segment_it->second->offloading_objects;
        if (!guaranteed && queue.size() >= offloading_queue_limit_) {
            return tl::make_unexpected(ErrorCode::KEYS_ULTRA_LIMIT);
        }
        const int64_t size = replica.get_descriptor()
                                 .get_memory_descriptor()
                                 .buffer_descriptor.size_;
        auto res = queue.emplace(
            MakeTenantScopedStorageKey(object_id.tenant_id, object_id.user_key),
            OffloadTaskItem{.tenant_id = object_id.tenant_id,
                            .key = object_id.user_key,
                            .size = size,
                            .guaranteed = guaranteed});
        if (!res.second) {
            return tl::make_unexpected(ErrorCode::OBJECT_ALREADY_EXISTS);
        }
    }
    return {};
}
```

- [ ] **Step 3: Build to verify compilation**

Run:
```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" 2>&1 | tail -15
```
Expected: builds cleanly. Existing callers omit `guaranteed` (default false) → unchanged behavior.

- [ ] **Step 4: Stage**

```bash
git add mooncake-store/include/master_service.h mooncake-store/src/master_service.cpp
```

---

## Task 7: `PutEnd` always offloads guaranteed objects (core behavior)

This is the heart of Phase 1. Changes the PutEnd offload condition so guaranteed objects offload at PutEnd regardless of `offload_on_evict_`, passing `guaranteed` to `PushOffloadingQueue`. Three behavioral tests verify: (A) guaranteed exempt from the normal-queue limit; (C) guaranteed offloads even under `offload_on_evict=true`; (D) flag-off degrades guaranteed to normal.

**Files:**
- Modify: `mooncake-store/src/master_service.cpp:3064-3084`
- Modify: `mooncake-store/tests/guaranteed_offload_test.cpp` (tests)

- [ ] **Step 1: Write the failing tests**

Append to `guaranteed_offload_test.cpp`:

```cpp
// Task 7A: guaranteed object is enqueued even when the normal offload queue is full.
TEST_F(GuaranteedOffloadTest, GuaranteedExemptFromNormalQueueLimit) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = true;
    config.offloading_queue_limit = 4;  // small limit so we can fill it
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true).has_value());

    // Fill the normal offload queue to the limit (no drain yet).
    for (int i = 0; i < 4; ++i) {
        PutObject(*service, ctx.client_id, "normal_" + std::to_string(i));
    }
    // A guaranteed object must still be enqueued despite the full normal queue.
    PutObject(*service, ctx.client_id, "guar", /*guaranteed_until_ms=*/60000);

    auto drained = DrainOffloadQueue(*service, ctx.client_id);
    bool found_guar = false;
    for (const auto& t : drained) {
        if (t.key == "guar") {
            found_guar = true;
            EXPECT_TRUE(t.guaranteed) << "guaranteed key must carry the flag";
        }
    }
    EXPECT_TRUE(found_guar) << "guaranteed object must be enqueued even when "
                               "the normal queue is full";

    service->RemoveAll();
}

// Task 7B: a normal object past the limit is NOT enqueued (regression guard).
TEST_F(GuaranteedOffloadTest, NormalStillRespectsQueueLimit) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = true;
    config.offloading_queue_limit = 4;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true).has_value());

    for (int i = 0; i < 4; ++i) {
        PutObject(*service, ctx.client_id, "normal_" + std::to_string(i));
    }
    PutObject(*service, ctx.client_id, "overflow");  // 5th normal -> KEYS_ULTRA_LIMIT

    auto drained = DrainOffloadQueue(*service, ctx.client_id);
    bool found_overflow = false;
    for (const auto& t : drained) {
        if (t.key == "overflow") found_overflow = true;
    }
    EXPECT_FALSE(found_overflow)
        << "5th normal object must be rejected (queue full)";

    service->RemoveAll();
}

// Task 7C: under offload_on_evict=true, guaranteed STILL offloads at PutEnd.
TEST_F(GuaranteedOffloadTest, GuaranteedOffloadsAtPutEndEvenWhenOffloadOnEvict) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = true;
    config.offload_on_evict = true;  // normally defers normal offload to eviction
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true).has_value());

    PutObject(*service, ctx.client_id, "guar", /*guaranteed_until_ms=*/60000);
    PutObject(*service, ctx.client_id, "normal");

    auto drained = DrainOffloadQueue(*service, ctx.client_id);
    bool found_guar = false, found_normal = false;
    for (const auto& t : drained) {
        if (t.key == "guar") found_guar = true;
        if (t.key == "normal") found_normal = true;
    }
    EXPECT_TRUE(found_guar)
        << "guaranteed must offload at PutEnd even under offload_on_evict=true";
    EXPECT_FALSE(found_normal)
        << "normal object must NOT PutEnd-offload under offload_on_evict=true";

    service->RemoveAll();
}

// Task 7D: with enable_guaranteed_cache=false, a guaranteed_until_ms>0 Put
// degrades to normal (no guaranteed offload under offload_on_evict=true).
TEST_F(GuaranteedOffloadTest, FlagOffDegradesGuaranteedToNormal) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = false;  // flag off
    config.offload_on_evict = true;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true).has_value());

    PutObject(*service, ctx.client_id, "would_be_guar", /*guaranteed_until_ms=*/60000);

    auto drained = DrainOffloadQueue(*service, ctx.client_id);
    EXPECT_TRUE(drained.empty())
        << "with flag off + offload_on_evict, guaranteed_until_ms>0 must NOT "
           "offload at PutEnd (degrades to normal)";

    service->RemoveAll();
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_offload_test \
    --gtest_filter='GuaranteedOffloadTest.Guaranteed*:GuaranteedOffloadTest.Normal*:GuaranteedOffloadTest.FlagOff*'
```
Expected: 7A/7C **FAIL** (guaranteed not offloaded — PutEnd still gated on `!offload_on_evict_` and/or not passing guaranteed); 7D may already pass; 7B may already pass (existing limit behavior).

- [ ] **Step 3: Change the PutEnd offload condition + pass guaranteed**

In `mooncake-store/src/master_service.cpp:3064-3084`, replace:

```cpp
    if (enable_offload_ && !offload_on_evict_) {
        auto& tenant_state = accessor.GetTenantState();
        bool task_created = false;
        metadata.VisitReplicas(
            [](const Replica& replica) {
                return replica.is_completed() && replica.is_memory_replica();
            },
            [this, &object_id, &tenant_state, &task_created](Replica& replica) {
                auto result = PushOffloadingQueue(object_id, replica);
                if (result) {
                    if (!task_created) {
                        replica.inc_refcnt();
                        tenant_state.offloading_tasks.emplace(
                            object_id.user_key,
                            OffloadingTask{replica.id(),
                                           std::chrono::system_clock::now()});
                        task_created = true;
                    }
                }
            });
    }
```

with:

```cpp
    if (enable_offload_ && (!offload_on_evict_ || metadata.guaranteed_)) {
        auto& tenant_state = accessor.GetTenantState();
        bool task_created = false;
        metadata.VisitReplicas(
            [](const Replica& replica) {
                return replica.is_completed() && replica.is_memory_replica();
            },
            [this, &object_id, &tenant_state, &task_created,
             guaranteed = metadata.guaranteed_](Replica& replica) {
                auto result =
                    PushOffloadingQueue(object_id, replica, guaranteed);
                if (result) {
                    if (!task_created) {
                        replica.inc_refcnt();
                        tenant_state.offloading_tasks.emplace(
                            object_id.user_key,
                            OffloadingTask{replica.id(),
                                           std::chrono::system_clock::now()});
                        task_created = true;
                    }
                }
            });
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
./build/mooncake-store/tests/guaranteed_offload_test \
    --gtest_filter='GuaranteedOffloadTest.Guaranteed*:GuaranteedOffloadTest.Normal*:GuaranteedOffloadTest.FlagOff*'
```
Expected: all PASS.

- [ ] **Step 5: Run the full new test file + existing offload tests (no regression)**

Run:
```bash
./build/mooncake-store/tests/guaranteed_offload_test && \
cmake --build build --target offload_on_evict_test -j"$(nproc)" && \
./build/mooncake-store/tests/offload_on_evict_test
```
Expected: both PASS (existing offload behavior unchanged — guaranteed defaults off, normal path untouched).

- [ ] **Step 6: Stage**

```bash
git add mooncake-store/src/master_service.cpp mooncake-store/tests/guaranteed_offload_test.cpp
```

---

## Task 8: `OffloadObjectHeartbeat` drains both maps + cleans both on disable

Extends the drain to return `guaranteed_offloading_objects` contents (with their `guaranteed` flag) alongside `offloading_objects`, and extends the disabled-cleanup to clear + refcnt-dec both maps.

**Files:**
- Modify: `mooncake-store/src/master_service.cpp:4690-4750`
- Modify: `mooncake-store/tests/guaranteed_offload_test.cpp` (tests)

- [ ] **Step 1: Write the failing tests**

Append to `guaranteed_offload_test.cpp`:

```cpp
// Task 8A: drain returns both guaranteed and normal tasks, each with its flag.
TEST_F(GuaranteedOffloadTest, DrainReturnsBothQueuesWithFlags) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = true;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true).has_value());

    PutObject(*service, ctx.client_id, "normal");
    PutObject(*service, ctx.client_id, "guar", /*guaranteed_until_ms=*/60000);

    auto drained = DrainOffloadQueue(*service, ctx.client_id);
    ASSERT_EQ(drained.size(), 2u);
    int guar_count = 0, normal_count = 0;
    for (const auto& t : drained) {
        if (t.key == "guar") {
            EXPECT_TRUE(t.guaranteed);
            ++guar_count;
        }
        if (t.key == "normal") {
            EXPECT_FALSE(t.guaranteed);
            ++normal_count;
        }
    }
    EXPECT_EQ(guar_count, 1);
    EXPECT_EQ(normal_count, 1);

    service->RemoveAll();
}

// Task 8B: when offloading is disabled, a queued guaranteed task is cleaned up
// (refcnt decremented, task erased) — the next enable-drain returns empty.
TEST_F(GuaranteedOffloadTest, DisableCleansGuaranteedQueue) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = true;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true).has_value());

    PutObject(*service, ctx.client_id, "guar", /*guaranteed_until_ms=*/60000);
    // Do NOT drain — leave it queued, then disable (clears + refcnt-decs).
    auto disable_res = service->OffloadObjectHeartbeat(ctx.client_id, false);
    ASSERT_TRUE(disable_res.has_value());
    EXPECT_TRUE(disable_res.value().empty());  // disabled branch returns {}

    // Re-enable: the guaranteed task was cleaned, not re-enqueued.
    auto drained = DrainOffloadQueue(*service, ctx.client_id);
    EXPECT_TRUE(drained.empty())
        << "disabled cleanup must clear the guaranteed queue";

    service->RemoveAll();
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_offload_test \
    --gtest_filter='GuaranteedOffloadTest.Drain*:GuaranteedOffloadTest.Disable*'
```
Expected: 8A may partially fail (drained size 2 might already work since both go to... no — without the drain change, guaranteed tasks are in `guaranteed_offloading_objects` which the current drain does NOT return, so 8A sees only 1 task → size 1 → FAIL). 8B fails (guaranteed task not cleaned by disable → next drain returns it, or it leaks).

- [ ] **Step 3: Extend the drain (enable) branch to return both maps**

In `mooncake-store/src/master_service.cpp`, replace the enable-drain block (lines 4708-4718):

```cpp
        if (enable_offloading) {
            std::vector<OffloadTaskItem> result;
            result.reserve(
                local_disk_segment_it->second->offloading_objects.size());
            for (const auto& [_, task] :
                 local_disk_segment_it->second->offloading_objects) {
                result.push_back(task);
            }
            local_disk_segment_it->second->offloading_objects.clear();
            return result;
        }
```

with:

```cpp
        if (enable_offloading) {
            std::vector<OffloadTaskItem> result;
            result.reserve(
                local_disk_segment_it->second->offloading_objects.size() +
                local_disk_segment_it->second->guaranteed_offloading_objects
                    .size());
            for (const auto& [_, task] :
                 local_disk_segment_it->second->offloading_objects) {
                result.push_back(task);
            }
            for (const auto& [_, task] :
                 local_disk_segment_it->second->guaranteed_offloading_objects) {
                result.push_back(task);
            }
            local_disk_segment_it->second->offloading_objects.clear();
            local_disk_segment_it->second->guaranteed_offloading_objects.clear();
            return result;
        }
```

- [ ] **Step 4: Extend the disable-cleanup to move + clean both maps**

In the same function, replace the disable branch (lines 4728-4748):

```cpp
        offloading_objects_copy =
            std::move(local_disk_segment_it->second->offloading_objects);
    }

    for (auto& [_, task] : offloading_objects_copy) {
        const auto object_id = MakeObjectIdentity(task.key, task.tenant_id);
        MetadataAccessorRW accessor(this, object_id);
        if (accessor.Exists()) {
            auto& tenant_state = accessor.GetTenantState();
            auto task_it =
                tenant_state.offloading_tasks.find(object_id.user_key);
            if (task_it != tenant_state.offloading_tasks.end()) {
                auto source =
                    accessor.Get().GetReplicaByID(task_it->second.source_id);
                if (source) {
                    source->dec_refcnt();
                }
                tenant_state.offloading_tasks.erase(task_it);
            }
        }
    }
    return {};
```

with (add a second copy map + a second loop):

```cpp
        offloading_objects_copy =
            std::move(local_disk_segment_it->second->offloading_objects);
        guaranteed_objects_copy =
            std::move(local_disk_segment_it->second->guaranteed_offloading_objects);
    }

    auto cleanup_copied =
        [this](std::unordered_map<std::string, OffloadTaskItem>& copy) {
            for (auto& [_, task] : copy) {
                const auto object_id =
                    MakeObjectIdentity(task.key, task.tenant_id);
                MetadataAccessorRW accessor(this, object_id);
                if (accessor.Exists()) {
                    auto& tenant_state = accessor.GetTenantState();
                    auto task_it = tenant_state.offloading_tasks.find(
                        object_id.user_key);
                    if (task_it != tenant_state.offloading_tasks.end()) {
                        auto source = accessor.Get().GetReplicaByID(
                            task_it->second.source_id);
                        if (source) {
                            source->dec_refcnt();
                        }
                        tenant_state.offloading_tasks.erase(task_it);
                    }
                }
            }
        };
    cleanup_copied(offloading_objects_copy);
    cleanup_copied(guaranteed_objects_copy);
    return {};
```

And add the second copy map declaration near line 4704. Find:

```cpp
    std::unordered_map<std::string, OffloadTaskItem> offloading_objects_copy;
```

and change to:

```cpp
    std::unordered_map<std::string, OffloadTaskItem> offloading_objects_copy;
    std::unordered_map<std::string, OffloadTaskItem> guaranteed_objects_copy;
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_offload_test \
    --gtest_filter='GuaranteedOffloadTest.Drain*:GuaranteedOffloadTest.Disable*'
```
Expected: PASS.

- [ ] **Step 6: Run full new test suite + existing offload tests**

Run:
```bash
./build/mooncake-store/tests/guaranteed_offload_test && \
./build/mooncake-store/tests/offload_on_evict_test
```
Expected: both PASS.

- [ ] **Step 7: Stage**

```bash
git add mooncake-store/src/master_service.cpp mooncake-store/tests/guaranteed_offload_test.cpp
```

---

## Task 9: Re-enqueue guaranteed objects on SSD-write NACK

On NACK (`metadata.data_size < 0`), a guaranteed object is re-enqueued into the independent queue (pin retained — no `dec_refcnt`), refreshing `start_time`, to be retried on the next drain. Non-guaranteed keeps the existing dec/erase path.

**Files:**
- Modify: `mooncake-store/src/master_service.cpp:4813-4829`
- Modify: `mooncake-store/tests/guaranteed_offload_test.cpp` (test)

- [ ] **Step 1: Write the failing test**

Append to `guaranteed_offload_test.cpp`:

```cpp
// Task 9: on SSD-write NACK, a guaranteed object is re-enqueued (retry).
TEST_F(GuaranteedOffloadTest, GuaranteedReenqueuedOnNack) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = true;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true).has_value());

    PutObject(*service, ctx.client_id, "guar", /*guaranteed_until_ms=*/60000);

    // Drain: master hands the task to the client (simulated).
    auto first_drain = DrainOffloadQueue(*service, ctx.client_id);
    ASSERT_EQ(first_drain.size(), 1u);
    ASSERT_EQ(first_drain[0].key, "guar");

    // Simulate an SSD-write failure: client NACKs via NotifyOffloadSuccess.
    std::vector<OffloadTaskItem> tasks = first_drain;
    std::vector<StorageObjectMetadata> metas(tasks.size());
    metas[0].data_size = -1;  // NACK sentinel
    auto nack_res = service->NotifyOffloadSuccess(ctx.client_id, tasks, metas);
    ASSERT_TRUE(nack_res.has_value());

    // The guaranteed object must have been re-enqueued for retry.
    auto second_drain = DrainOffloadQueue(*service, ctx.client_id);
    bool reenqueued = false;
    for (const auto& t : second_drain) {
        if (t.key == "guar") reenqueued = true;
    }
    EXPECT_TRUE(reenqueued)
        << "guaranteed object must be re-enqueued on NACK for retry";

    service->RemoveAll();
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_offload_test \
    --gtest_filter=GuaranteedOffloadTest.GuaranteedReenqueuedOnNack
```
Expected: FAIL — current NACK branch does dec_refcnt + erase, does not re-enqueue; `second_drain` is empty.

- [ ] **Step 3: Add the guaranteed re-enqueue in the NACK branch**

In `mooncake-store/src/master_service.cpp`, replace the NACK branch (lines 4813-4829):

```cpp
        if (metadata.data_size < 0) {
            std::shared_lock<std::shared_mutex> shared_lock(snapshot_mutex_);
            MetadataAccessorRW accessor(this, request_object_id);
            if (accessor.Exists()) {
                auto& tenant_state = accessor.GetTenantState();
                auto task_it = tenant_state.offloading_tasks.find(
                    request_object_id.user_key);
                if (task_it != tenant_state.offloading_tasks.end()) {
                    auto source = accessor.Get().GetReplicaByID(
                        task_it->second.source_id);
                    if (source != nullptr) {
                        source->dec_refcnt();
                    }
                    tenant_state.offloading_tasks.erase(task_it);
                }
            }
            continue;
        }
```

with (guaranteed re-enqueue before the existing dec/erase):

```cpp
        if (metadata.data_size < 0) {
            std::shared_lock<std::shared_mutex> shared_lock(snapshot_mutex_);
            MetadataAccessorRW accessor(this, request_object_id);
            if (accessor.Exists()) {
                auto& obj_metadata = accessor.Get();
                auto& tenant_state = accessor.GetTenantState();
                auto task_it = tenant_state.offloading_tasks.find(
                    request_object_id.user_key);
                if (task_it != tenant_state.offloading_tasks.end()) {
                    auto source = accessor.Get().GetReplicaByID(
                        task_it->second.source_id);
                    if (source != nullptr && obj_metadata.guaranteed_ &&
                        enable_guaranteed_cache_) {
                        // Guaranteed: re-enqueue for the next drain and retain
                        // the pin (no dec_refcnt). The PutEnd inc_refcnt stays
                        // until SSD write eventually succeeds.
                        auto result = PushOffloadingQueue(
                            request_object_id, *source, /*guaranteed=*/true);
                        if (result || result.error() ==
                                          ErrorCode::OBJECT_ALREADY_EXISTS) {
                            // Refresh start_time to reset the offload-task TTL,
                            // preventing the reaper from erasing the task.
                            // offloading_tasks maps to `const OffloadingTask`
                            // (master_service.h:1216), so the entry cannot be
                            // mutated in place — erase and re-emplace with a
                            // fresh start_time instead.
                            const auto source_id = task_it->second.source_id;
                            tenant_state.offloading_tasks.erase(task_it);
                            tenant_state.offloading_tasks.emplace(
                                request_object_id.user_key,
                                OffloadingTask{source_id,
                                               std::chrono::system_clock::now()});
                            continue;  // pin retained; skip dec/erase
                        }
                        // Re-enqueue failed (e.g. UNABLE_OFFLOADING) -> degrade:
                        // release the pin and erase the task (existing path).
                        if (source != nullptr) {
                            source->dec_refcnt();
                        }
                        tenant_state.offloading_tasks.erase(task_it);
                        continue;
                    }
                    // Non-guaranteed: existing behavior.
                    if (source != nullptr) {
                        source->dec_refcnt();
                    }
                    tenant_state.offloading_tasks.erase(task_it);
                }
            }
            continue;
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_offload_test \
    --gtest_filter=GuaranteedOffloadTest.GuaranteedReenqueuedOnNack
```
Expected: PASS.

- [ ] **Step 5: Run the full new test suite + existing offload/promotion tests (no regression)**

Run:
```bash
cmake --build build --target mooncake_store_tests -j"$(nproc)" 2>/dev/null || \
cmake --build build --target guaranteed_offload_test offload_on_evict_test promotion_on_hit_test -j"$(nproc)"
./build/mooncake-store/tests/guaranteed_offload_test
./build/mooncake-store/tests/offload_on_evict_test
./build/mooncake-store/tests/promotion_on_hit_test
```
Expected: all PASS.

- [ ] **Step 6: Stage**

```bash
git add mooncake-store/src/master_service.cpp mooncake-store/tests/guaranteed_offload_test.cpp
```

---

## Final Verification

- [ ] **Step 1: Build all store tests**

```bash
cmake --build build --target mooncake_store_tests -j"$(nproc)" 2>/dev/null || \
cmake --build build -j"$(nproc)"
```

- [ ] **Step 2: Run the full guaranteed test file**

```bash
./build/mooncake-store/tests/guaranteed_offload_test
```
Expected: all tests PASS (12 tests).

- [ ] **Step 3: Run the broader offload/promotion regression suite**

```bash
cd build && ctest -R "offload_on_evict_test|promotion_on_hit_test|guaranteed_offload_test" -V
```
Expected: all PASS — no regression to existing offload/promotion behavior.

- [ ] **Step 4: Confirm zero behavior change with flag off**

The default `enable_guaranteed_cache=false` means: `guaranteed_` is never set (Task 4 gate), so `metadata.guaranteed_` is always false, so the PutEnd condition reduces to `!offload_on_evict_` (original) and `PushOffloadingQueue` is called with `guaranteed=false` (original). `offload_on_evict_test` passing is the proof.

- [ ] **Step 5: (Optional) Full local CI**

If `scripts/run_ci_test.sh` exists, run it per the `mooncake-ci-local` skill. Otherwise the targeted ctest run above is sufficient for Phase 1.

---

## Supplemental Tests: Cases 5 & 9 (coverage gaps)

Spec §10 case 5 (guaranteed becomes normally evictable after SSD success) and case 9 (`enable_offload=false` degrades guaranteed to normal).

- [ ] **Add `#include <algorithm>`** (for `std::any_of`) to the test file's includes.

- [ ] **Case 5: `GuaranteedBecomesEvictableAfterSsdSuccess`**

```cpp
// Task 5: after SSD write success, a guaranteed object's pin is released and a
// LOCAL_DISK replica is recorded — it has transitioned from "guaranteed, pinned,
// in-queue" to "offloaded, LOCAL_DISK present, no in-flight task". (Spec §10
// case 5: guaranteed becomes normally evictable after SSD success.)
TEST_F(GuaranteedOffloadTest, GuaranteedBecomesEvictableAfterSsdSuccess) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = true;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true).has_value());

    PutObject(*service, ctx.client_id, "guar", /*guaranteed_until_ms=*/60000);

    // Drain the guaranteed queue: master hands the task to the client.
    auto drained = DrainOffloadQueue(*service, ctx.client_id);
    ASSERT_EQ(drained.size(), 1u);
    ASSERT_EQ(drained[0].key, "guar");

    // Simulate a successful SSD write: client reports back via
    // NotifyOffloadSuccess with a non-negative data_size + transport_endpoint.
    std::vector<OffloadTaskItem> tasks = drained;
    std::vector<StorageObjectMetadata> metas(tasks.size());
    metas[0].data_size = 1024;  // success (non-negative)
    metas[0].transport_endpoint = "tcp://fake_endpoint:1234";
    auto ack_res = service->NotifyOffloadSuccess(ctx.client_id, tasks, metas);
    ASSERT_TRUE(ack_res.has_value()) << "NotifyOffloadSuccess should succeed";

    // After success: a LOCAL_DISK replica must be recorded (offload completed).
    auto replica_list = service->GetReplicaList("guar", "default");
    ASSERT_TRUE(replica_list.has_value()) << "GetReplicaList should succeed";
    bool has_local_disk = std::any_of(
        replica_list->replicas.begin(), replica_list->replicas.end(),
        [](const Replica::Descriptor& d) { return d.is_local_disk_replica(); });
    EXPECT_TRUE(has_local_disk)
        << "after SSD success, the object must have a LOCAL_DISK replica";

    // The in-flight offloading task must be cleared (no longer pinned/queued).
    auto second_drain = DrainOffloadQueue(*service, ctx.client_id);
    EXPECT_TRUE(second_drain.empty())
        << "after success, no offloading task should remain in the queue";

    service->RemoveAll();
}
```

- [ ] **Case 9: `GuaranteedDegradesWhenOffloadDisabled`**

```cpp
// Task 9 (enable_offload=false): with offload entirely disabled, a guaranteed
// Put degrades to a normal in-memory object — it succeeds without error and
// incurs no offload. (Distinct from FlagOffDegradesGuaranteedToNormal, which
// covers the enable_guaranteed_cache=false direction.)
//
// Note: with enable_offload=false, MountLocalDiskSegment is rejected
// ("offload functionality is not enabled") and there is no offload queue to
// drain — so this test deliberately does NOT mount a local disk segment or
// drain. It asserts only that the Put path succeeds (no error, no offload),
// which is the observable contract of the degrade path.
TEST_F(GuaranteedOffloadTest, GuaranteedDegradesWhenOffloadDisabled) {
    MasterServiceConfig config;
    config.enable_offload = false;  // offload entirely off
    config.enable_guaranteed_cache = true;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);

    // A guaranteed_until_ms>0 Put must succeed (no error) even though offload is
    // disabled — it just degrades to a normal in-memory object. PutObject's
    // internal PutStart/PutEnd ASSERTs enforce "no error".
    PutObject(*service, ctx.client_id, "guar", /*guaranteed_until_ms=*/60000);

    // Confirm the object exists as a normal in-memory object (offload never ran).
    auto replica_list = service->GetReplicaList("guar", "default");
    ASSERT_TRUE(replica_list.has_value()) << "GetReplicaList should succeed";
    bool has_memory = std::any_of(
        replica_list->replicas.begin(), replica_list->replicas.end(),
        [](const Replica::Descriptor& d) { return d.is_memory_replica(); });
    EXPECT_TRUE(has_memory)
        << "with enable_offload=false, the object should be a plain in-memory "
           "object (no LOCAL_DISK replica, since offload is disabled)";

    service->RemoveAll();
}
```

- [ ] **Stage**

```bash
git add mooncake-store/tests/guaranteed_offload_test.cpp
```

---

## Notes

- **`guaranteed_` is const** (immutable after construction), threaded via a trailing defaulted ctor param so the HA-deserialize call site needs no change (defaults false = reset on restart, per spec §7.10).
- **No capacity limit** on the guaranteed queue — it is implicitly bounded by in-memory guaranteed objects awaiting offload (each holds a pinned memory replica).
- **Retry has no cap** — guaranteed means "must be written"; persistent SSD failure pins memory until ops intervenes (spec §7).
- **Client-side `BuildBucket` splitting** (spec §6.4, test case 10) is deferred to Phase 2 — it has no observable effect without Phase 2's bucket-level pin and lives in a different module. The master already propagates `OffloadTaskItem.guaranteed` to the client.

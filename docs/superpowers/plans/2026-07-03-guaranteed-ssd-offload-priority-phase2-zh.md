# Guaranteed SSD Offload 优先级 — Phase 2 实施计划（中文）

> **状态：✅ 已实现并验证（2026-07-06）。** 4 个任务全部完成，4 个测试通过。仅 client 侧：`BucketMetadata.guaranteed` + `YLT_REFL`（重启不丢）+ `OffloadObjects` 分同质 bucket + `SelectEvictionCandidate` 跳过 guaranteed bucket（FIFO 前向扫描；LRU 前向扫描不 erase）。已应用编译 bug 修复（漏改 DistributedStorageBackend override；`return res.error()`→`tl::make_unexpected`；NoopComplete 改自由函数）。参考 spec §6.4、§11 Phase 2。

> **给执行 agent：** 必用子技能：superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans，逐任务执行。步骤用 checkbox（`- [ ]`）跟踪。

**目标：** 保护 guaranteed 对象的 SSD 副本不被 client 侧 fifo/lru 驱逐 —— 一旦写入 SSD（Phase 1），guaranteed bucket 永不被选为驱逐候选。这是 SSD 管理生命周期设计的第二个 slice（spec §11 Phase 2）。

**架构：** 仅 client 侧（`storage_backend.cpp` + `file_storage.cpp`）。Phase 1 把 `OffloadTaskItem.guaranteed` 从 master 传到 client，但 client 当前在 `OffloadObjects` 处**丢弃它**（只建 `map<string,int64_t>` 的 key+size）。Phase 2 (a) 把 `guaranteed` 从 `OffloadObjects` 穿到 `BucketMetadata`，通过把 guaranteed/normal task 分成**同质 bucket**（一个 bucket 全是 guaranteed 或全是 normal）；(b) 让 `SelectEvictionCandidate` 跳过 guaranteed bucket。驱逐有**单一 chokepoint**（`PrepareEviction`→`SelectEvictionCandidate`），纯触发式于 `BatchOffload` 内 —— 无后台扫描。

**技术栈：** C++17、GoogleTest、struct_pack（`YLT_REFL`）、Mooncake Store client storage backend。

**参考 spec：** `docs/superpowers/specs/2026-07-02-guaranteed-ssd-offload-priority-design-zh.md`（§6.4、§11 Phase 2）。

**Git 约定：** 每个任务 `git add`（不 commit），按用户偏好。末尾做最终审查。

---

## 范围

Phase 2 = client 侧 SSD 副本驱逐保护。master 侧不变（Phase 1 已传标记）。

**范围内：**
- `OffloadObjects` 按 `guaranteed` 分组 → 同质 bucket。
- `BucketMetadata.guaranteed` 字段（经 `YLT_REFL` 持久化，`BatchLoad` 重启不丢）。
- `BuildBucket` 设 `bucket->guaranteed`。
- `SelectEvictionCandidate` 跳过 guaranteed bucket（FIFO：前向扫描；LRU：不 erase 跳过）。

**推迟到 Phase 3：** SSD 副本的 TTL（Phase 2 guaranteed bucket 永久保护；Phase 3 TTL 过期后可驱逐）。接受失败模式：guaranteed bucket 占满磁盘 → `WriteBucket` ENOSPC 失败下次 offload（backpressure）—— Phase 3 TTL 缓解。

**本机 build 环境可能坏**（缺 `libmsgpack-dev` + yalantinglibs include 未传播；`dependencies.sh` 未跑）。若 `cmake --build` 报 `ylt/util/expected.hpp`，是环境问题 —— 用户在 build 环境验证。写正确代码 + 读代码自审。

---

## 文件结构

| 文件 | 职责 | 任务 |
|------|------|------|
| `mooncake-store/include/storage_backend.h` | `BucketMetadata.guaranteed` 字段 + 4 个 ctor/assignment 更新 + `YLT_REFL` | 1 |
| `mooncake-store/src/file_storage.cpp` | `OffloadObjects` 按 guaranteed 分组 | 2 |
| `mooncake-store/src/storage_backend.cpp` | `BuildBucket` 设 guaranteed；`SelectEvictionCandidate` 跳过（FIFO+LRU） | 3, 4 |
| `mooncake-store/tests/guaranteed_eviction_test.cpp` | 新测试文件 | 1, 4 |
| `mooncake-store/tests/CMakeLists.txt` | 注册测试 target | 1 |

---

## Task 1: 给 `BucketMetadata` 加 `guaranteed` 字段（+ 测试 scaffold + CMake）

**文件：**
- 修改：`mooncake-store/include/storage_backend.h:33-91`（BucketMetadata）
- 新建：`mooncake-store/tests/guaranteed_eviction_test.cpp`
- 修改：`mooncake-store/tests/CMakeLists.txt:46`

- [ ] **Step 1: 在 CMake 注册新测试 target**

`mooncake-store/tests/CMakeLists.txt`，`add_store_test(guaranteed_offload_test guaranteed_offload_test.cpp)` 之后加：

```cmake
add_store_test(guaranteed_eviction_test guaranteed_eviction_test.cpp)
```

- [ ] **Step 2: 写失败测试（文件 scaffold + 字段默认值测试）**

新建 `mooncake-store/tests/guaranteed_eviction_test.cpp`。本任务只加 scaffold + 字段测试；驱逐测试在 Task 4。镜像 `storage_backend_test.cpp` 的 fixture 模式：

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

- [ ] **Step 3: 跑测试确认失败**

```bash
cmake --build build --target guaranteed_eviction_test -j"$(nproc)" 2>&1 | tail -20
```
预期：**编译失败** —— `BucketMetadata` 无 `guaranteed` 字段。（若报 `ylt/util/expected.hpp`，是本机环境问题 —— 记录并继续，读代码验证正确性。）

- [ ] **Step 4: 给 `BucketMetadata` 加 `guaranteed` 字段**

`mooncake-store/include/storage_backend.h`，`struct BucketMetadata`（line 33-91）：

line 37（`std::vector<BucketObjectMetadata> metadatas;`）之后加：

```cpp
    bool guaranteed{false};  // true => bucket is never selected for eviction
                             // (Phase 2: set when all keys are guaranteed).
```

**copy constructor**（line 50-56）初始化列表加 `guaranteed(other.guaranteed),`（`inflight_reads_(0),` 之前）：

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

**move constructor**（line 59-65）加 `guaranteed(other.guaranteed),`：

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

**copy assignment**（line 68-77）`if` 内加 `guaranteed = other.guaranteed;`：

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

**move assignment**（line 80-89）加 `guaranteed = other.guaranteed;`：

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

`YLT_REFL`（line 91）加 `guaranteed`（持久化到 `.meta` 文件，`BatchLoad` 重启不丢）：

```cpp
YLT_REFL(BucketMetadata, data_size, keys, metadatas, guaranteed);
```

- [ ] **Step 5: 跑测试确认通过**

```bash
cmake --build build --target guaranteed_eviction_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_eviction_test \
    --gtest_filter=GuaranteedEvictionTest.BucketMetadataDefaultsNonGuaranteed
```
预期：PASS。

- [ ] **Step 6: 暂存**

```bash
git add mooncake-store/include/storage_backend.h mooncake-store/tests/guaranteed_eviction_test.cpp mooncake-store/tests/CMakeLists.txt
```

---

## Task 2: `OffloadObjects` 按 `guaranteed` 分成同质 bucket

核心穿线修复：Phase 1 的 `guaranteed` 标记在 `OffloadObjects`（line 373 建 `map<string,int64_t>`）处被丢弃。把 task 分成 guaranteed/normal 两组，各自独立组桶 + offload，使 bucket 同质。

**文件：**
- 修改：`mooncake-store/src/file_storage.cpp:361-491`（OffloadObjects）

- [ ] **Step 1: 读当前 `OffloadObjects` 确认确切文本**

读 `mooncake-store/src/file_storage.cpp` line 361-491。当前流程（已验证）：
- line 366-375：建 `storage_object_sizes`（`map<string,int64_t>`，丢 `guaranteed`）+ `task_by_storage_key`（保留完整 task）。
- line 377-394：`AllocateOffloadingBuckets(storage_object_sizes, buckets_keys)`（或 fallback 单 bucket）。
- line 429-491：按 `buckets_keys` 组，建 `batch_object`、定义 `eviction_handler`、调 `storage_backend_->BatchOffload(...)`。

改动：不再用一个 `storage_object_sizes` map + 一次 `AllocateOffloadingBuckets`，而是分两个 map（`guaranteed_sizes` / `normal_sizes`），每组各调 `AllocateOffloadingBuckets`，并把 `guaranteed` 标记传给 `BatchOffload`（Task 3 加该参数）。因 `BatchOffload` 的 guaranteed 参数在 Task 3 落地，**本任务只做分组 + 给每组标 guaranteed bool** —— 实际的 `BatchOffload(..., guaranteed)` 调用在 Task 3 接通（或本任务加，若 Task 3 先合并；协调确保参数已存在）。

- [ ] **Step 2: 重构 `OffloadObjects` 按 guaranteed 分组**

`mooncake-store/src/file_storage.cpp`，把 line 366（`std::unordered_map<std::string, int64_t> storage_object_sizes;`）到 `AllocateOffloadingBuckets`/fallback 块末尾（~line 394）替换为分组版本。先读确切当前代码，然后：

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

然后把 per-bucket 循环（line 429-491）从遍历 `buckets_keys` 改为遍历 `groups`，并把 `group.guaranteed` 传给 `BatchOffload`。循环体（从 keys 建 `batch_object`、定义 `eviction_handler`、调 `storage_backend_->BatchOffload`）不变；只改外层迭代，从：

```cpp
    for (const auto& keys : buckets_keys) {
```

改为：

```cpp
    for (const auto& group : groups) {
        const bool group_guaranteed = group.guaranteed;
        for (const auto& keys : group.buckets_keys) {
            // ... existing body ...
            auto result = storage_backend_->BatchOffload(
                batch_object, complete_handler, eviction_handler,
                group_guaranteed);  // <-- 新第 4 参（Task 3 加该参数）
            // ...
        }
    }
```

**与 Task 3 协调：** `BatchOffload(..., group_guaranteed)` 第 4 参数需 Task 3 先给 `BucketStorageBackend::BatchOffload` 加 `bool guaranteed = false` 参数。建议顺序：先 Task 3（加参数 + 穿到 BuildBucket），再 Task 2（使用它）。

- [ ] **Step 3:（环境坏则跳过）构建 + 验证既有 file_storage 测试不回归**

```bash
cmake --build build --target file_storage_test -j"$(nproc)" && \
./build/mooncake-store/tests/file_storage_test
```
预期：既有测试通过（normal offload 路径行为不变 —— 无 guaranteed task 时只分一个 normal 组）。

- [ ] **Step 4: 暂存**

```bash
git add mooncake-store/src/file_storage.cpp
```

---

## Task 3: `BucketStorageBackend::BatchOffload` + `BuildBucket` 把 `guaranteed` 穿到 `BucketMetadata`

给 `BatchOffload` 加尾随 `bool guaranteed = false` 参数（默认 false → 既有调用方不变），传给 `BuildBucket`，在那里设 `bucket->guaranteed`。

**文件：**
- 修改：`mooncake-store/include/storage_backend.h`（`BucketStorageBackend::BatchOffload` 声明 ~626）
- 修改：`mooncake-store/src/storage_backend.cpp`（`BatchOffload` 定义 ~1279、`BuildBucket` 定义 ~1978）

- [ ] **Step 1: 给 `BatchOffload` 声明加 `guaranteed` 参数（base + override）**

基接口 `StorageBackendInterface::BatchOffload`（`mooncake-store/include/storage_backend.h:255`）是**纯虚**，且 `FileStorage::storage_backend_` 类型是 `std::shared_ptr<StorageBackendInterface>`（`file_storage.h:133`）。所以 `OffloadObjects` 的 `storage_backend_->BatchOffload(..., guaranteed)` 解析到**基类**签名 —— 基类必须加该参数，`BucketStorageBackend` override 也要。

`mooncake-store/include/storage_backend.h`，基接口（~line 255）：

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

`BucketStorageBackend` override（~line 626）加同样尾随参数：

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

检查**其他** `StorageBackendInterface` 子类的 `BatchOffload` override（grep `BatchOffload` in `storage_backend.h` —— 如 `FilePerKey`/legacy backend ~line 753/1036）。每个 override 都要加尾随 `bool guaranteed = false` 参数（即使忽略它 —— legacy backend 当非 guaranteed 处理）。给 header 里每个 `BatchOffload` override 加 `bool guaranteed = false`；`.cpp` 定义里接受并忽略（或透传）。保持基类契约一致。

- [ ] **Step 2: 更新 `BatchOffload` 定义 + 传给 `BuildBucket`**

`mooncake-store/src/storage_backend.cpp`（定义 ~1279），签名改为接受 `bool guaranteed`（定义里无默认值）：

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

然后改 `BuildBucket` 调用（~line 1306-1307）：

```cpp
    auto build_bucket_result =
        BuildBucket(bucket_id, batch_object, iovs, metadatas, guaranteed);
```

- [ ] **Step 3: 更新 `BuildBucket` 设 `bucket->guaranteed`**

`mooncake-store/src/storage_backend.cpp`，`BuildBucket` 定义（~1977），加尾随 `bool guaranteed` 参数并设字段。改签名：

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
    // ... 其余不变 ...
```

同时更新 `mooncake-store/include/storage_backend.h` 里 `BuildBucket` **声明**（在 private 段 grep `BuildBucket`）加 `bool guaranteed = false`。

- [ ] **Step 4:（环境坏则跳过）构建验证编译**

```bash
cmake --build build --target storage_backend_test -j"$(nproc)" 2>&1 | tail -15
```
预期：编译通过。既有调用方省略 `guaranteed`（默认 false）→ 非 guaranteed bucket → 行为不变。

- [ ] **Step 5: 暂存**

```bash
git add mooncake-store/include/storage_backend.h mooncake-store/src/storage_backend.cpp
```

---

## Task 4: `SelectEvictionCandidate` 跳过 guaranteed bucket（FIFO + LRU）

核心保护。FIFO：从 `begin()` 前向扫描到第一个非 guaranteed。LRU：在 `lru_index_` 里跳过 guaranteed 项**不 erase**（erase 会永久丢失）。

**文件：**
- 修改：`mooncake-store/src/storage_backend.cpp:2180-2225`（SelectEvictionCandidate）
- 修改：`mooncake-store/tests/guaranteed_eviction_test.cpp`（驱逐测试）

- [ ] **Step 1: 写失败测试**

追加到 `mooncake-store/tests/guaranteed_eviction_test.cpp`（`}  // namespace mooncake::test` 之前）：

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

**注意：** 用前先读 `storage_backend.h` 确认 `BucketStorageBackend` 有 public `Init()` / `IsExist(key)` 方法。`Cleanup()` 不存在 —— 栈对象靠析构清理（不调），TearDown 清临时目录。`IsExist` 返回 `tl::expected<bool, ErrorCode>`（`storage_backend_test.cpp` 用法已确认）。

- [ ] **Step 2: 跑测试确认失败**

```bash
cmake --build build --target guaranteed_eviction_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_eviction_test \
    --gtest_filter='GuaranteedEvictionTest.Fifo*'
```
预期：4A FAIL（guaranteed bucket 被驱逐 —— 还没跳过逻辑）；4B 可能过（既有 FIFO 驱逐 normal）。

- [ ] **Step 3: 实现 FIFO 跳过**

`mooncake-store/src/storage_backend.cpp`，`SelectEvictionCandidate`（~2181），FIFO case 从：

```cpp
        case BucketEvictionPolicy::FIFO:
            // buckets_ is ordered by bucket_id (monotonically increasing),
            // so begin() is always the oldest bucket.
            return buckets_.begin();
```

改为前向扫描跳过 guaranteed：

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

- [ ] **Step 4: 实现 LRU 跳过（不 erase）**

同函数，LRU case（~2200-2220）。guaranteed 项必须**跳过不 erase**（erase 会永久丢失——读路径不重插——故 bucket 即使 Phase 3 TTL 过期后也无法驱逐）。

**重要（死循环修复）：** 朴素的 `if (guaranteed) { ++top_it; continue; }` 会**死循环**——每次 `while` 迭代重置 `auto top_it = lru_index_.begin()`，若 `begin()` 是 guaranteed，`continue` 后又回到同一个 guaranteed `begin()`，永远不前进。改为从 `top_it` **前向扫描**找第一个非 guaranteed、仍存在的项。LRU 循环体替换为：

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

**关键：** guaranteed 项**永不 erase**（`chosen_it` 按构造必非 guaranteed；guaranteed 的 `top_it` 只读）。前向扫描保证终止（每次外层 while 要么 return、要么 repair 一个 stale 项、要么 return `end()`）。非 guaranteed 顶部项的 stale-repair（erase+emplace）保持不变。

- [ ] **Step 5: 跑测试确认通过**

```bash
cmake --build build --target guaranteed_eviction_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_eviction_test \
    --gtest_filter='GuaranteedEvictionTest.Fifo*'
```
预期：PASS。

- [ ] **Step 6: 加 LRU 跳过测试 + 跑全量**

追加到 `guaranteed_eviction_test.cpp`：

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

跑：
```bash
cmake --build build --target guaranteed_eviction_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_eviction_test
```
预期：全 PASS。用前读 `storage_backend.h` 确认 `BatchQuery` 签名（接 `vector<string>` keys + out-param `map<string,StorageObjectMetadata>`；返回 `tl::expected`）。

- [ ] **Step 7: 跑既有 storage_backend/file_storage 测试（无回归）**

```bash
cmake --build build --target storage_backend_test file_storage_test -j"$(nproc)" && \
./build/mooncake-store/tests/storage_backend_test && \
./build/mooncake-store/tests/file_storage_test
```
预期：PASS —— normal 驱逐行为不变（默认 `guaranteed=false` → 不跳过）。

- [ ] **Step 8: 暂存**

```bash
git add mooncake-store/src/storage_backend.cpp mooncake-store/tests/guaranteed_eviction_test.cpp
```

---

## 最终验证

- [ ] **Step 1: 构建所有受影响测试 target**

```bash
cmake --build build --target guaranteed_eviction_test storage_backend_test file_storage_test guaranteed_offload_test -j"$(nproc)"
```

- [ ] **Step 2: 跑全量新测试 + 回归套件**

```bash
./build/mooncake-store/tests/guaranteed_eviction_test
./build/mooncake-store/tests/storage_backend_test
./build/mooncake-store/tests/file_storage_test
./build/mooncake-store/tests/guaranteed_offload_test
```
预期：全 PASS。

- [ ] **Step 3: 确认对 Phase 1 零回归**

Phase 1 的 `guaranteed_offload_test` 必须仍原样通过（Phase 2 是 client 侧；master 侧 Phase 1 行为未动）。

- [ ] **Step 4: 端到端（可选，若有真实 master+client harness）**

若 Python e2e 测试能以 `--enable_guaranteed_cache=true --enable_offload=true --offload_on_evict=true` 跑 master + client，验证 guaranteed 对象在 SSD 上扛过内存驱逐压力。需全栈；仅单测时推迟。

---

## 备注

- **`guaranteed` 重启不丢**：`BucketMetadata.guaranteed` 在 `YLT_REFL` 里，`BatchLoad` 重启时重新标记受保护 bucket。不加则重启丢保护。
- **LRU 跳过不能 erase（且要前向扫描，不是 `++top_it`）**：从 `lru_index_` erase 一个 guaranteed 项是永久的（读路径不重插），故 bucket 即使 Phase 3 TTL 过期后也无法驱逐。跳过用前向扫描找第一个非 guaranteed 项（**不是** `++top_it; continue;`——那会死循环，因 `while` 每次重置 `top_it = begin()`，guaranteed 的 `begin()` 会被无限重访）。
- **接受失败模式**：guaranteed bucket 占满磁盘 → `WriteBucket` ENOSPC → offload 失败（backpressure）。Phase 3 TTL 缓解。Phase 2 不做 backpressure 队列（YAGNI）。
- **Phase 1 build 环境说明适用**：若 `cmake --build` 报 `ylt/util/expected.hpp`，是本机环境问题（`dependencies.sh` 未跑）。读代码验证；用户跑实际编译/测试。

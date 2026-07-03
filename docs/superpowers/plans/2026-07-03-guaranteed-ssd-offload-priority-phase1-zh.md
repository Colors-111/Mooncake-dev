# Guaranteed SSD Offload 优先级 — Phase 1 实施计划（中文）

> **给执行 agent：** 必用子技能：superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans，逐任务执行。步骤用 checkbox（`- [ ]`）跟踪。

**目标：** 确保 `guaranteed` 对象一定写入 SSD —— 不因 offload 队列满被拒（独立 per-client 队列、无 limit），不因 SSD 写失败被放弃（重试）—— 全程由 `enable_guaranteed_cache` flag 门控（默认 false）。

**架构：** 仅 master 侧。`guaranteed` 布尔沿 ReplicateConfig → ObjectMetadata → PushOffloadingQueue 流转，把 guaranteed 对象路由进独立的 per-client `guaranteed_offloading_objects` map（与现有 `promotion_objects` 先例并列，无 size limit）。`PutEnd` 无视 `offload_on_evict_` 总 offload guaranteed 对象。`OffloadObjectHeartbeat` drain 两个 map。`NotifyOffloadSuccess` 在 NACK 时重新入队 guaranteed 对象。全部由 `enable_guaranteed_cache_`（默认 false → 零行为变化）门控。

**技术栈：** C++17、GoogleTest、glog、struct_pack（`YLT_REFL`）、Mooncake Store master service。

**参考 spec：** `docs/superpowers/specs/2026-07-02-guaranteed-ssd-offload-priority-design-zh.md`（§4–§14）。

**Git 约定：** 用户偏好逐步 staging 而非逐步 commit。每个任务以 `git add` 结尾（非 `git commit`）；任务末尾或交由用户统一 commit。按用户偏好调整。

---

## 范围

本计划实现 **Phase 1（确保写入 SSD）** —— 仅 master 侧。覆盖 spec §4–§10、测试用例 1–9。

**推迟到 Phase 2**（已告知用户）：client 侧 `BuildBucket` 按 `OffloadTaskItem.guaranteed` 分流组桶（spec §6.4、测试用例 10）。理由：没有 Phase 2 的 client 侧 bucket 级 pin（驱逐保护），分流无可观察效果；且它在 `storage_backend.cpp`（不同模块）。master 已通过 `OffloadTaskItem`（Task 1）和 `OffloadObjectHeartbeat`（Task 8）把 `guaranteed` 标记传给 client，Phase 2 在 `BuildBucket` + `SelectEvictionCandidate` 消费即可。

---

## 文件结构

| 文件 | 职责 | 任务 |
|------|------|------|
| `mooncake-store/include/types.h` | `OffloadTaskItem.guaranteed` 线上字段 | 1 |
| `mooncake-store/include/replica.h` | `ReplicateConfig.guaranteed_until_ms` 标记 | 2 |
| `mooncake-store/src/master.cpp` | gflag `enable_guaranteed_cache` + config 读取 + 日志 | 3 |
| `mooncake-store/include/master_config.h` | config 结构 + copy 块 | 3 |
| `mooncake-store/include/master_service.h` | `enable_guaranteed_cache_` 成员；`ObjectMetadata.guaranteed_`；`PushOffloadingQueue` 声明 | 3, 4, 6 |
| `mooncake-store/src/master_service.cpp` | flag 初始化；标记；PutEnd；PushOffloadingQueue；heartbeat drain；NACK 重试 | 3, 4, 6, 7, 8, 9 |
| `mooncake-store/include/segment.h` | `LocalDiskSegment.guaranteed_offloading_objects` map | 5 |
| `mooncake-store/tests/guaranteed_offload_test.cpp` | 新测试文件 | 1, 7, 8, 9 |
| `mooncake-store/tests/CMakeLists.txt` | 注册新测试 target | 1 |

---

## Task 1: 给 `OffloadTaskItem` 加 `guaranteed` 字段

**文件：**
- 修改：`mooncake-store/include/types.h:263-273`
- 新建：`mooncake-store/tests/guaranteed_offload_test.cpp`
- 修改：`mooncake-store/tests/CMakeLists.txt:46`

- [ ] **Step 1: 在 CMake 注册新测试 target**

在 `mooncake-store/tests/CMakeLists.txt`，line 46（`add_store_test(offload_on_evict_test offload_on_evict_test.cpp)`）之后加：

```cmake
add_store_test(guaranteed_offload_test guaranteed_offload_test.cpp)
```

- [ ] **Step 2: 写失败测试（文件 scaffold + 等值/标记测试）**

新建 `mooncake-store/tests/guaranteed_offload_test.cpp`，fixture 改编自 `offload_on_evict_test.cpp:1-120`，含第一个测试：

```cpp
#include "master_service.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <thread>

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

    // NOTE: PutObject / DrainOffloadQueue helpers are added in Task 7, when
    // ReplicateConfig::guaranteed_until_ms and the offload-gating logic land.
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

选择（b）：保持 Task 1 自包含 —— 不引用尚不存在的 `ReplicateConfig::guaranteed_until_ms` 或 `MasterServiceConfig::enable_guaranteed_cache`。`PutObject`/`DrainOffloadQueue` helper 推迟到 Task 7 再加。目标：Task 1 后 `guaranteed_offload_test` 能编译、等值测试通过。

- [ ] **Step 3: 跑测试确认失败**

```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" 2>&1 | tail -20
```
预期：**编译失败** —— `OffloadTaskItem` 无 `.guaranteed` 字段。

- [ ] **Step 4: 给 `OffloadTaskItem` 加 `guaranteed` 字段**

在 `mooncake-store/include/types.h:263-273`，把：

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

替换为：

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

- [ ] **Step 5: 跑测试确认通过**

```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_offload_test \
    --gtest_filter=GuaranteedOffloadTest.OffloadTaskItemCarriesGuaranteedFlag
```
预期：PASS。

- [ ] **Step 6: 暂存**

```bash
git add mooncake-store/include/types.h mooncake-store/tests/guaranteed_offload_test.cpp mooncake-store/tests/CMakeLists.txt
```

---

## Task 2: 给 `ReplicateConfig` 加 `guaranteed_until_ms`

**文件：**
- 修改：`mooncake-store/include/replica.h:81-144`

- [ ] **Step 1: 写失败测试**

在 `mooncake-store/tests/guaranteed_offload_test.cpp` 末尾、`}  // namespace mooncake::test` 之前追加：

```cpp
// Task 2: ReplicateConfig carries guaranteed_until_ms (Phase 1: only >0 is checked).
TEST_F(GuaranteedOffloadTest, ReplicateConfigCarriesGuaranteedUntilMs) {
    ReplicateConfig config;
    EXPECT_EQ(config.guaranteed_until_ms, 0);  // default: no guarantee
    config.guaranteed_until_ms = 60000;
    EXPECT_GT(config.guaranteed_until_ms, 0);
}
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" 2>&1 | tail -10
```
预期：**编译失败** —— `ReplicateConfig` 无 `guaranteed_until_ms` 成员。

- [ ] **Step 3: 加字段**

在 `mooncake-store/include/replica.h`，line 97（`std::optional<std::vector<std::string>> group_ids{};`）之后、line 99（`ReplicateConfig ForSingleKey...`）之前插入：

```cpp
    // Guaranteed offload: when >0, this object's SSD offload is mandatory
    // (routed to the independent guaranteed queue, retried on failure).
    // Phase 1 treats this as a boolean marker (>0 => guaranteed).
    int64_t guaranteed_until_ms{0};
```

- [ ] **Step 4: 更新 `operator<<`（日志含此字段）**

同文件，`operator<<` 内（约 line 132，`group_ids` 块之后、`os << " }";` 之前）加：

```cpp
        os << ", guaranteed_until_ms: " << config.guaranteed_until_ms;
```

放在 `if` 块外，使其总是打印（和 `data_type` 一样）。

- [ ] **Step 5: 跑测试确认通过**

```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_offload_test \
    --gtest_filter=GuaranteedOffloadTest.ReplicateConfigCarriesGuaranteedUntilMs
```
预期：PASS。

- [ ] **Step 6: 暂存**

```bash
git add mooncake-store/include/replica.h mooncake-store/tests/guaranteed_offload_test.cpp
```

---

## Task 3: 布线 `enable_guaranteed_cache` config flag（默认 false）

基础设施（无独立行为；门控在 Task 7 验证）。镜像 `enable_offload` 贯穿 config 层。测试直接构造 `MasterServiceConfig`，所以**测试必需**部分是 `MasterServiceConfig` 字段 + `MasterService` 成员 + 初始化器（Step 3-5）；**生产 CLI** 部分（gflag、`MasterConfig`、`Wrapped`、copy 块、log）镜像 `enable_offload`（Step 7-8）。

**文件：**
- 修改：`mooncake-store/include/master_config.h`（MasterServiceConfig ~974、MasterConfig ~53、Wrapped + copy 块、builder、InProc）
- 修改：`mooncake-store/include/master_service.h:1902`（成员）
- 修改：`mooncake-store/src/master_service.cpp:205`（初始化器）
- 修改：`mooncake-store/src/master.cpp`（gflag、GetBool、override、log）
- 修改：`mooncake-store/tests/guaranteed_offload_test.cpp`（测试）

- [ ] **Step 1: 写失败测试（config 字段默认值 + 可赋值）**

追加到 `guaranteed_offload_test.cpp`：

```cpp
// Task 3: enable_guaranteed_cache defaults false and is settable on MasterServiceConfig.
TEST_F(GuaranteedOffloadTest, EnableGuaranteedCacheConfigField) {
    MasterServiceConfig config;
    EXPECT_FALSE(config.enable_guaranteed_cache);  // default off
    config.enable_guaranteed_cache = true;
    EXPECT_TRUE(config.enable_guaranteed_cache);
}
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" 2>&1 | tail -10
```
预期：**编译失败** —— `MasterServiceConfig` 无 `enable_guaranteed_cache`。

- [ ] **Step 3（测试必需）：给 `MasterServiceConfig` 加字段**

在 `mooncake-store/include/master_config.h`，`class MasterServiceConfig` 内 `bool enable_offload = false;`（约 line 974）之后加：

```cpp
    bool enable_guaranteed_cache = false;
```

- [ ] **Step 4（测试必需）：加 `MasterService` 成员**

在 `mooncake-store/include/master_service.h`，`const bool enable_offload_;`（约 line 1902）之后加：

```cpp
    const bool enable_guaranteed_cache_{false};
```

- [ ] **Step 5（测试必需）：构造函数初始化该成员**

在 `mooncake-store/src/master_service.cpp`，初始化列表里 `enable_offload_(config.enable_offload),`（约 line 205）之后加：

```cpp
      enable_guaranteed_cache_(config.enable_guaranteed_cache),
```

- [ ] **Step 6（测试必需）：自审** —— 确认测试必需链完整（config 字段 → 成员 → 初始化器）。

- [ ] **Step 7（生产 CLI）：在 `master_config.h` 和 `master.cpp` 处处镜像 `enable_offload`**

先找全部出现处：

```bash
grep -n "enable_offload" mooncake-store/include/master_config.h mooncake-store/src/master.cpp
```

每处加并行的 `enable_guaranteed_cache` 行，风格相同。需处理的出现处：

1. **gflag 定义**（`master.cpp`，`DEFINE_bool(enable_offload, false, ...)`）之后加：
   ```cpp
   DEFINE_bool(enable_guaranteed_cache, false,
               "Enable guaranteed offload: objects put with guaranteed_until_ms>0 "
               "are always written to SSD (independent queue, retried on failure). "
               "Defaults off for zero behavior change.");
   ```
2. **config 文件读取**（`master.cpp`，`default_config.GetBool("enable_offload", ...)`）之后加：
   ```cpp
   default_config.GetBool("enable_guaranteed_cache",
                           &master_config.enable_guaranteed_cache,
                           FLAGS_enable_guaranteed_cache);
   ```
3. **显式 CLI override 块**（`master.cpp`，`enable_offload` 的 `if ((google::GetCommandLineFlagInfo(...) && !info.is_default) || !conf_set) {...}`）之后加 `enable_guaranteed_cache` 的并行块。
4. **启动日志行**（`master.cpp`，`<< ", enable_offload=" << ...`）之后加：
   ```cpp
   << ", enable_guaranteed_cache=" << master_config.enable_guaranteed_cache
   ```
5. **`MasterConfig` 结构字段**（`master_config.h`，`bool enable_offload;`）附近加 `bool enable_guaranteed_cache = false;`。
6. **`WrappedMasterServiceConfig` 等配置变体**（`master_config.h`）：`enable_offload` 在 `WrappedMasterServiceConfig` 里是 `RequiredParam<bool>`。因 `enable_guaranteed_cache` **默认关（非必需）**，加为**普通 bool** `bool enable_guaranteed_cache = false;`（**不要**用 `RequiredParam`）。
7. **copy 块**（`master_config.h`，`enable_offload = config.enable_offload;`，约 4 处：~247/470/564/1043）每处加：
   ```cpp
   enable_guaranteed_cache = config.enable_guaranteed_cache;
   ```
   全部定位：
   ```bash
   grep -n "enable_offload = config.enable_offload" mooncake-store/include/master_config.h
   ```
8. **builder**（`master_config.h`，`set_enable_offload` + `enable_offload_` 成员 + `config.enable_offload = enable_offload_;` 写回）：镜像加 `set_enable_guaranteed_cache` + `enable_guaranteed_cache_` + 写回行。
9. **`InProcMasterConfig`**（`master_config.h`，`std::optional<bool> enable_offload;` ~1169 + InProc builder `enable_offload_` ~1186、setter `set_enable_offload` ~1218、写回 ~1272）：镜像为 `std::optional<bool> enable_guaranteed_cache;`；写回用 `config.enable_guaranteed_cache = enable_guaranteed_cache_.value_or(false);`（未设时默认关）。

- [ ] **Step 8（生产 CLI）：构建 master 确认全链编译**

```bash
cmake --build build --target mooncake_master -j"$(nproc)" 2>&1 | tail -20
```
预期：编译通过。若有 copy 块的源结构缺字段，编译器会指出——按 Step 6 加普通 bool 字段。

- [ ] **Step 9: 暂存**

```bash
git add mooncake-store/src/master.cpp mooncake-store/include/master_config.h \
        mooncake-store/include/master_service.h mooncake-store/src/master_service.cpp \
        mooncake-store/tests/guaranteed_offload_test.cpp
```

---

## Task 4: 标记 `ObjectMetadata` guaranteed + 从 config 传入

加 `guaranteed_` 成员（const，构造后不可变），由 `ReplicateConfig.guaranteed_until_ms > 0`（门控 `enable_guaranteed_cache_`）传入。Task 7 行为验证。

**文件：**
- 修改：`mooncake-store/include/master_service.h:862-887`（ctor）、`~912`（成员）
- 修改：`mooncake-store/src/master_service.cpp:2816-2820`（标记）。HA 站点 ~8330、~8527 无需改（尾随默认）。

- [ ] **Step 1: 加 `guaranteed_` 成员**

在 `mooncake-store/include/master_service.h`，`const bool hard_pinned{false};`（约 line 912）之后加：

```cpp
    const bool guaranteed_{false};        // immutable, set at creation
                                          // (Phase 1: boolean marker; Phase 3
                                          // upgrades to guaranteed_until TTL)
```

- [ ] **Step 2: 加尾随 ctor 参数 + 初始化器**

ctor 签名当前结尾（约 line 869）：

```cpp
        std::string tenant_id_ = "default",
        std::string user_key_ = {})
```

改为加尾随默认参数：

```cpp
        std::string tenant_id_ = "default",
        std::string user_key_ = {},
        bool enable_guaranteed = false)
```

初始化列表里 `hard_pinned(enable_hard_pin),`（约 line 879）之后加：

```cpp
          guaranteed_(enable_guaranteed),
```

- [ ] **Step 3: 在 `AllocateAndInsertMetadata` 传入**

`mooncake-store/src/master_service.cpp:2816-2820`，`emplace` 的 `forward_as_tuple` 末尾加 guaranteed 参数：

```cpp
    auto [it, inserted] = tenant_state.metadata.emplace(
        std::piecewise_construct, std::forward_as_tuple(key),
        std::forward_as_tuple(client_id, now, value_length, std::move(replicas),
                              config.with_soft_pin, config.with_hard_pin,
                              config.data_type, group_id, tenant_id, key,
                              enable_guaranteed_cache_ &&
                                  config.guaranteed_until_ms > 0));
```

- [ ] **Step 4: 确认 HA 反序列化站点无需改**

```bash
grep -n "make_unique<ObjectMetadata>\|metadata.emplace" mooncake-store/src/master_service.cpp
```

三个构造站点：
- **line 2816**（`AllocateAndInsertMetadata`）—— live 路径，Step 3 已改。
- **line 8330**（HA 快照 emplace，参数结尾 `..., metadata_ptr->group_id, tenant_id, user_key`）—— HA 反序列化。
- **line 8527**（`DeserializeMetadata` `make_unique<ObjectMetadata>(...)`，参数结尾 `..., is_hard_pinned, data_type, group_id`）—— HA 反序列化。

因 `enable_guaranteed` 是**尾随默认参数**（`user_key_` 之后），两 HA 站点都取默认 `false` —— 对 HA 重启正确（guaranteed_ 重置为 false，spec §7.10）。**不要改它们**。确认它们仍编译（会，因新参数都有默认值）。

- [ ] **Step 5: 构建验证编译**

```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" 2>&1 | tail -15
```
预期：编译通过。

- [ ] **Step 6: 暂存**

```bash
git add mooncake-store/include/master_service.h mooncake-store/src/master_service.cpp
```

---

## Task 5: 给 `LocalDiskSegment` 加 `guaranteed_offloading_objects` map

仅基础设施（Task 8 验证）。`LocalDiskSegment` 在 `segment.h`，**不是** `master_service.h`。

**文件：**
- 修改：`mooncake-store/include/segment.h:85-106`

- [ ] **Step 1: 加 map**

`mooncake-store/include/segment.h`，`offloading_objects` 声明（line 90-91）之后加：

```cpp
    // Guaranteed offload queue (parallel to offloading_objects). Populated by
    // PushOffloadingQueue when guaranteed=true. No size limit — guaranteed
    // objects must reach SSD. Drained by OffloadObjectHeartbeat alongside
    // offloading_objects. Same locking (offloading_mutex_).
    std::unordered_map<std::string, OffloadTaskItem> GUARDED_BY(
        offloading_mutex_) guaranteed_offloading_objects;
```

- [ ] **Step 2: 构建验证编译**

```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" 2>&1 | tail -10
```
预期：编译通过。构造函数无需改（map 默认构造）。

- [ ] **Step 3: 暂存**

```bash
git add mooncake-store/include/segment.h
```

---

## Task 6: `PushOffloadingQueue` 把 guaranteed 路由到独立队列

加尾随 `bool guaranteed = false` 参数（默认 false → 既有调用方不变），在 `offloading_objects`（normal）和 `guaranteed_offloading_objects`（guaranteed，无 limit）间选择，guaranteed 跳过 limit 检查。Task 7 验证行为。

**文件：**
- 修改：`mooncake-store/include/master_service.h`（声明）
- 修改：`mooncake-store/src/master_service.cpp:4932-4978`（定义）

- [ ] **Step 1: 更新声明**

在 `mooncake-store/include/master_service.h` 找 `PushOffloadingQueue` 声明：

```bash
grep -n "tl::expected<void, ErrorCode> PushOffloadingQueue" mooncake-store/include/master_service.h
```

签名从：

```cpp
    tl::expected<void, ErrorCode> PushOffloadingQueue(
        const ObjectIdentity& object_id, Replica& replica);
```

改为：

```cpp
    tl::expected<void, ErrorCode> PushOffloadingQueue(
        const ObjectIdentity& object_id, Replica& replica,
        bool guaranteed = false);
```

- [ ] **Step 2: 更新定义：选 map + guaranteed 跳过 limit**

`mooncake-store/src/master_service.cpp:4932-4978`，整个函数体替换为：

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

- [ ] **Step 3: 构建验证编译**

```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" 2>&1 | tail -15
```
预期：编译通过。既有调用方省略 `guaranteed`（默认 false）→ 行为不变。

- [ ] **Step 4: 暂存**

```bash
git add mooncake-store/include/master_service.h mooncake-store/src/master_service.cpp
```

---

## Task 7: `PutEnd` 总 offload guaranteed 对象（核心行为）

Phase 1 的核心。改 PutEnd offload 条件，使 guaranteed 对象无视 `offload_on_evict_` 总 offload，并传 `guaranteed` 给 `PushOffloadingQueue`。4 个行为测试。

**文件：**
- 修改：`mooncake-store/src/master_service.cpp:3064-3084`
- 修改：`mooncake-store/tests/guaranteed_offload_test.cpp`（测试）

- [ ] **Step 1: 写失败测试**

先在 fixture 里加 `PutObject`/`DrainOffloadQueue`/`WaitUntil` helper（Task 1 的 NOTE 注释处替换），再加 4 个测试。helper 代码：

```cpp
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
```

4 个测试（7A/7B/7C/7D）：

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

- [ ] **Step 2: 跑测试确认失败**

```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_offload_test \
    --gtest_filter='GuaranteedOffloadTest.Guaranteed*:GuaranteedOffloadTest.Normal*:GuaranteedOffloadTest.FlagOff*'
```
预期：7A/7C FAIL（guaranteed 未 offload —— 条件仍是 `!offload_on_evict_` 且/或未传 guaranteed）；7D 可能已过；7B 可能已过。

- [ ] **Step 3: 改 PutEnd offload 条件 + 传 guaranteed**

`mooncake-store/src/master_service.cpp:3064-3084`，把：

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

替换为：

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

- [ ] **Step 4: 跑测试确认通过**

```bash
./build/mooncake-store/tests/guaranteed_offload_test \
    --gtest_filter='GuaranteedOffloadTest.Guaranteed*:GuaranteedOffloadTest.Normal*:GuaranteedOffloadTest.FlagOff*'
```
预期：全 PASS。

- [ ] **Step 5: 跑全量新测试 + 既有 offload 测试（无回归）**

```bash
./build/mooncake-store/tests/guaranteed_offload_test && \
cmake --build build --target offload_on_evict_test -j"$(nproc)" && \
./build/mooncake-store/tests/offload_on_evict_test
```
预期：都 PASS（既有 offload 行为不变 —— guaranteed 默认关，normal 路径未动）。

- [ ] **Step 6: 暂存**

```bash
git add mooncake-store/src/master_service.cpp mooncake-store/tests/guaranteed_offload_test.cpp
```

---

## Task 8: `OffloadObjectHeartbeat` drain 两个 map + disable 时清理两个

扩展 drain 返回 `guaranteed_offloading_objects` 内容（带 `guaranteed` 标记），扩展 disable 清理清两个 map。

**文件：**
- 修改：`mooncake-store/src/master_service.cpp:4690-4750`
- 修改：`mooncake-store/tests/guaranteed_offload_test.cpp`（2 个测试）

- [ ] **Step 1: 写失败测试**

追加到 `guaranteed_offload_test.cpp`：

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

- [ ] **Step 2: 跑测试确认失败**

```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_offload_test \
    --gtest_filter='GuaranteedOffloadTest.Drain*:GuaranteedOffloadTest.Disable*'
```
预期：8A FAIL（drain 只返回 offloading_objects，guaranteed "guar" 缺失 → size 1 非 2）；8B FAIL（disable 未清 guaranteed → 下次 drain 返回它，或泄漏）。

- [ ] **Step 3: 扩展 drain（enable）分支返回两个 map**

`mooncake-store/src/master_service.cpp`，enable-drain 块（`if (enable_offloading) {...}`）替换为：

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

- [ ] **Step 4: 扩展 disable 清理：move + 清两个 map**

disable 分支（从 `offloading_objects_copy = std::move(...)` 到末尾 `return {};`）替换为（加第二个 copy map + cleanup lambda + 第二个循环）：

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

并在 `std::unordered_map<std::string, OffloadTaskItem> offloading_objects_copy;` 旁加第二个 copy map 声明：

```cpp
    std::unordered_map<std::string, OffloadTaskItem> offloading_objects_copy;
    std::unordered_map<std::string, OffloadTaskItem> guaranteed_objects_copy;
```

- [ ] **Step 5: 跑测试确认通过**

```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_offload_test \
    --gtest_filter='GuaranteedOffloadTest.Drain*:GuaranteedOffloadTest.Disable*'
```
预期：PASS。

- [ ] **Step 6: 跑全量新测试 + 既有 offload 测试**

```bash
./build/mooncake-store/tests/guaranteed_offload_test && \
./build/mooncake-store/tests/offload_on_evict_test
```
预期：都 PASS。

- [ ] **Step 7: 暂存**

```bash
git add mooncake-store/src/master_service.cpp mooncake-store/tests/guaranteed_offload_test.cpp
```

---

## Task 9: SSD 写 NACK 时重新入队 guaranteed

NACK（`metadata.data_size < 0`）时，guaranteed 对象重新入独立队列（pin 保持 —— 不 `dec_refcnt`），刷新 `start_time`，等下一批 drain 重试。非 guaranteed 保持既有 dec/erase 路径。

**文件：**
- 修改：`mooncake-store/src/master_service.cpp:4813-4829`
- 修改：`mooncake-store/tests/guaranteed_offload_test.cpp`（测试）

- [ ] **Step 1: 写失败测试**

追加到 `guaranteed_offload_test.cpp`：

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

- [ ] **Step 2: 跑测试确认失败**

```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_offload_test \
    --gtest_filter=GuaranteedOffloadTest.GuaranteedReenqueuedOnNack
```
预期：FAIL —— 当前 NACK 分支 dec_refcnt + erase，不重新入队；`second_drain` 空。

- [ ] **Step 3: 在 NACK 分支加 guaranteed 重新入队**

`mooncake-store/src/master_service.cpp`，NACK 分支（`if (metadata.data_size < 0) {...}`）替换为（guaranteed 重新入队在既有 dec/erase 之前）：

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
                            // offloading_tasks maps to `const OffloadingTask`,
                            // so the entry cannot be mutated in place — erase
                            // and re-emplace with a fresh start_time instead.
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

**注意**：`offloading_tasks` 的 mapped type 是 `const OffloadingTask`（[master_service.h:1216](../../../mooncake-store/include/master_service.h)），不能原地赋值 —— 必须 `erase` + 重新 `emplace`（带新 `start_time`）。这是 Phase 1 实现中踩过的坑，已修。

- [ ] **Step 4: 跑测试确认通过**

```bash
cmake --build build --target guaranteed_offload_test -j"$(nproc)" && \
./build/mooncake-store/tests/guaranteed_offload_test \
    --gtest_filter=GuaranteedOffloadTest.GuaranteedReenqueuedOnNack
```
预期：PASS。

- [ ] **Step 5: 跑全量新测试 + 既有 offload/promotion 测试（无回归）**

```bash
cmake --build build --target guaranteed_offload_test offload_on_evict_test promotion_on_hit_test -j"$(nproc)"
./build/mooncake-store/tests/guaranteed_offload_test
./build/mooncake-store/tests/offload_on_evict_test
./build/mooncake-store/tests/promotion_on_hit_test
```
预期：全 PASS。

- [ ] **Step 6: 暂存**

```bash
git add mooncake-store/src/master_service.cpp mooncake-store/tests/guaranteed_offload_test.cpp
```

---

## 补充测试：用例 5 & 9（覆盖缺口）

spec §10 用例 5（SSD 写成功后 guaranteed 变可驱逐）和用例 9（`enable_offload=false` 降级）。

- [ ] **加 `#include <algorithm>`**（用 `std::any_of`），在 includes 区。

- [ ] **用例 5：`GuaranteedBecomesEvictableAfterSsdSuccess`**

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

- [ ] **用例 9：`GuaranteedDegradesWhenOffloadDisabled`**

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

- [ ] **暂存**

```bash
git add mooncake-store/tests/guaranteed_offload_test.cpp
```

---

## 最终验证

- [ ] **Step 1: 构建所有 store 测试**

```bash
cmake --build build --target mooncake_store_tests -j"$(nproc)" 2>/dev/null || \
cmake --build build -j"$(nproc)"
```

- [ ] **Step 2: 跑全量 guaranteed 测试**

```bash
./build/mooncake-store/tests/guaranteed_offload_test
```
预期：全 PASS（12 个测试）。

- [ ] **Step 3: 跑更广的 offload/promotion 回归套件**

```bash
cd build && ctest -R "offload_on_evict_test|promotion_on_hit_test|guaranteed_offload_test" -V
```
预期：全 PASS —— 无回归。

- [ ] **Step 4: 确认 flag 关时零行为变化**

默认 `enable_guaranteed_cache=false` → `guaranteed_` 永不被设（Task 4 门控）→ `metadata.guaranteed_` 恒 false → PutEnd 条件退化为 `!offload_on_evict_`（原始）→ `PushOffloadingQueue` 传 `guaranteed=false`（原始）。`offload_on_evict_test` 通过即证明。

---

## 备注

- **`guaranteed_` 是 const**（构造后不可变），经尾随默认 ctor 参数传入，故两 HA 反序列化站点无需改（默认 false = HA 重启重置，spec §7.10）。
- **guaranteed 队列无 limit** —— 隐式有界于内存中待 offload 的 guaranteed 对象（各持一个 pinned memory replica）。
- **重试无上限** —— guaranteed 意为"必须写入"；SSD 持续故障时 pin 内存直至运维介入（spec §7）。
- **client 侧 `BuildBucket` 分流**（spec §6.4、测试用例 10）推迟到 Phase 2 —— 无 Phase 2 bucket pin 无可观察效果，且在不同模块。master 已把 `OffloadTaskItem.guaranteed` 传给 client。
- **本机 build 环境可能坏**（缺 `libmsgpack-dev` + yalantinglibs include 未传播；`dependencies.sh` 未跑）。若 `cmake --build` 报 `ylt/util/expected.hpp`，是环境问题 —— 用户在 build 环境验证。写正确代码 + 读代码自审。

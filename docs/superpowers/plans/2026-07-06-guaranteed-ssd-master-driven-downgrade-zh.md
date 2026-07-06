# Guaranteed SSD Master-Driven Downgrade 实施计划（中文）

> **状态：进行中（2026-07-06）。**
> - ✅ **Task 1 已实现并验证**（`ObjectMetadata.guaranteed_` bool→`guaranteed_until_` time_point + 3 用点改 `>now` + config 全层 + `GetGuaranteedUntilForTesting` + 4 单测通过）。无 #2676 依赖。
> - ⏳ **Task 2-5 阻塞于 PR #2676**（`PollRemoveAll` 通道未合入 main；`PollDowngradeKeys`/`DispatchGuaranteedExpiry`/worker 翻转/`BatchExpireGuaranteed` 依赖该模式）。#2676 合入后作 follow-up。
> - ⏳ **Task 6 待做**（读时续期 `GetReplicaList`，纯 master 侧，无 #2676 依赖，可现在做）。
> - 📌 **Minor 待补**：`master.cpp` gflag 接线（`DEFINE_int64(guaranteed_until_ms)` + config 文件 `GetInt64`），生产部署前补（Task 1 只改 struct 层，测试直接构造 config 不受影响）。
>
> **取代了**过时的 client-side TTL 方案（[2026-07-06-guaranteed-ssd-offload-priority-phase3-design-zh.md](2026-07-06-guaranteed-ssd-offload-priority-phase3-design-zh.md)，已废弃）。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 guaranteed 对象在 TTL 到期（或显式失效）后，由 master 通过 #2676 通道下发"降级"指令，worker 把 SSD 上对应 bucket 的 `guaranteed` 标记翻为 false，随后由**现成的 LRU 驱逐路径**自动回收——master 管驱逐时机，worker 复用既有删除路径。

**Architecture:** master 拥有 guaranteed 生命周期（`guaranteed_until_` 时间戳，object 级，master-only，不向 worker 同步，类似 `lease_timeout`）。到期时 master 反查持有该 LOCAL_DISK replica 的 `client_id`（`Replica::get_local_disk_client_id`），把 tenant-scoped storage key 推入该 client 的 `pending_downgrade_keys` 集合。worker 在 heartbeat 里轮询 `PollDowngradeKeys` 取回 key 列表，按 bucket_id 聚合，**仅当降级集覆盖 bucket 全部 key 时**才翻转 `BucketMetadata.guaranteed=false`（部分覆盖则延迟，符合"延迟桶级"选择）；翻转后现成 `SelectEvictionCandidate→PrepareEviction→FinalizeEviction→BatchEvictDiskReplica` 自动回收。内存驱逐对 guaranteed 不做任何额外处理（已验证现状）。

**Tech Stack:** C++17, ylt/ylt_ref (YLT_REFL wire 兼容), Mooncake Store master/worker RPC (ylt coro_rpc), gtest。

---

## 前置条件

- **PR #2676 必须先合入 main**（提供 `PollRemoveAll` bool 通道 + per-client `pending_remove_all` 标志 + `FileStorage::Heartbeat` STEP 2 轮询点 + 各后端 `RemoveAll()`）。本计划泛化其模式（bool 全清 → key 列表选择性降级），**不改 #2676 本身**，作为 follow-up。
- 本地 main 当前**已有** `BucketStorageBackend::DeleteBucket`（[storage_backend.cpp:2457](../../../mooncake-store/src/storage_backend.cpp#L2457)）、`object_bucket_map_`（key→`StorageObjectMetadata{bucket_id}`）、`Replica::get_local_disk_client_id`（[replica.h:411](../../../mooncake-store/include/replica.h#L411)）、`MakeTenantScopedStorageKey`（[file_storage.cpp:374](../../../mooncake-store/src/file_storage.cpp#L374)）——全部复用，无需新建。
- 本地 main **尚未**有 `PollRemoveAll`/`pending_remove_all`（属 #2676）。Task 2 假定其已合入并对照其模式实现。

## 与 Phase 3 文档方案对比

| 维度 | Phase 3 文档原方案（TTL-on-worker） | 本计划（master-driven downgrade） |
|---|---|---|
| TTL 存放 | master + wire 到 worker + worker runtime atomic | **仅 master**（object 级，类似 lease） |
| 跨节点 TTL 同步 | heartbeat 挎带 struct（改返回类型） | **消除**——只在到期时下发一次性降级列表 |
| worker `guaranteed_until_ns_` + grace | 需要 | **不需要** |
| `SelectEvictionCandidate` TTL 判断 | 改热路径 | **不改**（只读 `bucket->guaranteed`，到期靠翻转 bool） |
| 删除执行 | worker TTL 到期自删 | **现成 LRU 路径**（翻转后自动驱逐） |
| `Serializer<Replica>` bump 风险 | 中 | **无**（`guaranteed_until_` 在 `ObjectMetadata`，非 Replica，非 HA 序列化） |

**净效果**：消除 Phase 3 最难部分（跨节点 TTL 同步、热路径 TTL 判断、heartbeat 返回结构改造），代价是新增一条泛化通道 + worker 一处 bool 翻转（冷路径）。

## 维护性硬约束（来自项目既定原则）

- 加法优于侵入：`DowngradeKeys` 为 `StorageBackendInterface` 新增虚函数（默认 no-op）。
- 新字段尾随 + 默认值：wire 结构新增字段放尾部，默认 0/空。
- **避开 `Serializer<Replica>` 格式 bump**：`guaranteed_until_` 与 `downgrade_dispatched_` 都在 `ObjectMetadata`（master 运行时态，`SerializeMetadata` 不含 `guaranteed_`，已验证 [master_service.cpp:8455-8478](../../../mooncake-store/src/master_service.cpp#L8455)），**不进 HA**。
- feature flag `enable_guaranteed_cache`（默认 false）门控全部新行为。

## File Structure

> **测试代码约定**：本仓库**无** `master_service_test_base.h`，测试是独立 fixture（见 `mooncake-store/tests/guaranteed_offload_test.cpp:18` 的 `GuaranteedOffloadTest : public ::testing::Test`，在 `SetUp` 内联构造 `MasterService` + `service.PutEnd(client_id, key, "default", ReplicaType::MEMORY)`）。下文测试为简洁起见用 helper 名（`PutEndGuaranteed`/`OffloadToSSD`/`InspectObjectMetadata`/`PumpWorkerHeartbeat`/`TriggerSSDPressureEviction`/`DispatchGuaranteedExpiryForTest` 等）代指——**实施时按 `guaranteed_offload_test.cpp` 既有内联写法实现这些 fixture helper**，并新增 `MasterService` 的 test-only inspection accessor（如 `InspectObjectMetadata` 返回含 `guaranteed_until`/`downgrade_dispatched` 的 view，仅编入 test target）。场景类 helper（`OffloadToSSD` 等）参照既有 `guaranteed_offload_test.cpp` 的 offload 触发写法。生产代码（Task 1-6 的实现）全部为可直接编译的具体代码，不含此抽象。

| 文件 | 责任 | 改动类型 |
|---|---|---|
| `mooncake-store/include/master_service.h` | `ObjectMetadata` 升级 `guaranteed_`→`guaranteed_until_` + `downgrade_dispatched_`；声明 `PollDowngradeKeys`、`BatchExpireGuaranteed`、`DispatchGuaranteedExpiry` | Modify |
| `mooncake-store/src/master_service.cpp` | 实现：构造/续期/到期扫描/反查 client_id/填充 pending；PutEnd/NACK 用点改判断 | Modify |
| `mooncake-store/include/master_config.h` | 新增 `guaranteed_until_ms`、`guaranteed_renewal_ttl_ms`（尾随，默认 0） | Modify |
| `mooncake-store/include/segment.h` | `LocalDiskSegment` 加 `pending_downgrade_keys` | Modify |
| `mooncake-store/include/rpc_service.h`、`src/rpc_service.cpp` | `WrappedMasterService::PollDowngradeKeys` + `BatchExpireGuaranteed` 透传 | Modify |
| `mooncake-store/include/master_client.h`、`src/master_client.cpp` | client 侧 RPC 封装 | Modify |
| `mooncake-store/include/client_service.h`、`src/client_service.cpp` | `Client::PollDowngradeKeys` | Modify |
| `mooncake-store/include/storage_backend.h`、`src/storage_backend.cpp` | `StorageBackendInterface::DowngradeKeys`（默认 no-op）+ `BucketStorageBackend::DowngradeKeys`（聚合+延迟翻转） | Modify |
| `mooncake-store/include/file_storage.h`、`src/file_storage.cpp` | `FileStorage::Heartbeat` STEP 2 轮询 `PollDowngradeKeys` 并转交 backend | Modify |
| `mooncake-store/include/master_admin_service.h`、`src/master_admin_service.cpp` | `HandleBatchExpireGuaranteed` HTTP 端点 | Modify |
| `mooncake-store/tests/guaranteed_downgrade_test.cpp` | 全部新增单测 + E2E | Create |

---

## Task 1: master 侧 `guaranteed_until_` 时间戳（master-only，不进 HA/wire）

> **代码事实验证（2026-07-06，行号基于 Phase 1/2 后状态）**：`guaranteed_`（const bool）在 master_service.h:915；ctor 尾参 `enable_guaranteed` 在 :870、初始化 `guaranteed_(enable_guaranteed)` 在 :881；`AllocateAndInsertMetadata` 在 master_service.cpp:2822（`enable_guaranteed_cache_ && config.guaranteed_until_ms > 0` bool 归约）；3 用点：PutEnd 条件 :3067（`!offload_on_evict_ || metadata.guaranteed_`）、lambda capture :3075（`guaranteed = metadata.guaranteed_`）、NACK 重试 :4845（`obj_metadata.guaranteed_ && enable_guaranteed_cache_`）；HA `SerializeMetadata`（:8455-8478）**不含** `guaranteed_`（运行时态，重启重置）；config 3 层结构（MasterServiceConfig :54、MasterConfig :209/:401、InProc :987/:1185）各有 `enable_guaranteed_cache` + builder setter（:758/:1240）；**test-only accessor 先例**：`GetNoFSegmentMountedForTesting`/`GetTenantQuotaSnapshotForTesting`（master_service.h:96-100，`ForTesting` 后缀）。

**Files:**
- Modify: `mooncake-store/include/master_service.h:870-917`（ctor 尾参 + `guaranteed_` 字段 → `guaranteed_until_` + `downgrade_dispatched_`）+ 新增 test-only accessor 声明
- Modify: `mooncake-store/src/master_service.cpp:2822`（`AllocateAndInsertMetadata` 传 TTL）、`:3067`、`:3075`、`:4845`（PutEnd/NACK 用点改 `> now`）+ accessor 实现
- Modify: `mooncake-store/include/master_config.h:54`（config 字段，3 层同步）
- Create: `mooncake-store/tests/guaranteed_downgrade_test.cpp`（fixture + 单测）
- Modify: `mooncake-store/tests/CMakeLists.txt`（注册 target）

- [ ] **Step 1: 注册测试 target + 写 fixture scaffold**

`mooncake-store/tests/CMakeLists.txt`，`add_store_test(guaranteed_eviction_test ...)` 后加：
```cmake
add_store_test(guaranteed_downgrade_test guaranteed_downgrade_test.cpp)
```

新建 `mooncake-store/tests/guaranteed_downgrade_test.cpp`，fixture 镜像 `guaranteed_offload_test.cpp`（内联构造 `MasterService`，`PrepareSegment`/`PutObject`/`DrainOffloadQueue` helper 复制）：
```cpp
#include "master_service.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "types.h"

namespace mooncake::test {

class GuaranteedDowngradeTest : public ::testing::Test {
   protected:
    void SetUp() override {
        google::InitGoogleLogging("GuaranteedDowngradeTest");
        FLAGS_logtostderr = true;
    }
    void TearDown() override { google::ShutdownGoogleLogging(); }

    static constexpr size_t kDefaultSegmentBase = 0x300000000;

    // 构造 MasterService（enable_guaranteed_cache=true）。
    std::unique_ptr<MasterService> MakeMaster(
        int64_t guaranteed_until_ms = 60000) {
        MasterServiceConfig config;
        config.enable_offload = true;
        config.enable_guaranteed_cache = true;
        config.guaranteed_until_ms = guaranteed_until_ms;
        config.default_kv_lease_ttl = 2000;
        return std::make_unique<MasterService>(config);
    }
    // ... PrepareSegment / PutObject / DrainOffloadQueue helper
    //     （复制自 guaranteed_offload_test.cpp，保持一致）
};

}  // namespace mooncake::test
```

- [ ] **Step 2: 写失败单测 —— guaranteed_until_ 在 PutEnd 后被设置（用 `ForTesting` accessor）**

```cpp
// Task 1: PutEnd with guaranteed_until_ms sets guaranteed_until_ ≈ now+TTL.
TEST_F(GuaranteedDowngradeTest, GuaranteedUntilSetOnPutEnd) {
    auto master = MakeMaster(/*guaranteed_until_ms=*/60000);
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*master, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(master->MountLocalDiskSegment(ctx.client_id, true).has_value());

    auto t0 = std::chrono::system_clock::now();
    PutObject(*master, ctx.client_id, "k1", /*guaranteed_until_ms=*/60000);
    auto t1 = std::chrono::system_clock::now();

    // guaranteed_until_ ≈ t0 + 60s（容差覆盖 t0..t1 的构造开销）
    auto gu = master->GetGuaranteedUntilForTesting("k1", "default");
    ASSERT_TRUE(gu.has_value());
    EXPECT_GT(*gu, t0 + std::chrono::seconds(59));
    EXPECT_LT(*gu, t1 + std::chrono::seconds(61));
}

// Task 1: non-guaranteed Put (guaranteed_until_ms=0) leaves guaranteed_until_ = epoch.
TEST_F(GuaranteedDowngradeTest, NonGuaranteedPutLeavesEpoch) {
    auto master = MakeMaster();
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*master, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(master->MountLocalDiskSegment(ctx.client_id, true).has_value());

    PutObject(*master, ctx.client_id, "normal");  // guaranteed_until_ms=0
    auto gu = master->GetGuaranteedUntilForTesting("normal", "default");
    ASSERT_TRUE(gu.has_value());
    EXPECT_EQ(*gu, std::chrono::system_clock::time_point{});
}

// Task 1: enable_guaranteed_cache=false → guaranteed_until_ stays epoch even with TTL>0.
TEST_F(GuaranteedDowngradeTest, FlagOffLeavesEpochEvenWithTtl) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = false;  // flag off
    config.guaranteed_until_ms = 60000;
    config.default_kv_lease_ttl = 2000;
    MasterService master(config);
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(master, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(master.MountLocalDiskSegment(ctx.client_id, true).has_value());

    PutObject(master, ctx.client_id, "k1", /*guaranteed_until_ms=*/60000);
    auto gu = master.GetGuaranteedUntilForTesting("k1", "default");
    ASSERT_TRUE(gu.has_value());
    EXPECT_EQ(*gu, std::chrono::system_clock::time_point{});
}

// Task 1: key 不存在 → nullopt。
TEST_F(GuaranteedDowngradeTest, MissingKeyReturnsNullopt) {
    auto master = MakeMaster();
    auto gu = master->GetGuaranteedUntilForTesting("nope", "default");
    EXPECT_FALSE(gu.has_value());
}
```

- [ ] **Step 3: 跑测试确认失败**

```bash
cd mooncake-store && cmake --build build --target guaranteed_downgrade_test && \
./build/tests/guaranteed_downgrade_test --gtest_filter='GuaranteedDowngradeTest.Guaranteed*:GuaranteedDowngradeTest.NonGuaranteed*:GuaranteedDowngradeTest.FlagOff*:GuaranteedDowngradeTest.MissingKey*'
```
Expected: 编译失败 —— `GetGuaranteedUntilForTesting` 不存在 + `guaranteed_until_` 字段不存在。

- [ ] **Step 4: 升级字段类型（master_service.h:915）**

`ObjectMetadata` 成员：
```cpp
// 原: const bool guaranteed_{false};        // immutable, set at creation
// 改:
std::chrono::system_clock::time_point guaranteed_until_{};  // epoch = 非guaranteed
bool downgrade_dispatched_{false};  // runtime only, 非 HA: 到期已下发降级（Task 3 用，Task 1 先声明）
```

构造函数（master_service.h:870）尾参 `bool enable_guaranteed = false` → `int64_t guaranteed_until_ms = 0`，初始化列表（:881）：
```cpp
// 原: guaranteed_(enable_guaranteed),
// 改:
guaranteed_until_(guaranteed_until_ms > 0
    ? std::chrono::system_clock::now()
          + std::chrono::milliseconds(guaranteed_until_ms)
    : std::chrono::system_clock::time_point{}) ,
```
**注意**：`guaranteed_until_` 非 const（续期/降级要改），而 `hard_pinned` 仍 const。初始化列表顺序需匹配声明顺序（`hard_pinned` 在前，`guaranteed_until_` 在后）——确认 `hard_pinned(enable_hard_pin),` 在 `guaranteed_until_(...)` 之前。

- [ ] **Step 5: `AllocateAndInsertMetadata`（master_service.cpp:2822）传 TTL**

```cpp
// 原: enable_guaranteed_cache_ && config.guaranteed_until_ms > 0  (bool 归约)
// 改: 门控 flag 决定是否传 TTL，否则传 0（epoch）
const int64_t effective_guaranteed_ms =
    enable_guaranteed_cache_ ? config.guaranteed_until_ms : 0;
// forward_as_tuple 末尾用 effective_guaranteed_ms（替代原 bool）
```

- [ ] **Step 6: 3 处用点 `guaranteed_` → `guaranteed_until_ > now`（master_service.cpp:3067/3075/4845）**

PutEnd 条件（:3067）：
```cpp
// 原: if (enable_offload_ && (!offload_on_evict_ || metadata.guaranteed_)) {
// 改:
const auto now_pe = std::chrono::system_clock::now();
if (enable_offload_ &&
    (!offload_on_evict_ || metadata.guaranteed_until_ > now_pe)) {
```
lambda capture（:3075）——保持 bool 透传给 `PushOffloadingQueue`（`OffloadTaskItem.guaranteed` Phase 1 字段不变）：
```cpp
// 原: guaranteed = metadata.guaranteed_](Replica& replica) {
// 改: guaranteed = (metadata.guaranteed_until_ > now_pe)](Replica& replica) {
```
NACK 重试（:4845）：
```cpp
// 原: if (source != nullptr && obj_metadata.guaranteed_ && enable_guaranteed_cache_) {
// 改:
const auto now_nack = std::chrono::system_clock::now();
if (source != nullptr &&
    obj_metadata.guaranteed_until_ > now_nack && enable_guaranteed_cache_) {
```
**等价性**：TTL 活性期间（`guaranteed_until_ > now` 为 true）行为与 Phase 1 的 `guaranteed_` 等价；过期后变 false —— 这是 Task 1 引入的新语义（为 Task 3 派发铺路）。

- [ ] **Step 7: 新增 test-only accessor（master_service.h，紧邻 :96-100 `ForTesting` 群）**

声明：
```cpp
    // Test-only: returns guaranteed_until_ for an object, or nullopt if absent.
    std::optional<std::chrono::system_clock::time_point>
    GetGuaranteedUntilForTesting(const std::string& key,
                                 const std::string& tenant_id) const;
```
实现（master_service.cpp，紧邻其他 `ForTesting` 实现）：
```cpp
std::optional<std::chrono::system_clock::time_point>
MasterService::GetGuaranteedUntilForTesting(
    const std::string& key, const std::string& tenant_id) const {
    std::shared_lock<std::shared_mutex> shared_lock(snapshot_mutex_);
    const auto object_id = MakeObjectIdentityForRequest(key, tenant_id);
    MetadataAccessorRO accessor(this, object_id);
    if (!accessor.Exists()) return std::nullopt;
    return accessor.Get().guaranteed_until_;
}
```

- [ ] **Step 8: 新增 config 字段（master_config.h，3 层同步，尾随默认 0）**

`MasterServiceConfig`（:54 附近，`enable_guaranteed_cache` 后）：
```cpp
    bool enable_guaranteed_cache = false;
    int64_t guaranteed_until_ms = 0;        // 0 = 不启用 guaranteed TTL
    int64_t guaranteed_renewal_ttl_ms = 0;  // 0 = 不续期（严格 TTL）；>0 = 读时续期（Task 6）
```
3 层同步（grep `enable_guaranteed_cache` 找全 :209/:401/:987/:1185 等，每层加 `guaranteed_until_ms` 字段 + copy 块 + builder setter `set_guaranteed_until_ms`）。**MasterService 构造**（master_service.cpp:206 附近）读 `config.guaranteed_until_ms` 存成员 `guaranteed_until_ms_`（供 Task 3/6 用；Task 1 本身不直接用，但先存上避免 Task 3 再改 ctor）。

- [ ] **Step 9: 跑单测确认通过**

```bash
cd mooncake-store && cmake --build build --target guaranteed_downgrade_test && \
./build/tests/guaranteed_downgrade_test
```
Expected: 4 个新单测 PASS。

- [ ] **Step 10: 回归已有 guaranteed 测试（零回归）**

```bash
./build/tests/guaranteed_offload_test && \
./build/tests/guaranteed_eviction_test && \
./build/tests/offload_on_evict_test
```
Expected: PASS —— bool→时间戳在 TTL 活性期间行为等价，已有用例不回归。

- [ ] **Step 11: 暂存（不 commit，按用户偏好）**

```bash
git add mooncake-store/include/master_service.h mooncake-store/src/master_service.cpp \
        mooncake-store/include/master_config.h mooncake-store/tests/guaranteed_downgrade_test.cpp \
        mooncake-store/tests/CMakeLists.txt
```

---

## Task 2: per-client `pending_downgrade_keys` + `PollDowngradeKeys` RPC（泛化 #2676 模式）

**Files:**
- Modify: `mooncake-store/include/segment.h:84`（`LocalDiskSegment`）
- Modify: `mooncake-store/include/master_service.h`（声明 `PollDowngradeKeys`）
- Modify: `mooncake-store/src/master_service.cpp`（实现，对照 #2676 `PollRemoveAll`）
- Modify: `mooncake-store/include/rpc_service.h`、`src/rpc_service.cpp`（`WrappedMasterService` + `RegisterRpcService`）
- Modify: `mooncake-store/include/master_client.h`、`src/master_client.cpp`（`MasterClient::PollDowngradeKeys`）
- Modify: `mooncake-store/include/client_service.h`、`src/client_service.cpp`（`Client::PollDowngradeKeys`）
- Test: `mooncake-store/tests/guaranteed_downgrade_test.cpp`

- [ ] **Step 1: 写失败测试 —— Poll 原子 drain**

```cpp
TEST_F(GuaranteedDowngradeTest, PollDowngradeKeysDrainsAndClears) {
    auto* master = GetMasterService();
    master->EnqueueDowngradeForTest(GetHolderClientId(), {"tenant:k1", "tenant:k2"});
    auto r1 = master->PollDowngradeKeys(GetHolderClientId());
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(r1->size(), 2u);
    auto r2 = master->PollDowngradeKeys(GetHolderClientId());
    ASSERT_TRUE(r2.has_value());
    EXPECT_TRUE(r2->empty());  // 已 drain
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd mooncake-store && cmake --build build --target mooncake_store_tests && ./build/tests/guaranteed_downgrade_test --gtest_filter=GuaranteedDowngradeTest.PollDowngradeKeysDrainsAndClears`
Expected: FAIL（`PollDowngradeKeys` 不存在）

- [ ] **Step 3: `LocalDiskSegment` 加字段（segment.h，紧邻 #2676 的 `pending_remove_all`）**

```cpp
// Guaranteed 到期/显式失效待下发降级的 key 集合（tenant-scoped storage key）。
// worker heartbeat 轮询取回并翻转对应 bucket 的 guaranteed 标记。
// 与 offloading_objects 同锁（offloading_mutex_）。
std::unordered_set<std::string> GUARDED_BY(offloading_mutex_) pending_downgrade_keys;
```

- [ ] **Step 4: master 实现 `PollDowngradeKeys`（对照 #2676 `PollRemoveAll` 的 check-and-clear）**

`master_service.h`（声明，紧邻 `PollRemoveAll`）：
```cpp
auto PollDowngradeKeys(const UUID& client_id)
    -> tl::expected<std::vector<std::string>, ErrorCode>;
```
`master_service.cpp`（实现，紧邻 `PollRemoveAll`）：
```cpp
auto MasterService::PollDowngradeKeys(const UUID& client_id)
    -> tl::expected<std::vector<std::string>, ErrorCode> {
    std::shared_lock<std::shared_mutex> shared_lock(snapshot_mutex_);
    ScopedLocalDiskSegmentAccess access = segment_manager_.getLocalDiskSegmentAccess();
    auto& segs = access.getClientLocalDiskSegment();
    auto it = segs.find(client_id);
    if (it == segs.end()) {
        return tl::make_unexpected(ErrorCode::SEGMENT_NOT_FOUND);
    }
    std::vector<std::string> result;
    {
        MutexLocker locker(&it->second->offloading_mutex_);
        result.reserve(it->second->pending_downgrade_keys.size());
        result.assign(it->second->pending_downgrade_keys.begin(),
                      it->second->pending_downgrade_keys.end());
        it->second->pending_downgrade_keys.clear();  // drain（幂等：worker 翻转 false→false 无害）
    }
    return result;
}
```

- [ ] **Step 5: 贯通 wire（rpc_service / master_client / client_service）**

完全对照 #2676 对 `PollRemoveAll` 的改法：`WrappedMasterService::PollDowngradeKeys`、`RegisterRpcService` 注册 `&WrappedMasterService::PollDowngradeKeys`、`RpcNameTraits` 特化、`MasterClient::PollDowngradeKeys`（`invoke_rpc<..., std::vector<std::string>>(client_id_)`）、`Client::PollDowngradeKeys`（转调 `master_client_`）。返回类型 `std::vector<std::string>`，ylt 已支持 vector 序列化，无需新增结构。

- [ ] **Step 6: 跑测试确认通过**

Run: `cd mooncake-store && cmake --build build --target mooncake_store_tests && ./build/tests/guaranteed_downgrade_test --gtest_filter=GuaranteedDowngradeTest.PollDowngradeKeysDrainsAndClears`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add mooncake-store/include/segment.h mooncake-store/include/master_service.h \
        mooncake-store/src/master_service.cpp mooncake-store/include/rpc_service.h \
        mooncake-store/src/rpc_service.cpp mooncake-store/include/master_client.h \
        mooncake-store/src/master_client.cpp mooncake-store/include/client_service.h \
        mooncake-store/src/client_service.cpp mooncake-store/tests/guaranteed_downgrade_test.cpp
git commit -m "feat(store): add PollDowngradeKeys RPC (generalize #2676 PollRemoveAll to selective key list)"
```

---

## Task 3: 周期性到期扫描 —— 在 `TaskCleanupThreadFunc` 里派发降级

**Files:**
- Modify: `mooncake-store/src/master_service.cpp:1866`（`TaskCleanupThreadFunc` loop）
- Modify: `mooncake-store/include/master_service.h`（声明 `DispatchGuaranteedExpiry`）
- Test: `mooncake-store/tests/guaranteed_downgrade_test.cpp`

- [ ] **Step 1: 写失败测试 —— 到期后 pending 被填充到正确的 client**

```cpp
TEST_F(GuaranteedDowngradeTest, ExpiredGuaranteedDispatchedToHolderClient) {
    auto* master = GetMasterService();
    UUID holder = GetHolderClientId();  // 持有 LOCAL_DISK replica 的 worker
    // 用 1ms TTL 写入并 offload 到 SSD，等其过期
    PutEndGuaranteed("k1", /*guaranteed_until_ms=*/1);
    OffloadToSSD("k1", holder);  // 触发 NotifyOffloadSuccess，建立 LOCAL_DISK replica
    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // 过期
    master->DispatchGuaranteedExpiryForTest();  // 单次触发扫描
    auto keys = master->PollDowngradeKeys(holder);
    ASSERT_TRUE(keys.has_value());
    ASSERT_EQ(keys->size(), 1u);
    EXPECT_EQ((*keys)[0], MakeTenantScopedStorageKey("tenant", "k1"));
    // 已派发：downgrade_dispatched_ = true（不重复派发）
    auto meta = master->InspectObjectMetadata("k1", "tenant");
    EXPECT_TRUE(meta->downgrade_dispatched);
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd mooncake-store && cmake --build build --target mooncake_store_tests && ./build/tests/guaranteed_downgrade_test --gtest_filter=GuaranteedDowngradeTest.ExpiredGuaranteedDispatchedToHolderClient`
Expected: FAIL（`DispatchGuaranteedExpiry` 不存在）

- [ ] **Step 3: 实现 `DispatchGuaranteedExpiry`**

`master_service.h`（声明）：
```cpp
// 周期扫描：guaranteed_until_ 到期且未派发的对象 → 反查 holder client_id
// → 推入其 pending_downgrade_keys，置 downgrade_dispatched_=true。幂等。
void DispatchGuaranteedExpiry();
```
`master_service.cpp`（实现，遍历 shard + tenant，模式参照 `RemoveAll` 的遍历 [master_service.cpp:4394](../../../mooncake-store/src/master_service.cpp#L4394)；`MakeTenantScopedStorageKey` 已在 [types.h:242](../../../mooncake-store/include/types.h#L242)，master 已在用 [master_service.cpp:5027](../../../mooncake-store/src/master_service.cpp#L5027)）：
```cpp
void MasterService::DispatchGuaranteedExpiry() {
    if (!enable_guaranteed_cache_) return;
    std::shared_lock<std::shared_mutex> shared_lock(snapshot_mutex_);
    const auto now = std::chrono::system_clock::now();
    for (size_t i = 0; i < kNumShards; ++i) {
        MetadataShardAccessorRW shard(this, i);
        for (auto& [tenant_id, tenant_state] : shard->tenants) {  // 外层 key = tenant_id
            for (auto& [user_key, metadata] : tenant_state.metadata) {
                if (metadata.downgrade_dispatched_) continue;
                if (metadata.guaranteed_until_ == std::chrono::system_clock::time_point{}
                    || metadata.guaranteed_until_ > now) continue;  // 非guaranteed 或未到期
                // 反查 LOCAL_DISK replica 的 holder client_id
                UUID holder{};
                bool found = false;
                metadata.VisitReplicas(&Replica::fn_is_completed,
                    [&](const Replica& r) {
                        if (!found && r.is_local_disk_replica()) {
                            auto cid = r.get_local_disk_client_id();
                            if (cid.has_value()) { holder = *cid; found = true; }
                        }
                    });
                if (!found) continue;  // 无 LOCAL_DISK replica，交由 lease reaper 处理
                const auto storage_key = MakeTenantScopedStorageKey(tenant_id, user_key);
                ScopedLocalDiskSegmentAccess access =
                    segment_manager_.getLocalDiskSegmentAccess();
                auto& segs = access.getClientLocalDiskSegment();
                auto it = segs.find(holder);
                if (it != segs.end()) {
                    MutexLocker locker(&it->second->offloading_mutex_);
                    it->second->pending_downgrade_keys.insert(storage_key);
                }
                metadata.downgrade_dispatched_ = true;  // 派发一次
            }
        }
    }
}
```

- [ ] **Step 4: 挂到周期线程**

`master_service.cpp:1881`（`TaskCleanupThreadFunc` 内 `write_access.prune_finished_tasks();` 之后）：
```cpp
        write_access.prune_finished_tasks();
        DispatchGuaranteedExpiry();  // 周期派发 guaranteed 降级
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd mooncake-store && cmake --build build --target mooncake_store_tests && ./build/tests/guaranteed_downgrade_test --gtest_filter=GuaranteedDowngradeTest.ExpiredGuaranteedDispatchedToHolderClient`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mooncake-store/src/master_service.cpp mooncake-store/include/master_service.h \
        mooncake-store/tests/guaranteed_downgrade_test.cpp
git commit -m "feat(store): dispatch guaranteed downgrade on TTL expiry via periodic sweep"
```

---

## Task 4: worker 端延迟桶级降级翻转（复用现成 LRU 驱逐）

**Files:**
- Modify: `mooncake-store/include/storage_backend.h`（`StorageBackendInterface::DowngradeKeys` 默认 no-op + `BucketStorageBackend` override）
- Modify: `mooncake-store/src/storage_backend.cpp`（`BucketStorageBackend::DowngradeKeys` 实现）
- Modify: `mooncake-store/src/file_storage.cpp:701`（`Heartbeat` STEP 2 轮询并转交）
- Test: `mooncake-store/tests/guaranteed_downgrade_test.cpp`

- [ ] **Step 1: 写失败测试 —— 全 key 降级翻转，部分覆盖则延迟**

```cpp
TEST_F(GuaranteedDowngradeTest, WorkerDefersPartialBucketDowngrade) {
    // 准备一个含 k1,k2 的 guaranteed bucket（BuildBucket multi-key）
    auto* backend = GetBucketStorageBackend();
    int64_t bid = OffloadGuaranteedBucket({"k1", "k2"});  // 同 bucket 两 key
    EXPECT_TRUE(backend->IsBucketGuaranteed(bid));
    // 仅降级 k1：bucket 仍 guaranteed（延迟）
    backend->DowngradeKeys({MakeStorageKey("k1")});
    EXPECT_TRUE(backend->IsBucketGuaranteed(bid));
    // 降级 k1+k2：bucket 翻转
    backend->DowngradeKeys({MakeStorageKey("k1"), MakeStorageKey("k2")});
    EXPECT_FALSE(backend->IsBucketGuaranteed(bid));
    // 翻转后 LRU 可驱逐（触发 SSD 压力即回收）
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd mooncake-store && cmake --build build --target mooncake_store_tests && ./build/tests/guaranteed_downgrade_test --gtest_filter=GuaranteedDowngradeTest.WorkerDefersPartialBucketDowngrade`
Expected: FAIL（`DowngradeKeys` 不存在）

- [ ] **Step 3: 接口加默认 no-op（storage_backend.h，紧邻 #2676 的 `RemoveAll()` 声明）**

`StorageBackendInterface`：
```cpp
// 降级给定 key 所属的 guaranteed bucket（翻 guaranteed=false），交由现成 LRU 驱逐。
// 默认 no-op：仅 BucketStorageBackend 有 guaranteed 桶级 pin（Phase 2）。
virtual void DowngradeKeys(const std::vector<std::string>& /*keys*/) {}
```
`BucketStorageBackend`（override 声明）：
```cpp
void DowngradeKeys(const std::vector<std::string>& keys) override;
```
`StorageBackendAdaptor`（透传，紧邻其 `RemoveAll` override）：
```cpp
void DowngradeKeys(const std::vector<std::string>& keys) override {
    if (storage_backend_) storage_backend_->DowngradeKeys(keys);
}
```

- [ ] **Step 4: `BucketStorageBackend::DowngradeKeys` 实现（storage_backend.cpp）**

```cpp
void BucketStorageBackend::DowngradeKeys(const std::vector<std::string>& keys) {
    if (keys.empty()) return;
    // 1. 共享锁下：key → bucket_id 聚合
    std::unordered_map<int64_t, std::unordered_set<std::string>> by_bucket;
    {
        SharedMutexLocker lock(&mutex_, shared_lock);
        for (const auto& key : keys) {
            auto it = object_bucket_map_.find(key);
            if (it != object_bucket_map_.end()) {
                by_bucket[it->second.bucket_id].insert(key);
            }
        }
    }
    // 2. 互斥锁下：仅当降级集 ⊇ bucket->keys 才翻转（延迟部分覆盖）
    SharedMutexLocker lock(&mutex_);  // 单参 = exclusive（见 mutex.h）
    for (auto& [bucket_id, downgrade_set] : by_bucket) {
        auto bit = buckets_.find(bucket_id);
        if (bit == buckets_.end()) continue;
        auto& bucket = bit->second;
        if (!bucket->guaranteed) continue;  // 已降级
        if (downgrade_set.size() != bucket->keys.size()) continue;  // 部分，延迟
        bool all = true;
        for (const auto& k : bucket->keys) {
            if (!downgrade_set.count(k)) { all = false; break; }
        }
        if (all) {
            bucket->guaranteed = false;  // 降级 → SelectEvictionCandidate 可选 → LRU 回收
            VLOG(1) << "DowngradeKeys: bucket " << bucket_id << " downgraded";
        }
    }
}
```

- [ ] **Step 5: `FileStorage::Heartbeat` STEP 2 轮询（file_storage.cpp，紧邻 #2676 的 `PollRemoveAll` 调用）**

```cpp
    // === STEP 2: Poll master-commanded guaranteed downgrades ===
    auto downgrade_result = client_->PollDowngradeKeys();
    if (downgrade_result && !downgrade_result->empty()) {
        storage_backend_->DowngradeKeys(*downgrade_result);
    }
    // （#2676 的 PollRemoveAll 轮询保留在其原位置）
```

- [ ] **Step 6: 跑测试确认通过**

Run: `cd mooncake-store && cmake --build build --target mooncake_store_tests && ./build/tests/guaranteed_downgrade_test --gtest_filter=GuaranteedDowngradeTest.WorkerDefersPartialBucketDowngrade`
Expected: PASS

- [ ] **Step 7: 回归 SSD 驱逐现有测试**

Run: `cd mooncake-store && ./build/tests/offload_on_evict_test ./build/tests/master_service_ssd_test`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add mooncake-store/include/storage_backend.h mooncake-store/src/storage_backend.cpp \
        mooncake-store/src/file_storage.cpp mooncake-store/tests/guaranteed_downgrade_test.cpp
git commit -m "feat(store): worker deferred bucket-level guaranteed downgrade (reuse LRU eviction)"
```

---

## Task 5: `BatchExpireGuaranteed` 显式 ops 失效（HTTP + RPC）

**Files:**
- Modify: `mooncake-store/include/master_service.h`、`src/master_service.cpp`（`BatchExpireGuaranteed` 实现）
- Modify: `mooncake-store/include/rpc_service.h`、`src/rpc_service.cpp`（透传 + 注册）
- Modify: `mooncake-store/include/master_admin_service.h`、`src/master_admin_service.cpp`（HTTP 端点，对照 #2676 `HandleRemoveAll`）
- Test: `mooncake-store/tests/guaranteed_downgrade_test.cpp`

- [ ] **Step 1: 写失败测试 —— 显式失效立即派发降级**

```cpp
TEST_F(GuaranteedDowngradeTest, BatchExpireGuaranteedDispatchesDowngrade) {
    auto* master = GetMasterService();
    UUID holder = GetHolderClientId();
    PutEndGuaranteed("k1", /*guaranteed_until_ms=*/600000);  // 长 TTL，未到期
    OffloadToSSD("k1", holder);
    auto resp = master->BatchExpireGuaranteed("tenant", {"k1"});
    ASSERT_TRUE(resp.has_value());
    EXPECT_EQ(resp->expired_count, 1u);
    auto keys = master->PollDowngradeKeys(holder);
    ASSERT_TRUE(keys.has_value());
    ASSERT_EQ(keys->size(), 1u);
    EXPECT_EQ((*keys)[0], MakeTenantScopedStorageKey("tenant", "k1"));
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd mooncake-store && cmake --build build --target mooncake_store_tests && ./build/tests/guaranteed_downgrade_test --gtest_filter=GuaranteedDowngradeTest.BatchExpireGuaranteedDispatchesDowngrade`
Expected: FAIL（`BatchExpireGuaranteed` 不存在）

- [ ] **Step 3: master 实现 `BatchExpireGuaranteed`**

`master_service.h`：
```cpp
struct BatchExpireGuaranteedResponse {
    uint32_t expired_count{0};
};
auto BatchExpireGuaranteed(const std::string& tenant_id,
                           const std::vector<std::string>& user_keys)
    -> tl::expected<BatchExpireGuaranteedResponse, ErrorCode>;
```
`master_service.cpp`（线性扫描 tenant metadata，exact `(tenant_id, user_key)` 匹配，置 `guaranteed_until_ = epoch` + 直接派发降级，复用 Task 3 的派发内联逻辑——抽 helper `EnqueueDowngrade(metadata, tenant_id, user_key)`）：
```cpp
auto MasterService::BatchExpireGuaranteed(
    const std::string& tenant_id,
    const std::vector<std::string>& user_keys)
    -> tl::expected<BatchExpireGuaranteedResponse, ErrorCode> {
    if (!enable_guaranteed_cache_) return BatchExpireGuaranteedResponse{};
    std::shared_lock<std::shared_mutex> shared_lock(snapshot_mutex_);
    const auto tenant = NormalizeRequestTenantId(tenant_id);
    std::unordered_set<std::string> key_set(user_keys.begin(), user_keys.end());
    uint32_t count = 0;
    for (size_t i = 0; i < kNumShards; ++i) {
        MetadataShardAccessorRW shard(this, i);
        auto tenant_it = shard->tenants.find(tenant);
        if (tenant_it == shard->tenants.end()) continue;
        for (auto& [user_key, metadata] : tenant_it->second.metadata) {
            if (!key_set.count(user_key)) continue;
            if (metadata.guaranteed_until_ == std::chrono::system_clock::time_point{}
                || metadata.downgrade_dispatched_) continue;
            metadata.guaranteed_until_ = {};  // 立即失效
            EnqueueDowngrade(metadata, tenant, user_key);  // helper: 反查 client_id + 入 pending + set flag
            ++count;
        }
    }
    return BatchExpireGuaranteedResponse{count};
}
```
> **重构**：把 Task 3 `DispatchGuaranteedExpiry` 里的"反查 client_id + 入 pending + set `downgrade_dispatched_`"抽成 `EnqueueDowngrade(ObjectMetadata&, tenant_id, user_key)`，Task 3 与本 Task 共用，消除重复。

- [ ] **Step 4: 贯通 wire（rpc_service / master_client）与 HTTP 端点**

`rpc_service.h/cpp`：`WrappedMasterService::BatchExpireGuaranteed` 透传 + `RegisterRpcService` 注册；`master_client.h/cpp`：`MasterClient::BatchExpireGuaranteed`（`invoke_rpc<..., BatchExpireGuaranteedResponse>(tenant_id, keys)`）。
`master_admin_service.h/cpp`：对照 #2676 `HandleRemoveAll` 加 `HandleBatchExpireGuaranteed` + `RegisterHandler` 注册路由 `POST /api/v1/batch_expire_guaranteed?tenant_id=...`（body 为 JSON key 列表）。

- [ ] **Step 5: 跑测试确认通过**

Run: `cd mooncake-store && cmake --build build --target mooncake_store_tests && ./build/tests/guaranteed_downgrade_test --gtest_filter=GuaranteedDowngradeTest.BatchExpireGuaranteedDispatchesDowngrade`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mooncake-store/include/master_service.h mooncake-store/src/master_service.cpp \
        mooncake-store/include/rpc_service.h mooncake-store/src/rpc_service.cpp \
        mooncake-store/include/master_client.h mooncake-store/src/master_client.cpp \
        mooncake-store/include/master_admin_service.h mooncake-store/src/master_admin_service.cpp \
        mooncake-store/tests/guaranteed_downgrade_test.cpp
git commit -m "feat(store): BatchExpireGuaranteed explicit ops invalidation (HTTP + RPC)"
```

---

## Task 6:（可选）读时续期 `guaranteed_until_`（config 门控，默认关）

> 仅当需要"活跃读取的 guaranteed 对象不被降级"时启用。默认 `guaranteed_renewal_ttl_ms=0` = 严格 TTL，本 Task 行为不生效。

**Files:**
- Modify: `mooncake-store/src/master_service.cpp:2497-2505`（`GetReplicaList` 续期点）
- Test: `mooncake-store/tests/guaranteed_downgrade_test.cpp`

- [ ] **Step 1: 写失败测试 —— 读时续期（config 开启时）**

```cpp
TEST_F(GuaranteedDowngradeTest, ReadRenewsGuaranteedUntilWhenEnabled) {
    SetConfigGuaranteedRenewalTtlMs(30000);  // 开启续期
    auto* master = GetMasterService();
    PutEndGuaranteed("k1", /*guaranteed_until_ms=*/10000);  // 10s TTL
    auto before = master->InspectObjectMetadata("k1", "tenant")->guaranteed_until;
    GetReplicaList("k1");  // 读
    auto after = master->InspectObjectMetadata("k1", "tenant")->guaranteed_until;
    EXPECT_GT(after, before);  // 被续期（向后推）
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd mooncake-store && cmake --build build --target mooncake_store_tests && ./build/tests/guaranteed_downgrade_test --gtest_filter=GuaranteedDowngradeTest.ReadRenewsGuaranteedUntilWhenEnabled`
Expected: FAIL

- [ ] **Step 3: 在 `GetReplicaList` 续期（紧邻现有 `GrantLease`，master_service.cpp:2501 之后）**

```cpp
        if (ts) {
            GrantLeaseForGroup(*ts, key, metadata);
        } else {
            metadata.GrantLease(default_kv_lease_ttl_, default_kv_soft_pin_ttl_);
        }
        // 可选：guaranteed 读时续期（config 门控）
        if (enable_guaranteed_cache_ && guaranteed_renewal_ttl_ms_ > 0) {
            const auto now_renew = std::chrono::system_clock::now();
            if (metadata.guaranteed_until_ > now_renew) {  // 仍是 guaranteed 才续
                metadata.guaranteed_until_ = std::max(
                    metadata.guaranteed_until_,
                    now_renew + std::chrono::milliseconds(guaranteed_renewal_ttl_ms_));
            }
        }
```
> `guaranteed_renewal_ttl_ms_` 在 `MasterService` 构造时从 config 读（参照 `default_kv_lease_ttl_` 的读法 [master_service.cpp:188](../../../mooncake-store/src/master_service.cpp#L188)）。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd mooncake-store && cmake --build build --target mooncake_store_tests && ./build/tests/guaranteed_downgrade_test --gtest_filter=GuaranteedDowngradeTest.ReadRenewsGuaranteedUntilWhenEnabled`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mooncake-store/src/master_service.cpp mooncake-store/tests/guaranteed_downgrade_test.cpp
git commit -m "feat(store): optional read-time guaranteed_until_ renewal (config-gated, default off)"
```

---

## Task 7: 回归保护 + feature flag 门控 + 端到端

**Files:**
- Test: `mooncake-store/tests/guaranteed_downgrade_test.cpp`
- Verify: `mooncake-store/src/master_service.cpp:6907-6960`（内存/quota 驱逐）

- [ ] **Step 1: 写回归测试 —— 内存驱逐不特殊保护 guaranteed（用户硬约束）**

```cpp
TEST_F(GuaranteedDowngradeTest, MemoryEvictionDoesNotProtectGuaranteed) {
    // guaranteed 内存 replica 在内存压力下应可被驱逐（guaranteed 是 SSD pin 概念）
    auto* master = GetMasterService();
    PutEndGuaranteed("k1", /*guaranteed_until_ms=*/600000);
    TriggerMemoryPressureEviction();  // 走 quota 驱逐路径
    // guaranteed 内存 replica 被驱逐（非 hard_pinned、lease 过期后可驱逐）
    EXPECT_FALSE(master->HasMemoryReplica("k1", "tenant"));
    // 但 SSD replica 仍 guaranteed（未到期未降级）
    auto meta = master->InspectObjectMetadata("k1", "tenant");
    EXPECT_GT(meta->guaranteed_until, std::chrono::system_clock::now());
}
```

- [ ] **Step 2: 核验内存驱逐代码无 guaranteed 特判**

人工核验 `master_service.cpp:6907-6960`（`try_evict_group_or_object` / `can_evict_replicas`）仅查 `IsHardPinned/IsLeaseExpired/IsSoftPinned`，**无 `guaranteed_until_` 判断**。若发现遗留 `guaranteed_` 引用，改为不影响驱逐的判断（guaranteed 不应阻止内存驱逐）。跑测试：
Run: `cd mooncake-store && cmake --build build --target mooncake_store_tests && ./build/tests/guaranteed_downgrade_test --gtest_filter=GuaranteedDowngradeTest.MemoryEvictionDoesNotProtectGuaranteed`
Expected: PASS

- [ ] **Step 3: 写 E2E 测试 —— 完整降级回收链路**

```cpp
TEST_F(GuaranteedDowngradeTest, EndToEndTTLExpireDowngradeAndRecycle) {
    auto* master = GetMasterService();
    UUID holder = GetHolderClientId();
    PutEndGuaranteed("k1", /*guaranteed_until_ms=*/1);
    OffloadToSSD("k1", holder);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    // master 周期扫描派发（或手动触发）
    master->DispatchGuaranteedExpiryForTest();
    // worker heartbeat 轮询降级
    PumpWorkerHeartbeat(holder);
    // 触发 SSD 压力 → 现成 LRU 回收已降级 bucket
    TriggerSSDPressureEviction(holder);
    // master 收到 BatchEvictDiskReplica 通知，metadata 被清
    EXPECT_FALSE(master->InspectObjectMetadata("k1", "tenant").has_value());
    // SSD 文件已删
    EXPECT_FALSE(GetBucketStorageBackend()->IsExist(MakeStorageKey("k1")));
}
```

- [ ] **Step 4: 跑 E2E**

Run: `cd mooncake-store && cmake --build build --target mooncake_store_tests && ./build/tests/guaranteed_downgrade_test --gtest_filter=GuaranteedDowngradeTest.EndToEndTTLExpireDowngradeAndRecycle`
Expected: PASS

- [ ] **Step 5: 全量回归 + CI**

Run: `bash scripts/run_ci_test.sh`
Expected: PASS（含既有 `guaranteed_offload_test`、`offload_on_evict_test`、`master_service_ssd_test`）

- [ ] **Step 6: feature flag 默认关闭核验**

确认 `enable_guaranteed_cache=false` 时：`AllocateAndInsertMetadata` 传 `guaranteed_until_ms=0`（`guaranteed_until_` 为 epoch）→ `DispatchGuaranteedExpiry` 早退 → `BatchExpireGuaranteed` 早退 → 全部新行为休眠，与 main 行为等价。补一个门控测试：
```cpp
TEST_F(GuaranteedDowngradeTest, FeatureFlagDefaultOffIsDormant) {
    SetEnableGuaranteedCache(false);
    auto* master = GetMasterService();
    PutEnd("k1");  // 非 guaranteed
    master->DispatchGuaranteedExpiryForTest();
    auto keys = master->PollDowngradeKeys(GetHolderClientId());
    ASSERT_TRUE(keys.has_value());
    EXPECT_TRUE(keys->empty());
}
```

- [ ] **Step 7: Commit**

```bash
git add mooncake-store/tests/guaranteed_downgrade_test.cpp
git commit -m "test(store): regression + E2E for guaranteed downgrade lifecycle; feature-flag dormancy"
```

---

## 实施顺序与依赖

```
Task 1 (guaranteed_until_ master-only)  ── 无外部依赖
   └─> Task 2 (PollDowngradeKeys RPC)   ── 依赖 #2676 已合入
         └─> Task 3 (周期派发)            ── 依赖 Task 1+2
               └─> Task 4 (worker 翻转)   ── 依赖 Task 2；可与 Task 5 并行
                     └─> Task 7 (E2E)     ── 依赖 Task 1-5
Task 5 (BatchExpireGuaranteed)           ── 依赖 Task 1+2（抽 EnqueueDowngrade helper）
Task 6 (可选续期)                        ── 依赖 Task 1，独立可延后
```

**建议执行顺序**：1 → 2 → 3 → 4 → 5 → 7 → 6（可选）。

## 待实施者确认的点（不阻塞，写代码时定）

1. `InspectObjectMetadata`（test-only view，含 `guaranteed_until`/`downgrade_dispatched`）与各场景 fixture helper 的具体实现——按 `guaranteed_offload_test.cpp` 既有内联模式写在新增的 `guaranteed_downgrade_test.cpp` fixture 里。
2. `BatchExpireGuaranteed` 的 wire 结构 `BatchExpireGuaranteedResponse` 用 `YLT_REFL` 尾随字段（`expired_count`），确保向后兼容。
3. Task 5 抽取的 `EnqueueDowngrade(ObjectMetadata&, tenant_id, user_key)` helper 与 Task 3 内联逻辑的合并顺序——建议 Task 3 先内联实现跑通，Task 5 再提取共用（已在 Task 5 Step 3 标注）。

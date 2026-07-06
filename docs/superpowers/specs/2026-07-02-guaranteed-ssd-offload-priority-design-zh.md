# Guaranteed 对象 SSD 写入保证 — 设计规范

**日期:** 2026-07-02
**父设计:** `docs/explicit_context_cache_design.md`（Phase 1，第 1 项收窄版）
**分支:** `remove_disk_data_in_ssd_offload`

## 1. 目标

让 guaranteed 对象在 `PutEnd` 时**保证成功进入 offload 队列**（异步写 SSD），SSD 写失败时**重试**，
写完后内存副本当普通对象驱逐。**核心保证**：

> **guaranteed 对象最终一定写入 SSD** —— 不会因 offload 队列满被拒（独立 guaranteed 队列无 limit），不会因 SSD 写
> 失败被放弃（失败重试）。写完后内存可正常释放，数据留在 SSD。

**设计意图**：guaranteed 的**生命周期由 SSD 管理**——这正是"确保写入 SSD"的根本原因：SSD 是生命周期
的载体，不写入就没有保护对象。内存只负责把 guaranteed 参数传到 offload 队列（HIGH 入队），写完 SSD
立即当普通对象（弱语义，不长期 pin 内存）。TTL/续期/驱逐保护/主动失效等生命周期逻辑将作用于 **SSD
副本**，属后续 slice（见 §13）——本 slice 是其**写入前提**：先保证数据进入 SSD，才有 SSD 侧生命周期
可管。

当前痛点：`PushOffloadingQueue` 在 `offloading_objects.size() >= offloading_queue_limit_`（默认 50000）
时返回 `KEYS_ULTRA_LIMIT`；`offload_force_evict_=true` 时 MEMORY replica 被直接删除不写 SSD —— 数据丢失。
本规范堵住"队列满丢入队"和"写失败丢保证"两个口。

## 2. 范围（大幅收窄）

父设计 Phase 1 第 1 项的核心 + 最小标记 + 失败重试。**eviction 零改动。**

| 项 | 状态 |
|------|--------|
| `PushOffloadingQueue` 独立 guaranteed 队列（无 limit/无抢占，核心） | ✅ 在范围内 |
| `OffloadTaskItem.guaranteed` 标记（master→client 传播，drain 携带） | ✅ 在范围内 |
| `ReplicateConfig.guaranteed_until_ms` 标记 | ✅ 在范围内 |
| `ObjectMetadata.guaranteed_` 布尔标记 | ✅ 在范围内 |
| client 端 `BuildBucket` 按 `guaranteed` 分流组同质 bucket | ⏭️ Phase 2（见 §6.4） |
| `AllocateAndInsertMetadata` 记录标记 | ✅ 在范围内 |
| `PutEnd` guaranteed 总走 offload + HIGH 优先级 | ✅ 在范围内 |
| `NotifyOffloadSuccess` guaranteed 失败重试 | ✅ 在范围内 |
| `IsHardPinned` 改 | ❌ 不要（内存靠 refcount pin，不靠 hard-pin） |
| `BatchEvict` 删除排除 / force-evict 旁路 | ❌ 不要（eviction 跟普通对象一样） |
| `GrantLease` 续期 + `GetReplicaListWithGuaranteed` RPC | ❌ 不要（不靠续期保生命周期） |
| `guaranteed_memory_used_` / 容量限制 | ❌ 不要（不长期占内存） |
| `guaranteed_until_` 时间戳 / TTL 语义 | ❌ 本 slice 不实现（Phase 2） |
| `BatchExpireGuaranteed` RPC | ❌ 推迟 |

## 3. 保证语义与已知边界

经验证的技术事实（见 §3.x）：

**① ✓ refcount pin 保护内存副本直到 SSD 写完** — `PushOffloadingQueue` 成功后 `inc_refcnt()`
（[3075](../../../mooncake-store/src/master_service.cpp)、[6824](../../../mooncake-store/src/master_service.cpp)、
[7069](../../../mooncake-store/src/master_service.cpp)），所有内存驱逐 guard 查 `get_refcnt()==0`
（[6761](../../../mooncake-store/src/master_service.cpp) 等）。SSD 写完 → `NotifyOffloadSuccess` 释放 pin
（[4858](../../../mooncake-store/src/master_service.cpp)）→ 内存副本变普通可驱逐对象。
**此链路现有机制已支撑，无需 `IsHardPinned`。**

**② ⚠ SSD 副本可被 fifo/lru 驱逐（已知边界，接受）** — LOCAL_DISK replica 无 pin 保护，SSD 满时 client
侧 fifo/lru 会驱逐它（`PrepareEviction`，[storage_backend.cpp:2227](../../../mooncake-store/src/storage_backend.cpp)）。
所以 guaranteed 的"保证"是**保证至少写入 SSD 一次**，不保证 SSD 副本永驻。SSD 容量压力下 guaranteed 副本
仍可能被 SSD 驱逐 —— 这与"后续 eviction 跟普通对象一样"的语义一致，本 slice 接受。若需 SSD 副本保护
（强语义），属后续工作。

**③ ✅ SSD 写失败重试（本 slice 修复）** — 现状 `NotifyOffloadSuccess` 收 NACK（`data_size<0`，
[4813](../../../mooncake-store/src/master_service.cpp)）时释放 pin + 删 task，不重排 → guaranteed 丢保证。
本 slice 改为：guaranteed 失败时重新 `PushOffloadingQueue` + `inc_refcnt`（重试）。

**④ ⚠ offload 任务 600s TTL 过期（已知边界，接受）** — 任务过期后 pin 被释放
（[5602](../../../mooncake-store/src/master_service.cpp)），极端 SSD 延迟（>600s）下 guaranteed 内存副本
可能提前变可驱逐，且重试也会因 task 已删而失效。本 slice 不动 TTL（用户选择）。已知限制，文档标注。

**降级场景**：`enable_offload_=false`（整个 offload 关闭）或 client `enable_offloading=false` 时，
`PushOffloadingQueue` 返回 `UNABLE_OFFLOADING`，guaranteed 无法入队 → 降级为普通对象（无保证）。
这是合理的（无 SSD 无法保证）。

## 4. 数据模型

### 4.1 `ReplicateConfig`（[mooncake-store/include/replica.h:81-144](../../../mooncake-store/include/replica.h)）

尾随字段（保留父设计形式，兼容 Phase 2 TTL）：

```cpp
int64_t guaranteed_until_ms{0};  // 0 = 非 guaranteed; >0 = guaranteed（本 slice 只用 >0 判断）
```

向后兼容：聚合体无 `YLT_REFL`，struct_pack 默认配置容忍尾随字段。**本 slice 不使用时间值，只用 `>0`
作为"该对象是 guaranteed"的布尔判断；TTL 语义（续期/过期驱逐）属 Phase 2。**

### 4.2 `ObjectMetadata`（[mooncake-store/include/master_service.h:851-1154](../../../mooncake-store/include/master_service.h)）

布尔标记（当前不用时间戳，YAGNI；Phase 2 升级 TTL 时改为 `time_point`）：

```cpp
const bool guaranteed_{false};  // true = PutEnd 保证 offload 到 SSD
```

构造函数加尾随参数 `bool guaranteed = false`（默认 false → 现有调用方不变）。`IsHardPinned()` 不动。

### 4.3 `OffloadTaskItem`（[mooncake-store/include/types.h:263-273](../../../mooncake-store/include/types.h)）

加 `guaranteed` 标记（入队时从对象 `guaranteed_` 快照，client 端 `BuildBucket` 据此分流组同质 bucket）：

```cpp
struct OffloadTaskItem {
    std::string tenant_id;
    std::string key;
    int64_t size;
    bool guaranteed{false};  // 从对象 guaranteed_ 快照
    // ... operator==、YLT_REFL 更新含 guaranteed
};
```

尾随 bool 字段，向后兼容。**无 `OffloadPriority` 枚举**——独立队列本身即分类，入队侧只需 `bool guaranteed`
决定入哪个 map，task 侧 `guaranteed` 标记供 client 分流。

## 5. `PutEnd` 路径：guaranteed 总走 offload

现状（[master_service.cpp:3064-3084](../../../mooncake-store/src/master_service.cpp)）：仅 `offload_on_evict_=false`
（默认）时 offload 所有 completed memory replica。`offload_on_evict_=true` 时 PutEnd 不 offload（延迟到驱逐）。

改为：guaranteed 对象**无视 `offload_on_evict_`**，PutEnd 时总 offload：

```cpp
if (enable_offload_ && (!offload_on_evict_ || metadata.guaranteed_)) {
    const bool guaranteed = metadata.guaranteed_;
    metadata.VisitReplicas(
        [](const Replica& r) { return r.is_completed() && r.is_memory_replica(); },
        [this, &object_id, &tenant_state, &task_created, guaranteed](Replica& replica) {
            auto result = PushOffloadingQueue(object_id, replica, guaranteed);
            if (result) {
                if (!task_created) {
                    replica.inc_refcnt();
                    tenant_state.offloading_tasks.emplace(
                        object_id.user_key,
                        OffloadingTask{replica.id(), std::chrono::system_clock::now()});
                    task_created = true;
                }
            }
        });
}
```

非 guaranteed 在 `offload_on_evict_=true` 时仍不 PutEnd offload（不变）。

## 6. `PushOffloadingQueue`：独立 guaranteed 队列

guaranteed 与 normal 分两个 per-client map，guaranteed 无 limit、无抢占。比"共享队列 + HIGH 抢占 LOW"
更简单，且有现成先例。

### 6.1 数据结构

每个 client 的 `LocalDiskSegment` 已有 `offloading_objects`（normal）和 `promotion_objects`
（promotion-on-hit，[master_service.cpp:5004](../../../mooncake-store/src/master_service.cpp) 先例）。新增
`guaranteed_offloading_objects`（per-client `unordered_map<string, OffloadTaskItem>`），与
`offloading_objects` 并列，受同一 `offloading_mutex_` 保护。

### 6.2 入队逻辑

`PushOffloadingQueue` 加 `bool guaranteed = false` 尾随参数（默认 false，向后兼容）：

- `guaranteed=true` → 入 `guaranteed_offloading_objects`，**不检查 limit**，永不返回 `KEYS_ULTRA_LIMIT`。
- `guaranteed=false` → 入 `offloading_objects`，`size() >= offloading_queue_limit_` 时返回
  `KEYS_ULTRA_LIMIT`（现有行为不变）。

无需抢占、无需扩容。guaranteed 队列无 limit 但隐式有界：队列里的 task 对应 memory replica 还在内存
（pin 着等 offload），队列长度 ≤ 内存中待 offload 的 guaranteed 对象数 ≤ 内存容量。

### 6.3 drain

`OffloadObjectHeartbeat`（[master_service.cpp:4708-4717](../../../mooncake-store/src/master_service.cpp)）扩展：
drain 时合并 `guaranteed_offloading_objects` 与 `offloading_objects` 返回（签名不变，仍是
`vector<OffloadTaskItem>`），task 带 `guaranteed` 标记。disable 分支同理清理两个 map + refcount
（复用 [4719-4749](../../../mooncake-store/src/master_service.cpp) 的 copy-then-release）。

### 6.4 client 端分流组桶（⏭️ Phase 2）

client `OffloadObjects`（[file_storage.cpp:361](../../../mooncake-store/src/file_storage.cpp)）按 `OffloadTaskItem.guaranteed`
把 tasks 分成两组（guaranteed / normal），各自独立走 `AllocateOffloadingBuckets` → `BatchOffload`，产出**同质 bucket**
（全 guaranteed 或全 normal）。`BuildBucket` 据 bucket 组别设 `BucketMetadata.guaranteed`。配合 Phase 2.B 的
`SelectEvictionCandidate` 跳过 guaranteed bucket。

**关键**：Phase 1 把 `guaranteed` 放到 `OffloadTaskItem`，但 `OffloadObjects` 当前降级成 `map<string,int64_t>` 时**丢弃了它**
（[file_storage.cpp:373](../../../mooncake-store/src/file_storage.cpp)）。Phase 2 必须先在 `OffloadObjects` 按 guaranteed 分组，
把标记穿到 `BuildBucket`/`BucketMetadata`。详见 §11 Phase 2.A。

**归入 Phase 2 而非 Phase 1**：在 Phase 1（无 client 侧 bucket pin）下，分流无可观察效果。且 `BuildBucket` 在
`storage_backend.cpp`（不同模块）。Phase 1 已通过 `OffloadTaskItem.guaranteed`（§4.3）+ `OffloadObjectHeartbeat` drain
（§6.3）把标记传到 client，Phase 2 在 `OffloadObjects` 消费即可。

## 7. `NotifyOffloadSuccess`：guaranteed 失败重试

现状 NACK 分支（`data_size < 0`，[master_service.cpp:4813-4830](../../../mooncake-store/src/master_service.cpp)）：
释放 pin（dec_refcnt）+ 删 task，不重排 → guaranteed 丢保证。

改为：guaranteed 失败时**重新入队等下一批 drain**，**pin 保持不动**（不 inc、不 dec）。PutEnd 时 inc 的
refcount 一直持有，直到 SSD 写成功才 dec。重新入队只刷新 task 的 `start_time` 重置 offload-task TTL（防
reaper 600s 删 task 导致内存副本提前变可驱逐）：

```cpp
if (metadata.data_size < 0) {
    MetadataAccessorRW accessor(this, request_object_id);
    if (accessor.Exists()) {
        auto& obj_metadata = accessor.Get();
        auto& tenant_state = accessor.GetTenantState();
        auto task_it = tenant_state.offloading_tasks.find(request_object_id.user_key);
        if (task_it != tenant_state.offloading_tasks.end()) {
            auto source = obj_metadata.GetReplicaByID(task_it->second.source_id);
            if (source != nullptr && obj_metadata.guaranteed_) {
                // guaranteed: 重新入队等下一批 drain。pin 保持不动（不 inc 不 dec）。
                auto result = PushOffloadingQueue(request_object_id, *source,
                                                   /*guaranteed=*/true);
                if (result || result.error() == ErrorCode::OBJECT_ALREADY_EXISTS) {
                    // 入队成功，或 key 仍在队列（NACK 早于 drain）——刷新 start_time 重置 TTL，等下一批
                    task_it->second = OffloadingTask{task_it->second.source_id,
                                                     std::chrono::system_clock::now()};
                    continue;  // 跳过 dec_refcnt/erase，pin 保持
                }
                // 入队失败（UNABLE_OFFLOADING 等）→ 降级
            }
            // 非 guaranteed 或降级：现有行为，dec_refcnt + erase task
            if (source != nullptr) source->dec_refcnt();
            tenant_state.offloading_tasks.erase(task_it);
        }
    }
    continue;
}
```

**为什么不用碰 refcount**：PutEnd 时 `inc_refcnt` 一次（refcnt=1），drain 给 client 写 SSD，失败 NACK，
重新入队等下一批 drain 再写……整个重试循环中 refcnt 一直是 1，直到 SSD 写成功 `NotifyOffloadSuccess` 走
success 分支才 `dec_refcnt`（[4858](../../../mooncake-store/src/master_service.cpp)）。不 inc 不 dec，无
双 inc 风险。pin 全程保护内存副本不被驱逐，直到 SSD 写完。

**重试无次数上限**：guaranteed 要"保证写入"。SSD 持续故障时 guaranteed 会一直 pin 内存等下一批——这是
"保证写入"的代价。持续故障需运维介入。可选退避/上限留作后续。

**drain 交互**：NACK 时 client 已通过 `OffloadObjectHeartbeat` 把 key 从 master 队列 drain 清空
（snapshot-and-clear），所以重新 `PushOffloadingQueue` 不会冲突；若 NACK 早于 drain（key 仍在队列），
返回 `OBJECT_ALREADY_EXISTS` —— 也视为"已在队列等下一批"，保持 pin 即可。

**TTL 交互**：刷新 `start_time` 重置 600s offload-task TTL，防 reaper 删 task。若 TTL 已过期（task 被
reaper 删），`task_it == end` → 进不了重试分支，走现状降级，guaranteed 当普通对象。这是边界 ④，接受。

## 8. 并发与锁序

- 锁序保持 **Shard Lock → `offloading_mutex_`**。失败重试在 shard 锁内调 `PushOffloadingQueue`
  （取 `offloading_mutex_`），顺序正确。
- `PushOffloadingQueue` 的入队（选 map + emplace）在 `offloading_mutex_` 下原子完成，无需 copy-then-release
  （独立队列无抢占，无被抢占项需清理）。
- `NotifyOffloadSuccess` 重试时已在 shard 锁内（通过 accessor），重入 `PushOffloadingQueue` 安全。
- `guaranteed_` 是 `const`（构造后不变），无需锁。

## 9. 向后兼容性

- `ReplicateConfig.guaranteed_until_ms` 默认 0 → 现有 PutStart 不变。
- `ObjectMetadata.guaranteed_` 默认 false → `IsHardPinned()` 不变，offload 行为不变。
- `OffloadTaskItem.guaranteed` 默认 false + 尾随容忍 → 新旧 client/server 互通。
- `PutEnd` 条件 `(!offload_on_evict_ || metadata.guaranteed_)`：`guaranteed_=false` 时退化为 `!offload_on_evict_`（原条件）。
- `PushOffloadingQueue` 新增 `bool guaranteed=false` 参数：现有调用方（[6821](../../../mooncake-store/src/master_service.cpp)、
  [7066](../../../mooncake-store/src/master_service.cpp) 驱逐路径）不传（默认 false），入 `offloading_objects`，行为不变。
- `NotifyOffloadSuccess` NACK 分支：`guaranteed_=false` 时走原路径（释放 pin + 删 task）。
- **零回归。**

## 10. 测试计划

镜像 `tests/offload_on_evict_test.cpp` 模式（公开 API：`PutStart`/`PutEnd`/`OffloadObjectHeartbeat`）。
新文件 `tests/guaranteed_offload_test.cpp`。用例：

1. **guaranteed 不受 normal 队列 limit 约束**（默认异步路径）—— 用 normal 对象把 `offloading_objects`
   填到 `offloading_queue_limit`，Put 一个 guaranteed 对象（`guaranteed_until_ms>0`），断言 `PutEnd` 时
   guaranteed 入 `guaranteed_offloading_objects` 成功（非 `KEYS_ULTRA_LIMIT`），`OffloadObjectHeartbeat` 返回
   含 guaranteed（带 `guaranteed` 标记）。
2. **normal 队列满仍 KEYS_ULTRA_LIMIT** —— `offloading_objects` 满 + 非 guaranteed Put →
   `KEYS_ULTRA_LIMIT`（不变），且不影响 guaranteed 队列。
3. **两队列独立** —— normal 队列满时，guaranteed 仍入队；guaranteed 队列有多个 task 时不挤占 normal
   队列的 limit 配额。
4. **guaranteed SSD 写失败重试** —— mock SSD 写失败（NACK），断言 guaranteed 重新入队（task 保留/刷新，
   pin 不释放），内存 replica 不被驱逐。
5. **guaranteed SSD 写成功后内存变普通可驱逐** —— guaranteed 写 SSD 成功，pin 释放，触发内存驱逐 →
   memory 副本被删，LOCAL_DISK 副本留存。
6. **offload_on_evict=true 时 guaranteed 仍 PutEnd offload** —— `offload_on_evict_=true`，Put guaranteed
   → `PutEnd` 入队（非延迟到驱逐）；对比非 guaranteed 不 PutEnd offload。
7. **非 guaranteed 在 offload_on_evict=true 时不 PutEnd offload** —— 不变（向后兼容）。
8. **默认零行为变化** —— `enable_guaranteed_cache=false`（默认）且 `guaranteed_until_ms=0` 时所有行为与
   现状一致（flag 门控的硬保证，便于 main 合并后先验证上游行为不变）。
9. **enable_offload=false 时 guaranteed 降级** —— guaranteed 不入队，当普通对象（无保证，不报错）。

> 用例 10（client 端 `BuildBucket` 分流组同质 bucket）随 §6.4 归入 **Phase 2**，不在 Phase 1 测试范围内。

## 11. 整体分阶段路线图

**设计主线**：guaranteed 对象的**生命周期由 SSD 管理**。这要求三步：(1) 先确保数据写入 SSD；(2) 在 SSD 上
保护 guaranteed 副本不被驱逐；(3) 管理 SSD 副本的 TTL（续期 + 过期降级 + 主动失效）。本 spec 覆盖全部，
分阶段实施，每阶段独立可测、可交付。

```
Phase 1: 确保写入 SSD（本 spec 的可交付 slice）
  ├─ 1.A 标记 + 数据模型
  ├─ 1.B PutEnd 保证入队 + 独立 guaranteed 队列
  └─ 1.C 失败重试
Phase 2: SSD 副本驱逐保护（guaranteed 写进 SSD 后不被 fifo/lru 删）
  ├─ 2.A 标记传到 client + per-key 元数据
  └─ 2.B client 侧 SelectEvictionCandidate 跳过 guaranteed
Phase 3: SSD 副本 TTL 管理（生命周期在 SSD）
  ├─ 3.A guaranteed_until 落到 SSD 副本 + 续期
  ├─ 3.B 过期降级（SSD 副本 TTL 到期后可驱逐）
  └─ 3.C BatchExpireGuaranteed 主动失效 RPC
Phase 4: SGLang HiCache 集成（端到端）
```

### Phase 1：确保写入 SSD（**本 spec 详细设计，立即实施**）

guaranteed 的"保证"落在 PutEnd 入队 + 失败重试，内存写完即普通（弱语义）。详见 §4–§10。

| 子阶段 | 改动 | 测试 |
|------|------|------|
| **1.A 标记** | config/gflag `enable_guaranteed_cache`（默认 false）；`ReplicateConfig.guaranteed_until_ms`（仅判 `>0`）；`ObjectMetadata.guaranteed_` 布尔；`AllocateAndInsertMetadata` 记录（全部门控在 `enable_guaranteed_cache_` 下） | 用例 8（零变化） |
| **1.B PutEnd + 独立队列** | `PutEnd` guaranteed 无视 `offload_on_evict_` 总 offload；新增 `guaranteed_offloading_objects` per-client map（promotion_objects 先例）；`PushOffloadingQueue` 加 `bool guaranteed` 参数，guaranteed 入独立队列无 limit、永不 `KEYS_ULTRA_LIMIT`；`OffloadObjectHeartbeat` 合并 drain 两 map，`OffloadTaskItem.guaranteed` 标记传给 client | 用例 1–3、6、7 |
| **1.C 失败重试** | `NotifyOffloadSuccess` NACK 分支：guaranteed 重新入独立队列、pin 保持、等下一批 drain | 用例 4、5 |

### Phase 2：SSD 副本驱逐保护（后续 slice）

**问题**：Phase 1 后 guaranteed 一定写进 SSD，但 SSD 副本可被 client 侧 fifo/lru 驱逐（边界 ②）。Phase 2
堵住这个口，让 guaranteed 的 SSD 副本在 TTL 内不被驱逐。

**关键架构约束**（已验证）：
- SSD 驱逐是 **client 侧** `BucketStorageBackend` 触发：`PrepareEviction`→`SelectEvictionCandidate`→
  `FinalizeEviction` 删文件（[storage_backend.cpp:1314-1326](../../../mooncake-store/src/storage_backend.cpp)）。
- master `BatchEvictDiskReplica` 是**事后通知**，返回错误**不阻止 client 删文件**（`FinalizeEviction`
  无条件删，handler 忽略返回值，[file_storage.cpp:481-489](../../../mooncake-store/src/file_storage.cpp)）。
- 所以保护**必须落在 client 侧 `SelectEvictionCandidate`**（[storage_backend.cpp:2181](../../../mooncake-store/src/storage_backend.cpp)），不是 master 拒绝。

**子阶段：**

**2.A `guaranteed` 标记穿到 client 侧 bucket（修复断链 + 同质 bucket 分组）**

**关键发现（已验证）**：Phase 1 已把 `guaranteed` 放到 `OffloadTaskItem`（[types.h:267](../../../mooncake-store/include/types.h)），但
client 侧 `FileStorage::OffloadObjects`（[file_storage.cpp:361-375](../../../mooncake-store/src/file_storage.cpp)）把它降级成
`unordered_map<string,int64_t>`（只留 key+size）传给 `AllocateOffloadingBuckets`——**`guaranteed` 在此处断链**，到不了
`BuildBucket`/`BucketMetadata`。Phase 2 必须修复这条链路。

- `OffloadObjects` 按 `task.guaranteed` 把 tasks 分成两组（guaranteed / normal），每组独立走
  `AllocateOffloadingBuckets` → `BatchOffload`，使**同质 bucket**（全 guaranteed 或全 normal）天然分离。这同时实现
  §6.4 的 client 端分流组桶。
- `BucketMetadata`（[storage_backend.h:33](../../../mooncake-store/include/storage_backend.h)）增加 `bool guaranteed{false};`。
  `BuildBucket`（[storage_backend.cpp:1978](../../../mooncake-store/src/storage_backend.cpp)）构造时设置（整 bucket 同质，单一 bool 即可，无需 `guaranteed_key_count`）。
- **持久化决策**：`BucketMetadata` 经 `YLT_REFL` 序列化到 `.meta` 文件，`BatchLoad` 重启恢复时读回
  （[storage_backend.cpp:1590](../../../mooncake-store/src/storage_backend.cpp)）。若**不**加进 `YLT_REFL`，重启后 guaranteed
  bucket 丢失标记变可驱逐——Phase 2 TTL 还没引入，丢失标记等于丢保护。故 **加进 `YLT_REFL`**（与 §14 维护性权衡：
  这是 client 本地 `.meta` 文件格式，不跨 master，合并风险可控）。注意 `BucketMetadata` 有自定义 copy/move ctor
  （因 atomic 成员），新增 bool 字段需在 4 个 ctor/assignment 里显式拷贝。

**2.B client 侧 SelectEvictionCandidate 跳过 guaranteed bucket**

**单一 chokepoint 确认**：`PrepareEviction`（[storage_backend.cpp:2227](../../../mooncake-store/src/storage_backend.cpp)）是唯一循环
`SelectEvictionCandidate` 的地方，纯触发式（无后台驱逐线程）。eviction 只在新 offload 需要空间时于 `BatchOffload` 同步触发
（[storage_backend.cpp:1317](../../../mooncake-store/src/storage_backend.cpp)）。故只需改 `SelectEvictionCandidate`
（[storage_backend.cpp:2181](../../../mooncake-store/src/storage_backend.cpp)）一处。guaranteed bucket 跳过后永不进
`PendingEviction`（既不 `FinalizeEviction` 删文件，也不 `eviction_handler` 通知 master），保护完整。

**两种驱逐策略不同处理**（已验证 [storage_backend.cpp:2183-2224](../../../mooncake-store/src/storage_backend.cpp)）：
- **FIFO**（[2187](../../../mooncake-store/src/storage_backend.cpp)）：当前 `return buckets_.begin();`。改为从 `begin()` 前向扫描，
  返回第一个 `!guaranteed` 的 bucket（`buckets_` 是 `std::map`，迭代便宜）。
- **LRU**（[2200-2220](../../../mooncake-store/src/storage_backend.cpp)）：遍历 `lru_index_`（`std::set<{ts, bucket_id}>`）。
  当解析到的 bucket 是 guaranteed 时，**不能 erase**（erase 会让该项永久从 LRU 索引丢失，因读路径不重插），改为 `++top_it`
  跳到下一项继续。非 guaranteed 项的 stale-repair 逻辑（erase+重插）保持不变。

**磁盘满失败模式（接受）**：guaranteed bucket 占满空间时，`SelectEvictionCandidate` 返回 `buckets_.end()`，`PrepareEviction`
循环 break（[2293](../../../mooncake-store/src/storage_backend.cpp)），`WriteBucket` 随后 ENOSPC 失败，offload 以失败上报。
这是"guaranteed 不可驱逐"的代价——Phase 2 接受此硬失败（Phase 3 TTL 过期后 guaranteed bucket 变可驱逐，缓解）。
后续可视情加 backpressure（offload 排队等空间），Phase 2 不做。

- TTL 在 Phase 3 引入；Phase 2 暂为"guaranteed SSD 副本永久保护"。

### Phase 3：SSD 副本 TTL 管理（后续 slice，"生命周期在 SSD"的核心）

**问题**：Phase 2 的 guaranteed SSD 副本永久保护，需要 TTL 才能自动回收 + 续期 + 主动失效。

**方案演进（重要）**：早期设想的 **client-side TTL**（TTL 落 client `StorageObjectMetadata` + 跨节点
heartbeat 同步 TTL + `SelectEvictionCandidate` 热路径 TTL 判断）已被探索验证**推翻**，改用
**master-driven downgrade**。推翻原因（探索验证）：

- **无 master→client push RPC**：所有 48 个 RPC handler 都在 master 侧，client 只 `Connect(master)`。无 push。
- **client 读路径不刷新 `object_bucket_map_` TTL**：`BatchQuery`/`BatchLoad` 只接 keys，拿不到 TTL；reader 与
  holder 常是不同 client，读路径刷新不可行。跨节点 TTL 同步只能靠 heartbeat 挎带——复杂且要改热路径。
- **`prefix_hash` 不是 blake3、不是 key 前缀**：是 HA OpLog 全 key XXH32 哈希（[oplog_manager.cpp:155](../../../mooncake-store/src/ha/oplog/oplog_manager.cpp)）。`BatchExpireGuaranteed` 必须按 exact `(tenant_id, user_key)` 线性扫描。
- **`ObjectMetadata.guaranteed_` 是 `const bool`**（Phase 1），HA 不含它（[master_service.cpp:8455-8478](../../../mooncake-store/src/master_service.cpp)）。

**采用方案：master-driven downgrade**

TTL **只在 master**（`ObjectMetadata.guaranteed_until_` 时间戳，object 级，非 HA 序列化，类似 `lease_timeout`，
**不向 worker 同步**）。到期/显式失效时 master 反查 holder `client_id`（`Replica::get_local_disk_client_id`）
→ 推入 per-client `pending_downgrade_keys` → worker heartbeat 轮询 `PollDowngradeKeys` 取回 key 列表 →
**延迟桶级翻转** `BucketMetadata.guaranteed=false`（仅当降级集覆盖 bucket 全部 key）→ **现成 LRU 驱逐路径自动回收**
（不新建 worker DeleteKey，不改 `SelectEvictionCandidate` 热路径判断）。内存驱逐对 guaranteed 不额外处理
（已验证 master_service.cpp:6940 仅查 `IsHardPinned`/`IsLeaseExpired`/`IsSoftPinned`）。

**净效果**：消除跨节点 TTL 同步、热路径 TTL 判断、heartbeat 返回结构改造（client-side TTL 最难部分），
代价是新增一条泛化通道（复用 PR #2676 `PollRemoveAll` 模式，bool 全清 → key 列表选择性降级）+ worker 一处
bool 翻转（冷路径）。**无 `Serializer<Replica>` bump 风险**（TTL 在 `ObjectMetadata`，非 Replica，非 HA）。

**子阶段**：

**3.A（已实现 Task 1）`guaranteed_until_` 时间戳升级（master-only）**
- `ObjectMetadata.guaranteed_`（const bool）→ `guaranteed_until_`（`system_clock::time_point`，非 const，默认 epoch）。
  HA 仍不含（运行时态，重启重置为 epoch，符合父设计 7.10）。
- 3 用点改 `> now`：`AllocateAndInsertMetadata`（`effective_guaranteed_ms = enable_guaranteed_cache_ ? config.guaranteed_until_ms : 0`）、PutEnd 条件、NACK 重试。
- **行为等价**：TTL 活性期间 `guaranteed_until_ > now` ≡ Phase 1 `guaranteed_ == true`，零回归。
- 不依赖 #2676，已实现并验证（4 单测通过）。

**3.B（阻塞 #2676）降级派发链路**
- Task 2：per-client `pending_downgrade_keys` + `PollDowngradeKeys` RPC（泛化 #2676 `PollRemoveAll`）。
- Task 3：周期到期扫描 `DispatchGuaranteedExpiry`（挂 `TaskCleanupThreadFunc`）→ 反查 holder `client_id` → 入 pending。
- Task 4：worker `DowngradeKeys` 延迟桶级翻转 `BucketMetadata.guaranteed=false` → 现成 LRU 回收。
- Task 5：`BatchExpireGuaranteed` 显式 ops 失效（HTTP + RPC，exact key 线性扫描，非 prefix_hash）。
- **阻塞 PR #2676**：`PollDowngradeKeys` 复用 `PollRemoveAll` 通道模式，#2676 合入后做 follow-up。

**3.C（待做 Task 6）读时续期**
- `GetReplicaList` 读 LOCAL_DISK 副本时，若 `guaranteed_until_` 未过期，续期 `= max(当前, now + renewal_ttl)`。
- 纯 master 侧，无 #2676 依赖，config 门控（`guaranteed_renewal_ttl_ms`，默认 0 = 严格 TTL）。

**完整 Phase 3 实施计划**：[plans/2026-07-06-guaranteed-ssd-master-driven-downgrade-zh.md](../plans/2026-07-06-guaranteed-ssd-master-driven-downgrade-zh.md)。
**已废弃的 client-side TTL 设计**：[plans/2026-07-06-guaranteed-ssd-offload-priority-phase3-design-zh.md](../plans/2026-07-06-guaranteed-ssd-offload-priority-phase3-design-zh.md)（⚠️ 仅留作对比）。
### Phase 4：SGLang HiCache 集成（端到端）
- HiCache Controller 写回路径判断 cache_control token 范围 → write_through + `guaranteed_until_ms`。
- HiCache 读取 L3 路径：请求带 cache_control 时调 `GetReplicaListWithGuaranteed` 续期。
- GUARANTEED_CAPACITY_EXCEEDED 降级为普通 PutStart（Phase 1 未含容量限制，Phase 3/4 视需要再加）。
- Router 侧 cache_control 解析 → cc_token_offsets。
- 端到端集成测试。

## 12. Phase 1 关键文件

| 文件 | 修改 |
|------|---------|
| [mooncake-store/include/replica.h](../../../mooncake-store/include/replica.h) | `ReplicateConfig.guaranteed_until_ms`（尾随） |
| [mooncake-store/include/master_service.h](../../../mooncake-store/include/master_service.h) | `ObjectMetadata.guaranteed_` 成员 + 构造参数；`LocalDiskSegment` 新增 `guaranteed_offloading_objects` per-client map；`PushOffloadingQueue` 签名加 `bool guaranteed`（`IsHardPinned` 不动）；`enable_guaranteed_cache_` 成员 |
| [mooncake-store/include/types.h](../../../mooncake-store/include/types.h) | `OffloadTaskItem.guaranteed` + `YLT_REFL` |
| [mooncake-store/src/master_service.cpp](../../../mooncake-store/src/master_service.cpp) | `enable_guaranteed_cache_` 从 config 读取（默认 false，门控全部 guaranteed 行为）；`AllocateAndInsertMetadata` 记录标记；`PutEnd` 条件 + guaranteed；`PushOffloadingQueue` 按 guaranteed 选 map（guaranteed 无 limit）；`OffloadObjectHeartbeat` 合并 drain 两 map；`NotifyOffloadSuccess` guaranteed 重试回独立队列；驱逐路径调用点不改（默认 false） |
| [mooncake-store/tests/guaranteed_offload_test.cpp](../../../mooncake-store/tests/) | 新测试文件 |

## 13. 阶段依赖与边界

**依赖链**：Phase 1（写入）→ Phase 2（SSD 保护）→ Phase 3（TTL）→ Phase 4（集成）。每阶段是前一阶段的
前提，但每阶段独立可交付价值。

**跨阶段一致性**：
- `guaranteed_until_ms` 字段在 Phase 1 引入（仅判 `>0`），Phase 3 升级为真正 TTL 语义（对象级
  `guaranteed_until_` 时间戳 + SSD 副本级 `guaranteed_until`）。Phase 1 用布尔 `guaranteed_` 是临时简化，
  Phase 3 替换为时间戳，YAGNI 原则。
- Phase 1 的 `OffloadTaskItem.guaranteed`（bool，client 分流组桶）与 Phase 2.A 的 client 侧
  `StorageObjectMetadata.guaranteed` 标记同源——Phase 2 设计时确认是否复用同一字段贯穿到 `object_bucket_map_`。

**各阶段接受边界**：
- **Phase 1**：边界 ②（SSD 副本可被驱逐）接受 —— 保证"至少写入 SSD 一次"。
- **Phase 2**：guaranteed SSD 副本永久保护（无 TTL），靠 Phase 3.C 主动失效回收 —— 需 ops 介入或等 Phase 3。
- **Phase 3**：TTL 跨节点时钟漂移用 grace period 缓解；600s offload-task TTL 窗口（边界 ④）仍存在但被
  Phase 1.C 重试的 `start_time` 刷新缓解。
- **所有阶段**：`enable_offload=false` 时 guaranteed 降级为普通对象（无 SSD 无法保证）。

## 14. 维护性：便于定期合并社区 main 分支

**约束**：整个显式缓存（guaranteed）特性**无法合入社区 main**，需在长期分支上维护，定期 `git merge main`。
因此**最小化合并冲突**是设计的硬约束，贯穿所有阶段。每次 main 合并的痛苦程度由"改动有多侵入核心热路径"
决定——`master_service.cpp`（~9500 行）、`storage_backend.cpp`、`serializer.cpp` 是上游高频改动文件，改动越
深越痛。

### 14.1 设计原则

1. **加法优于侵入** — 新增字段/函数/RPC/config，而非改既有函数体。新字段一律**尾随 + 默认值**（聚合体
   struct_pack 容忍尾随，`group_ids` 即此先例）。
2. **默认参数保持既有调用方不变** — `PushOffloadingQueue(..., bool guaranteed = false)` 等，新调用方
   传 `guaranteed=true`，既有调用方零改动，上游新增调用方拿默认 false 也不破坏。
3. **逻辑隔离到命名 helper** — guaranteed 专有逻辑提取为命名方法（如 `MaybeEnqueueGuaranteedOffload(...)`、
   `HandleGuaranteedOffloadFailure(...)`），热路径里只加**一处调用**而非内联逻辑。这样 main 改热路径函数体时，
   合并冲突只在一行调用上，而非整段逻辑。
4. **避开序列化格式 bump** — **最高风险点**是 `Serializer<Replica>`（LOCAL_DISK 硬编码 3 元素，
   [serializer.cpp:709](../../../mooncake-store/src/serializer.cpp)，版本锁定）。Phase 3 把 TTL 放 client 侧
   `StorageObjectMetadata`（本地、不参与 HA 序列化）而非 `LocalDiskDescriptor`，**完全规避**序列化改动。
   `ObjectMetadata` 的 `guaranteed_`/`guaranteed_until_` 是运行时状态（HA 重置），快照重建时构造为默认值，
   不改 `Serializer`（规划时验证快照路径）。
5. **新代码放新文件** — 测试新文件（已计划）；可考虑 `guaranteed_manager` 辅助模块集中 guaranteed 逻辑，
   把改动从 `master_service.cpp` 抽出。
6. **feature flag 门控** — 新增 config `enable_guaranteed_cache`（默认 false）+ gflag。所有 guaranteed 行为
   在 `enable_guaranteed_cache_` 门控下。上游合并时特性**休眠**（默认关），行为零变化，合并更安全、bisect
   更易、可随时关闭。代价是热路径多几个 `if`（轻微）。

### 14.2 各阶段风险与对策

| 阶段 | 改动点 | 合并风险 | 对策 |
|------|--------|---------|------|
| Phase 1 | `ReplicateConfig` 尾随字段 | 低 | 尾随+默认0 |
| Phase 1 | `PushOffloadingQueue` 签名 | 低 | 默认 param `guaranteed=false` |
| Phase 1 | `PutEnd`/`NotifyOffloadSuccess`/`AllocateAndInsertMetadata` 热路径 | 中 | 提 helper、单调用点 |
| Phase 1 | `ObjectMetadata` 快照序列化 | 中 | guaranteed_ 运行时态、重建默认 false |
| Phase 2 | `StorageObjectMetadata`/`OffloadTaskItem` 字段 | 低 | 尾随+YLT_REFL |
| Phase 2 | `SelectEvictionCandidate` | 中 | helper + `guaranteed_key_count` 计数 |
| Phase 3 | `GetReplicaList` 续期 | 中 | helper 调用点 |
| Phase 3 | ~~`Serializer<Replica>` bump~~ | ~~高~~ | **规避**：TTL 放 client 本地 |

### 14.3 分支策略

- 长期特性分支，**`git merge main`（非 rebase）** 定期同步，保留历史、无需 force-push、利于团队共享。
- 设计文档（本 spec、父设计）放 `docs/`，随分支合并，rationale 不丢。
- 合并冲突优先**加法解决**：保留上游改动 + 本特性新增部分，避免删除任一侧。
- feature flag 默认关，合并后先验证上游行为不变，再按需开 flag 测本特性。

### 14.4 验证护栏

- 每次合并 main 后跑 `scripts/run_ci_test.sh`（mooncake-ci-local skill），确保无回归。
- `enable_guaranteed_cache=false` 时全部既有测试必须原样通过（零行为变化的硬保证）。

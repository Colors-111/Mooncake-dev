# Guaranteed SSD Offload 优先级 — Phase 3 设计计划（中文）

> **本文档是设计计划，不含实现代码。** 用户要求先完善设计，重点分析 3.C 实现方式，再决定写代码。

## 目标

Phase 3 = SSD 副本 TTL 管理：让 Phase 2 永久保护的 guaranteed bucket 能 TTL 过期自动回收 + 续期 + 主动失效。实现"生命周期由 SSD 管理"的核心。

## 探索验证的关键事实（推翻了 spec 早期假设）

| 早期假设 | 探索发现 | 影响 |
|---|---|---|
| client 读路径刷新本地 TTL | `BatchQuery`/`BatchLoad` 只接 keys，拿不到 TTL；reader 与 holder 常是不同 client | **跨节点 TTL 同步只能靠 heartbeat 捎带** |
| `BatchExpireGuaranteed` 按 `prefix_hash`（blake3）匹配 | `prefix_hash` 是 HA OpLog 的全 key XXH32 哈希，非 key 前缀；store 无 blake3 | **必须按 exact `(tenant_id, user_key)` 线性扫描** |
| 新 RPC 通知 client 失效 | 无 master→client push RPC（48 handler 全在 master 侧），client 无 RPC server | **只能 heartbeat 捎带** |
| `ObjectMetadata.guaranteed_` 是 bool | 是 `const bool`（Phase 1），HA 不含 | 升级需去 const + 改 `time_point` |

## 整体设计

```
master                                      client (holder)
─────                                       ─────────────
ObjectMetadata.guaranteed_until_    ──►    StorageObjectMetadata.guaranteed_until_ms
  (time_point, 运行时, HA 不持久)            (object_bucket_map_, client 本地)
        │                                          │
        │ 续期 (GetReplicaList)                     │ 写回 (heartbeat 挺带)
        ▼                                          ▼
  GetReplicaListResponse.guaranteed_until_ms ──► BucketMetadata.guaranteed_until_ns_
  (YLT_REFL 兼容尾随字段)                       (runtime atomic, 非 YLT_REFL)
                                               SelectEvictionCandidate: now < ts?
```

**TTL 流转**：master 是 TTL 的权威（续期在此），client 是 TTL 的执行者（驱逐决策在此）。master→client 同步靠 heartbeat 挎带。

---

## 3.A guaranteed_until 时间戳升级 + 续期

### master 侧升级（`ObjectMetadata.guaranteed_` bool → `guaranteed_until_` time_point）

- **成员**：`master_service.h:915` `const bool guaranteed_{false}` → `std::chrono::system_clock::time_point guaranteed_until_{}`（去 const，默认 epoch = 无保证）。
- **构造函数**：`master_service.h:881` `guaranteed_(enable_guaranteed)` → `guaranteed_until_(...)`。尾随参数 `bool enable_guaranteed` → `int64_t guaranteed_until_ms`（或直接 `time_point`）。
- **`AllocateAndInsertMetadata`**（`master_service.cpp:2822`）：`enable_guaranteed_cache_ && config.guaranteed_until_ms > 0`（bool 归约）→ `now + std::chrono::milliseconds(config.guaranteed_until_ms)`（存时间戳，门控 flag 仍查）。
- **3 个用点改 `> now` 判断**：
  - PutEnd 条件（`master_service.cpp:3067`）：`metadata.guaranteed_` → `metadata.guaranteed_until_ > now`。
  - PutEnd lambda capture（`master_service.cpp:3075`）：`guaranteed = metadata.guaranteed_`（bool）→ 传 `guaranteed_until_ > now` 的 bool（或传时间戳）。
  - NACK 重试（`master_service.cpp:4845`）：`obj_metadata.guaranteed_` → `obj_metadata.guaranteed_until_ > now`。
- **HA 不含 `guaranteed_until_`**（运行时态，重启重置为 epoch，符合父设计 7.10）。已确认 `SerializeMetadata`（`master_service.cpp:8455-8478`）不含 `guaranteed_`。

### OffloadTaskItem 透传 TTL（master→client wire）

- **`OffloadTaskItem`**（`types.h:263`）：保留 `bool guaranteed`（Phase 1）+ 加尾随 `int64_t guaranteed_until_ms{0}`（YLT_REFL 兼容，比替换类型安全）。`operator==` 更新。
- **`PushOffloadingQueue`**（`master_service.cpp:4986`）：签名 `bool guaranteed` → 加 `int64_t guaranteed_until_ms`（或一并传）。emplace 设 `.guaranteed_until_ms`。
- **client `OffloadObjects`**（`file_storage.cpp:376`）：`if (task.guaranteed)` → `if (task.guaranteed_until_ms > 0)`（或保留 `task.guaranteed`，两者一致）。

### client 侧存 TTL

- **`StorageObjectMetadata`**（`types.h:543`）：加尾随 `int64_t guaranteed_until_ms{0}`（YLT_REFL，wire 兼容）。`BuildBucket` 从 `OffloadTaskItem.guaranteed_until_ms` 写入 `object_bucket_map_`。
- **`BucketMetadata`**（`storage_backend.h:38`）：保留 Phase 2 的 `bool guaranteed`（"曾 guaranteed"）+ 加 runtime atomic `int64_t guaranteed_until_ns_{0}`（**非 YLT_REFL**，runtime only，避免改磁盘格式）。`SelectEvictionCandidate` 改用时间戳判断。

### 续期（master GetReplicaList）

- **`GetReplicaList`**（`master_service.cpp:2501`）：`GrantLeaseForGroup`/`GrantLease` 后，加并行 `guaranteed_until_` 续期：若 `guaranteed_until_ > now`，`guaranteed_until_ = max(当前, now + renewal_ttl)`。仅对有 LOCAL_DISK 副本的对象（guaranteed 对象都在 SSD）。
- **`GetReplicaListResponse`**（`rpc_types.h:32`）：加尾随 `int64_t guaranteed_until_ms{0}`（YLT_REFL 兼容）。返回刷新后的 `guaranteed_until_`。

### 跨节点同步（heartbeat 挎带，非读路径）

- **关键限制**：client `BatchQuery`/`BatchLoad` 拿不到 `GetReplicaListResponse` 的 TTL，无法在读路径写回 `object_bucket_map_`。reader 与 holder 常是不同 client。
- **方案**：client 在 `OffloadObjectHeartbeat` 响应里拿 master 侧续期后的 `guaranteed_until_ms`，写回 `object_bucket_map_` + `BucketMetadata.guaranteed_until_ns_`。
- **延迟**：≤ heartbeat 间隔（默认秒级）。可接受（guaranteed TTL 通常分钟级，秒级同步足够）。
- **实现**：heartbeat 返回类型升级（见 3.C 方案 1 的 struct，或单独加 TTL 字段）。

---

## 3.B 过期降级

### client 侧（核心）

- `SelectEvictionCandidate`（`storage_backend.cpp:2181`）：Phase 2 的 `!bucket->guaranteed` 判断 → `now < bucket->guaranteed_until_ns_ + grace`。过期则当普通可驱逐。
- **grace period**（时钟漂移）：client 侧比较加 grace（如 30s），避免 master 刚续期 client 因 `system_clock` 差误驱逐。`guaranteed_until_ns_ + grace > now` 才保护。
- FIFO 前向扫描 + LRU 前向扫描逻辑保持（Phase 2 已实现），只改判断条件。

### master 侧

- `guaranteed_until_` 过期后，PutEnd/NACK 的 `> now` 判断自然降级（不再 guaranteed offload）。
- 无需改 `IsHardPinned`（Phase 1 决定不改，master 内存驱逐靠 lease/soft_pin，SSD 驱逐靠 client）。

---

## 3.C BatchExpireGuaranteed 主动失效（设计重点）

### 问题定义

ops 场景需立即失效 guaranteed 对象（更新 system prompt、RAG doc 错误、调试）。难点：**无 master→client push RPC，client 侧 `object_bucket_map_` 的 TTL 怎么清？**

master 置 `guaranteed_until_ = epoch` 简单（遍历 shard），但 client 侧本地 TTL 还在，不清则仍保护。

### 方案对比

#### 方案 1：heartbeat 挎带失效列表（推荐）

**机制**：
1. master 新增 per-client 待通知队列：`unordered_map<UUID, set<string>> expired_guaranteed_pending_`（client_id → 待通知失效 key 集合）。
2. `BatchExpireGuaranteed(tenant_id, keys)`：
   - 遍历 `tenant_state.metadata`，按 exact key 匹配，置 `guaranteed_until_ = epoch`。
   - 查 key → holder client_id 映射（`guaranteed_offloading_objects` 按 client 组织，可反查；或查 LOCAL_DISK replica 的 `client_id`）。
   - 把 key 加入对应 client 的 `expired_guaranteed_pending_`。
3. `OffloadObjectHeartbeat` 返回类型：`vector<OffloadTaskItem>` → struct `OffloadHeartbeatResponse { vector<OffloadTaskItem> tasks; vector<string> expired_guaranteed_keys; }`（YLT_REFL 兼容，尾随字段）。
4. master 在 client heartbeat 时，返回其 `expired_guaranteed_pending_` 并清空。
5. client 收到 `expired_guaranteed_keys`，清 `object_bucket_map_` 对应项的 `guaranteed_until_ms` + `BucketMetadata.guaranteed_until_ns_`。`SelectEvictionCandidate` 即可驱逐。

**优点**：近实时（≤ heartbeat 间隔）；复用 pull 架构；wire 兼容。
**缺点**：master 维护 per-client 待通知队列（内存，heartbeat 后清）；需 key→holder 映射。
**延迟**：秒级（heartbeat 间隔）。
**复杂度**：中。改 heartbeat 返回类型 + master 维护队列 + client 清理逻辑。

**难点：master 怎么知道 key 属于哪个 holder client？**
- 选项 A：`guaranteed_offloading_objects`（Phase 1）按 client_id 组织，但 offload 完成后 key 从该 map 移除（已写 SSD）。需查对象当前的 LOCAL_DISK replica 的 `client_id`（`LocalDiskReplicaData.client_id`）。
- 选项 B：`BatchExpireGuaranteed` 遍历 metadata 时，对每个匹配对象查其 LOCAL_DISK replica 的 client_id，加入对应 client 队列。O(K) 查找，K = 匹配对象数。

#### 方案 2：client 主动查询（heartbeat 时批量查）

**机制**：
1. client 维护本地 guaranteed key 集合（offload 时记录）。
2. heartbeat 时把集合发给 master，master 返回哪些已失效（`guaranteed_until_ <= now`）。
3. client 清失效项。

**优点**：无需 master 维护 per-client 队列。
**缺点**：每次 heartbeat 传 key 列表（数据量）；client 维护集合。
**延迟**：秒级。
**复杂度**：中。比方案 1 多传数据，少维护队列。

#### 方案 3：TTL 自然到期（最简，非立即）

**机制**：
1. `BatchExpireGuaranteed` 只置 master 侧 `guaranteed_until_ = epoch`。
2. 不通知 client。client 侧 TTL 自然到期后 `SelectEvictionCandidate` 驱逐。
3. master 侧 `GetReplicaList` 不再续期（已过期），client 下次 heartbeat 拿不到续期，本地 TTL 不再刷新。

**优点**：零跨节点通知改动；最简。
**缺点**：延迟 = client 本地 TTL 剩余（可能几分钟）。**不满足"立即失效"**。
**延迟**：分钟级。
**复杂度**：低。
**适用**：容忍几分钟延迟的场景（如非紧急的 prompt 更新）。

#### 方案 4：新 client RPC server（不推荐）

给 client 加 RPC server，master 主动 push。架构改动大，违背 client 无 server 设计。排除。

### 推荐

- **若需近实时失效**：方案 1（heartbeat 挎带）。master 维护 per-client 待通知队列，heartbeat 返回 struct 挎带 `expired_guaranteed_keys`。
- **若容忍分钟延迟**：方案 3（TTL 自然到期）。零通知改动，最简。
- **方案 2** 介于两者，无显著优势。

### `BatchExpireGuaranteed` 匹配方式（所有方案通用）

- **非 `prefix_hash`**（store 无 blake3 前缀结构，`prefix_hash` 是 HA OpLog 的全 key XXH32 哈希）。
- 按 `exact (tenant_id, user_key)` 或 `user_key` 子串匹配，**线性扫描 `tenant_state.metadata`**。
- O(N) 全量扫描，N = tenant 对象数。若需 prefix 匹配，可用 `user_key` 的 `starts_with`（substring，非 hash）。
- RPC：`BatchExpireGuaranteedRequest { string tenant_id; vector<string> user_keys; }` → `Response { uint32 expired_count; }`。

---

## 待决策点

1. **3.C 选方案 1 还是 3？** 近实时（方案 1，复杂）vs 容忍延迟（方案 3，简）。
2. **`OffloadTaskItem` 保留 `bool guaranteed` 还是替换为 `int64_t guaranteed_until_ms`？** 推荐保留 bool + 加尾随 ms（兼容）。
3. **`BucketMetadata` 加 `guaranteed_until_ns_` runtime atomic 还是改 `guaranteed` 为时间戳进 YLT_REFL？** 推荐加 runtime atomic（不改磁盘格式，维护性更好）。
4. **续期 TTL 值**：`renewal_ttl` 多长？默认 5 分钟（spec 父设计 300000ms）？
5. **grace period**：client 时钟漂移容忍 30s 够吗？

## 下一步

- 用户确认 3.C 方案后，写 Phase 3 实施计划（含代码），再 subagent-driven 实现。
- 若选方案 3，3.C 极简，Phase 3 主要是 3.A+3.B。
- 若选方案 1，3.C 需改 heartbeat 返回类型 + master per-client 队列 + client 清理，工作量与 3.A 相当。

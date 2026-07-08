# Guaranteed 对象 SSD 生命周期管理 — 架构设计（总领文档）

**日期:** 2026-07-06
**特性:** Mooncake Store guaranteed 对象的"生命周期由 SSD 管理"——确保写入 SSD、保护 SSD 副本不被驱逐、TTL 回收。
**状态总览:** [docs/superpowers/README.md](superpowers/README.md)
**父背景:** [docs/explicit_context_cache_design.md](../explicit_context_cache_design.md)（显式上下文缓存总设计，早期版本，部分内容已被本架构修订——见 §6 与父文档的差异）

---

## 1. 目标

为 Mooncake Store 的 guaranteed 对象实现完整的 SSD 生命周期管理：对象标记 guaranteed 后**一定写入 SSD**，
SSD 副本在 TTL 内**不被驱逐**，TTL 到期或显式失效后**自动降级回收**。这是 SGLang 显式上下文缓存
（`cache_control` 标记）的 L3 确定性保证层（父设计 [explicit_context_cache_design.md](../explicit_context_cache_design.md) §3.1）。

**核心保证链**：

> 写入 SSD（Phase 1）→ SSD 副本不被驱逐（Phase 2）→ TTL 到期降级回收（Phase 3）→ SGLang 集成（Phase 4）

**设计原则**：
- **guaranteed 生命周期由 SSD 管**——非内存。SSD 是生命周期权威，内存副本写完即普通（可释放）。
- **master 管驱逐时机，worker 复用既有删除路径**——master 不直接删 client 的 SSD 文件（无法 push），
  而是命令 worker"降级（撤销 pin）"，由 worker 现成 LRU 路径回收。
- **`enable_guaranteed_cache` flag**（默认 false）门控全部行为——零行为变化，可随时关闭。
- **维护性硬约束**：整个特性无法合入社区 main，需长期分支定期 `git merge main`。加法优于侵入、新字段尾随+默认、
  逻辑提取命名 helper、**避开 `Serializer<Replica>` 格式 bump**（最高合并风险点）。

### 1.1 特性总览

> 显式缓存特性（无法合入社区 main，需易维护设计：加法优于侵入、尾随+默认、避开 `Serializer<Replica>` bump、
> `enable_guaranteed_cache` flag 门控）。详见 §4 路线图 + §5 各 Phase 任务详解。

**已实现：**
- **a. 全链路 `guaranteed_ttl` + 优先级队列**（独立无 limit 队列）—— Phase 1
- **b. guaranteed 强制写入 SSD + 写失败重试**（`PutEnd` 总 offload + NACK 重试）—— Phase 1
- **c. SSD 副本驱逐保护**（bucket 级 pin，FIFO/LRU 跳过 guaranteed bucket）—— Phase 2
- **d. `guaranteed_until_` 时间戳 + 读时续期**（request 级 `renew_guaranteed_ttl_ms`，只续不创建）—— Phase 3 Task 1/6

**计划中（阻塞 PR #2676 / Phase 4）：**
- **e. TTL 到期降级回收**（master-driven downgrade：TTL 只在 master，到期下发一次性降级列表，worker 翻 bucket bool 后**现成 LRU 路径**回收）—— Phase 3 Task 2-5（阻塞 #2676）
- **f. 主动失效 `BatchExpireGuaranteed`**（HTTP+RPC，exact `(tenant_id, user_key)` 线性扫描，**非 prefix_hash**）—— Phase 3 Task 5（阻塞 #2676）
- **g. 新增 `PollDowngradeKeys` RPC + 与 SGLang 集成**（cache_control 解析 → `write_through` + `guaranteed_until_ms`；读 L3 时带 `renew_guaranteed_ttl_ms` 续期；Router 解析 token 断点）—— Phase 3 Task 2 + Phase 4

**不做（已决策）：**
- **容量限制**——弱语义：写完 SSD 即普通内存对象（可驱逐释放），靠 TTL 回收，不主动拒绝新 guaranteed。
- **改 `IsHardPinned()`**——它闸的是内存驱逐，与 SSD 写入保证（`PushOffloadingQueue` 优先级）正交，不改。

## 2. 设计定位：与早期父设计的演进

| 维度 | 父设计（explicit_context_cache_design.md） | 本架构（演进后） |
|------|--------------------------------------------|------------------|
| guaranteed 生命周期 | master 侧（`guaranteed_until_`） | 同——TTL 在 master `ObjectMetadata.guaranteed_until_` |
| `IsHardPinned()` | 增加 guaranteed 时间戳检查 | **不改**（闸的是内存驱逐，非 offload；与 SSD 写入保证正交） |
| BatchEvict 跳过 | BatchEvict 跳过未过期对象 | 不靠 BatchEvict——SSD 驱逐是 client 侧，master BatchEvict 是内存 |
| TTL 跨节点同步 | （隐含 client-side） | **master-driven downgrade**（TTL 只在 master，到期下发一次性降级列表） |
| 主动失效 | BatchExpireGuaranteed RPC | 同（exact key 线性扫描，非 prefix_hash） |
| 容量限制 | guaranteed_memory_used_/limit | **不要**（弱语义：写完 SSD 即普通，靠 TTL 回收） |
| Phase 划分 | Phase 1 基础设施 / Phase 2 HiCache / Phase 3 端点 | Phase 1 写入 / 2 驱逐保护 / 3 TTL / 4 HiCache 集成 |

**父文档的价值**：§1 目标、§2 与阿里百炼定位、§3 分层职责（L1/L2/L3）、cache_control 解析——这些仍是架构背景，
**未被推翻**。本架构文档聚焦 Mooncake Store 侧的 4-Phase 实施（父文档 §7/§11 已过时，以本架构 + superpowers spec 为准）。

## 3. 整体架构

### 3.1 分层职责

| 层级 | 位置 | guaranteed 中的角色 | 保证级别 |
|------|------|---------------------|---------|
| L1 (GPU) | SGLang Worker 本地 | 推理临时持有，正常驱逐 | 不参与保证 |
| L2 (CPU) | SGLang Worker 本地 | 同引擎命中加速 | 不参与保证 |
| L3 (Mooncake Store) | 分布式存储 | **确定性保证层**（SSD 为生命周期权威） | 硬保证 |

L1/L2 不参与保证（请求可能路由到任何引擎）。L3 是所有引擎共享，天然支持跨引擎。

### 3.2 架构图

```
  SGLang (cache_control 标记)                        Phase 4
       │
       ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Mooncake Store (L3)                                         │
  │                                                              │
  │  ┌─────────────────────────────────────────────────────┐    │
  │  │ Master (TTL 权威)                          Phase 3    │    │
  │  │  ObjectMetadata.guaranteed_until_ (time_point)  3.A ✅  │    │
  │  │  - PutEnd: TTL>now → 总走 offload (Phase 1)            │    │
  │  │  - NACK: TTL>now → 重试 (Phase 1)                      │    │
  │  │  - 到期/失效: 反查 holder client_id → pending_downgrade │    │
  │  │  - GetReplicaList: 读时续期 (Phase 3 Task 6 ✅)        │    │
  │  │  - BatchExpireGuaranteed RPC (Phase 3 Task 5, #2676)   │    │
  │  └───────────────┬─────────────────────────────────────┘    │
  │     offload       │           PollDowngradeKeys (#2676)      │
  │     (Phase 1 ✅)  │           (Phase 3 Task 2, #2676)        │
  │                  ▼                                          │
  │  ┌─────────────────────────────────────────────────────┐    │
  │  │ Worker / Client (SSD 执行者)                         │    │
  │  │  offload 队列: 独立 guaranteed 队列 (无 limit) Phase 1✅│   │
  │  │  BucketMetadata.guaranteed (bool, YLT_REFL 持久) P2 ✅ │    │
  │  │  - OffloadObjects: 分同质 bucket (Phase 2 ✅)          │    │
  │  │  - SelectEvictionCandidate: 跳过 guaranteed bucket    │    │
  │  │    (FIFO 前向扫描 / LRU 前向扫描不 erase) (Phase 2 ✅)  │    │
  │  │  - DowngradeKeys: 延迟桶级翻转 guaranteed=false        │    │
  │  │    → 现成 LRU 路径自动回收 (Phase 3 Task 4, #2676)     │    │
  │  └─────────────────────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────────────┘
```

### 3.3 关键架构约束（探索验证）

- **SSD 驱逐是 client 侧 bucket 粒度**：`BucketStorageBackend::SelectEvictionCandidate`（[storage_backend.cpp:2181](../../mooncake-store/src/storage_backend.cpp)）
  选整个 bucket，非 per-key。master `BatchEvictDiskReplica` 是事后通知，返回错误**不阻止 client 删文件**。
  → **SSD 副本保护必须落 client 侧 `SelectEvictionCandidate`**，不能靠 master 拒绝。
- **无 master→client push RPC**：48 个 handler 全在 master 侧，client 只 `Connect(master)`。无 push。
  → 跨节点同步（降级指令）只能 **heartbeat 捎带**（复用 PR #2676 `PollRemoveAll` 模式）。
- **`offloading_tasks` map mapped type 是 `const OffloadingTask`**（[master_service.h:1216](../../mooncake-store/include/master_service.h)）：
  不能原地赋值，必须 `erase + emplace`（Phase 1 实现中踩过的坑）。
- **`prefix_hash` 不是 blake3、不是 key 前缀**：是 HA OpLog 全 key XXH32（[oplog_manager.cpp:155](../../mooncake-store/src/ha/oplog/oplog_manager.cpp)）。
  → `BatchExpireGuaranteed` 按 exact `(tenant_id, user_key)` 线性扫描，非 prefix_hash。
- **`Serializer<Replica>` 是最高合并风险点**：LOCAL_DISK 硬编码 3 元素数组（[serializer.cpp:709](../../mooncake-store/src/serializer.cpp)），
  版本锁定。→ TTL 放 `ObjectMetadata`（非 Replica，非 HA），client 侧用 `BucketMetadata`（本地 `YLT_REFL`，非 HA Replica）。
- **LRU `SelectEvictionCandidate` 的 `while` 每次重置 `top_it=begin()`**：朴素 `++top_it; continue` 死循环。
  → guaranteed 跳过用前向扫描找首个非 guaranteed 候选（Phase 2 已实现）。
- **时间源取舍（`system_clock` vs `SteadyClock`）**：父设计建议 `SteadyClock`（[explicit_context_cache_design.md:341](../explicit_context_cache_design.md)）
  避免系统时间回拨，但 Task 1 实现用 `std::chrono::system_clock` 存 `guaranteed_until_`（[master_service.h:885](../../mooncake-store/include/master_service.h)）。
  **取舍**：`system_clock` 可被 NTP 跳变影响（回拨→TTL 异常延长，前跳→提前过期）；但 `guaranteed_until_` 是 master
  单机内比较（不跨节点，无时钟漂移同步问题），且与既有 `lease_timeout`（`system_clock`）一致。当前接受 `system_clock`
  的跳变风险（master 单点，回拨罕见且影响有限）。若需严格单调，可改 `steady_clock`，但与 `lease_timeout` 不一致、
  且 `steady_clock` 不能跨进程比较（HA 不持久化所以无碍）。**当前取舍：`system_clock`，文档承认此风险**。

## 4. 4-Phase 路线图

| Phase | 目标 | 状态 | 关键机制 |
|------|------|------|---------|
| 1 | 确保写入 SSD | ✅ 已实现验证 | 独立 guaranteed offload 队列（无 limit）+ PutEnd 总 offload + NACK 重试，`enable_guaranteed_cache` 门控 |
| 2 | SSD 副本驱逐保护 | ✅ 已实现验证 | `BucketMetadata.guaranteed`（`YLT_REFL` 持久）+ `OffloadObjects` 分同质 bucket + `SelectEvictionCandidate` 跳过（FIFO/LRU） |
| 3 | SSD 副本 TTL 管理 | 进行中（Task 1/6 ✅，Task 2-5 阻塞 #2676） | master-driven downgrade：TTL 在 master，到期下发降级列表，worker 翻转 bool 后现成 LRU 回收 |
| 4 | SGLang HiCache 集成 | 未开始 | HiCache 写回判断 cache_control → write_through + `guaranteed_until_ms`；读取续期；Router 解析 |

### Phase 3 细分（master-driven downgrade）

| 子任务 | 状态 | 依赖 |
|------|------|------|
| Task 1：`guaranteed_until_` 时间戳升级（master-only） | ✅ 已实现验证（4 单测） | 无 |
| Task 2：`PollDowngradeKeys` RPC（泛化 #2676 `PollRemoveAll`） | ⏳ 阻塞 PR #2676 | #2676 合入 |
| Task 3：周期到期扫描 `DispatchGuaranteedExpiry` | ⏳ 阻塞 #2676 | Task 1+2 |
| Task 4：worker `DowngradeKeys` 延迟桶级翻转 → 现成 LRU 回收 | ⏳ 阻塞 #2676 | Task 2 |
| Task 5：`BatchExpireGuaranteed` 显式 ops 失效（HTTP+RPC） | ⏳ 阻塞 #2676 | Task 1+2 |
| Task 6：读时续期 `GetReplicaList`（request 级 `renew_guaranteed_ttl_ms`） | ✅ 已实现验证（2 单测） | 无（纯 master） |

## 5. 各 Phase 任务详解

> 每个 Phase 拆成 task，说明"干啥的"。详细实现代码见各 Phase plan（[plans/](superpowers/plans/)）。

### Phase 1：确保写入 SSD（✅ 9 task + 补充用例，已实现验证）

master 侧：guaranteed 对象一定进 offload 队列、SSD 写失败重试、写完内存即普通。`enable_guaranteed_cache` 门控。

| Task | 干啥 | 状态 |
|------|------|------|
| 1 | `OffloadTaskItem` 加 `guaranteed` 布尔（master→client 线上标记，YLT_REFL） | ✅ |
| 2 | `ReplicateConfig` 加 `guaranteed_until_ms`（请求级标记，Phase 1 仅判 >0） | ✅ |
| 3 | `enable_guaranteed_cache` config flag 贯穿全 7 层（默认 false）+ gflag | ✅ |
| 4 | `ObjectMetadata` 加 `guaranteed_` 布尔（Phase 1），`AllocateAndInsertMetadata` 标记 | ✅ |
| 5 | `LocalDiskSegment` 加 `guaranteed_offloading_objects` 独立队列（per-client） | ✅ |
| 6 | `PushOffloadingQueue` 路由：guaranteed 进独立队列（无 limit），normal 限 limit | ✅ |
| 7 | `PutEnd` 总 offload guaranteed（无视 `offload_on_evict_`）—— 核心保证 | ✅ |
| 8 | `OffloadObjectHeartbeat` drain 两个 map + disable 时清理两个（refcount + task） | ✅ |
| 9 | SSD 写 NACK 时重新入队 guaranteed（pin 保持、刷 start_time、等下一批） | ✅ |
| 补充 | 用例 5（写成功后变可驱逐）+ 用例 9（`enable_offload=false` 降级） | ✅ |

### Phase 2：SSD 副本驱逐保护（✅ 4 task，已实现验证）

client 侧：guaranteed bucket 不被 fifo/lru 驱逐。

| Task | 干啥 | 状态 |
|------|------|------|
| 1 | `BucketMetadata` 加 `guaranteed` 布尔（`YLT_REFL` 持久化，重启不丢）+ 4 ctor/assign | ✅ |
| 2 | `OffloadObjects` 按 `guaranteed` 分两组 → **同质 bucket**（全 guaranteed 或全 normal） | ✅ |
| 3 | `BatchOffload` + `BuildBucket` 把 `guaranteed` 穿到 `BucketMetadata`（base + 全部 4 子类 override） | ✅ |
| 4 | `SelectEvictionCandidate` 跳过 guaranteed bucket（FIFO 前向扫描；LRU 前向扫描**不 erase**——防死循环+保 Phase 3 可驱逐） | ✅ |

### Phase 3：SSD 副本 TTL 管理（master-driven downgrade，进行中）

TTL 只在 master（object 级，类似 lease），到期下发一次性降级列表，worker 翻转 bucket bool 后由**现成 LRU 路径**回收。

| Task | 干啥 | 状态 | 依赖 |
|------|------|------|------|
| 1 | `ObjectMetadata.guaranteed_`(bool)→`guaranteed_until_`(time_point)，3 用点改 `>now`，config 全层 | ✅ 已验证 | 无 |
| 2 | per-client `pending_downgrade_keys` + `PollDowngradeKeys` RPC（泛化 #2676 `PollRemoveAll`：bool 全清→key 列表） | ⏳ | #2676 |
| 3 | 周期到期扫描 `DispatchGuaranteedExpiry`（挂 `TaskCleanupThreadFunc`）→ 反查 holder `client_id` → 入 pending | ⏳ | Task 1+2 |
| 4 | worker `DowngradeKeys`：聚合 key→bucket，**仅当降级集覆盖 bucket 全部 key** 才翻 `guaranteed=false` → 现成 LRU 回收 | ⏳ | Task 2 |
| 5 | `BatchExpireGuaranteed` 显式 ops 失效（HTTP + RPC，exact key 线性扫描，**非 prefix_hash**） | ⏳ | Task 1+2 |
| 6 | 读时续期：`GetReplicaList` 加 request 级 `renew_guaranteed_ttl_ms`（只续不创建、`std::max` 不缩短） | ✅ 已验证 | 无 |
| 7 | 回归保护 + feature flag 门控 + E2E（全链路降级回收测试） | ⏳ | Task 1-5 |
| Minor | `master.cpp` gflag 接线（`guaranteed_until_ms`/`renewal_ttl_ms`） | ✅（Task 6 顺带） | — |

### Phase 4：SGLang HiCache 集成（未开始，无 task 分解）

端到端：HiCache 写回判断 cache_control token 范围 → write_through + `guaranteed_until_ms`；读取 L3 时带 cache_control 传 `renew_guaranteed_ttl_ms` 续期；Router 解析 `cache_control` → token 断点。
**注意**：cache_control 字符偏移需在 SGLang 最终 prompt 构造后用 tokenizer offset mapping 解析（非字符直映）；"radix tree 前缀匹配无窗口限制"但断点数有上限（遵循 OpenAI 4 个）。task 待 Phase 3 完成后分解。

## 6. 关键设计决策

1. **guaranteed 生命周期由 SSD 管**（非内存）：写入 SSD（P1）→ 保护 SSD 副本（P2）→ TTL 回收（P3）。内存副本写完即普通。
2. **Phase 3 = master-driven downgrade**（非 client-side TTL）：TTL 只在 master（object 级，类似 lease，不进 HA），
   到期下发一次性降级列表，worker 翻转 bucket bool 后由**现成 LRU 路径**回收。消除跨节点 TTL 同步与
   `SelectEvictionCandidate` 热路径改动。
   - **bucket 粒度 TTL 放大（已知取舍）**：降级是 **bucket 级延迟翻转**——仅当降级集覆盖 bucket 全部 key 才翻
     `guaranteed=false`。Phase 2 把 guaranteed 对象混入同质 bucket，故同 bucket 内**已过期对象会被未过期对象"续命"**，
     实际 SSD 回收晚于对象 TTL。这是 **bucket-level TTL upper-bound**（回收 ≤ bucket 内最晚过期对象的 TTL），
     非对象级精确 TTL。代价是空间回收延迟，收益是简单（无需 bucket 分裂）。若放大不可接受，Phase 3 后可按过期
     时间分桶或 guaranteed 单独小桶，但当前接受此放大。
3. **`IsHardPinned()` 不改**：闸的是内存驱逐，不是 offload。SSD 写入保证来自 `PushOffloadingQueue` 优先级，
   与 `IsHardPinned` 正交。（曾误判"IsHardPinned=true 会让 guaranteed 永不 offload"——错，PutEnd offload 不查 IsHardPinned。）
4. **弱语义**：写完 SSD 即普通内存对象（可驱逐释放），不做容量限制（靠 TTL 回收，不主动拒绝）。
5. **`enable_guaranteed_cache` flag**（默认 false）门控全部——合并 main 后特性休眠，零行为变化。
6. **避开 `Serializer<Replica>` bump**：TTL 在 `ObjectMetadata`（非 Replica，非 HA），client 侧 `BucketMetadata`（本地 `YLT_REFL`）。
7. **#2676 阻塞 Phase 3 Task 2-5**：`PollDowngradeKeys` 复用 `PollRemoveAll` 通道模式，#2676 合入后做 follow-up。
8. **维护性硬约束：整个显式缓存特性无法合入社区 main**——需长期特性分支定期 `git merge main`（非 rebase，
   保留历史）。因此**最小化合并冲突**是设计的硬约束，贯穿所有 Phase：加法优于侵入（新字段/函数/RPC 而非改函数体）、
   新字段尾随+默认值（struct_pack 容忍尾随）、逻辑提取命名 helper（热路径只加一处调用而非内联大段逻辑）、
   feature flag 门控（默认关，上游合并时特性休眠、零行为变化、易 bisect）。`master_service.cpp`/`storage_backend.cpp`/
   `serializer.cpp` 是上游高频改动文件，改动越深越痛——尤其避开 `Serializer<Replica>` 格式 bump（决策 6）。
   详见 [spec §14 维护性](superpowers/specs/2026-07-02-guaranteed-ssd-offload-priority-design-zh.md#14-维护性便于定期合并社区-main-分支)。

## 7. 与父文档（explicit_context_cache_design.md）的差异

父文档是设计**起点**（2026-07-02），其 §7（Mooncake Store 修改）/§11（实施分阶段）描述的是早期父设计，**已过时**：

- 父 §11 "IsHardPinned() 增加时间戳检查" → 本架构**不改 IsHardPinned**（决策 3）。
- 父 §11 "guaranteed_until_ 落 client StorageObjectMetadata + GrantLease guaranteed_ttl" → **master-driven downgrade**
  推翻（TTL 只在 master，不向 worker 同步，决策 2）。
- 父 §11 "guaranteed_memory_used_/limit 容量检查" → **不要容量限制**（弱语义，决策 4）。
- 父 §11 Phase 划分（1 基础设施/2 HiCache/3 端点） → 本架构 4-Phase（1 写入/2 驱逐/3 TTL/4 HiCache）。

**父文档仍有效部分**：§1 目标、§2 与阿里百炼定位、§3 分层职责、§4 cache_control 解析（解析伪代码的理想化
见下方注）、§5 请求处理流程的 **SGLang 写入路径大意有效**。SGLang 侧零新增状态、radix tree 节点无新增字段——
这些设计原则保留。

**父文档 §5 已过时部分（以本架构 + master-driven-downgrade plan 为准）**：§5.3 读取续期流程（`GrantLease`
自动续期、`guaranteed_ttl_ms` 参数）—— 实际是 **request 级续期**（Phase 3 Task 6：只有带 cache_control 的请求传
`renew_guaranteed_ttl_ms` 才续，非全局自动续）；§5.4 过期降级（`IsHardPinned` 过期降级、`BatchEvict` 回收）——
实际是 **master-driven downgrade**（决策 2：master 到期下发降级列表，worker 翻转 bucket bool 后现成 LRU 回收，
不改 `IsHardPinned`、不靠 `BatchEvict`）。

**Phase 4 注意（cache_control 解析）**：父文档 §4 按字符偏移提取断点（[explicit_context_cache_design.md:92](../explicit_context_cache_design.md)），
但真实 chat prompt 经 chat template 含 role/special tokens/工具/图片块，字符偏移**不能直接映射到最终 prompt
token**。Phase 4 集成时需在 SGLang **最终 prompt 构造后**用 tokenizer offset mapping 解析，或把断点绑定到构造
prompt 的同一阶段。另：父文档"不做窗口限制"（§1）与"保留最后 4 个断点"（§4.2 `len > 4` 截断）语义冲突，Phase 4
需统一（建议遵循 OpenAI 语义保留 4 个断点，文档"不做窗口限制"改为"radix tree 前缀匹配无窗口限制"——匹配无窗口，
断点数有上限）。

## 8. 文档导航

| 文档 | 用途 |
|------|------|
| 本文档 | 总领架构（4-Phase 路线、§5 各 Phase 任务详解、关键决策、与父文档差异） |
| [superpowers/README.md](superpowers/README.md) | 实施文档索引 + 状态总览 |
| [superpowers/specs/2026-07-02-guaranteed-ssd-offload-priority-design-zh.md](superpowers/specs/2026-07-02-guaranteed-ssd-offload-priority-design-zh.md) | Phase 1 详细设计（数据模型、PutEnd、PushOffloadingQueue、测试计划、§14 维护性） |
| [superpowers/plans/](superpowers/plans/) | 各 Phase 实施计划（Phase 1/2 ✅，Phase 3 master-driven-downgrade 进行中） |
| [explicit_context_cache_design.md](../explicit_context_cache_design.md) | 显式上下文缓存总设计（SGLang + Mooncake，早期版本，§1-§5 仍有效） |

## 9. 测试

| 测试 target | Phase | 内容 |
|------------|------|------|
| `guaranteed_offload_test` | 1 | 12 测试（含补充用例 5&9） |
| `guaranteed_eviction_test` | 2 | 4 测试（字段 + FIFO/LRU 跳过） |
| `guaranteed_downgrade_test` | 3 Task 1 | 4 单测（`guaranteed_until_` 设置/降级/flag off/missing key） |

本机 build 环境坏（缺 msgpack + yalantinglibs，`dependencies.sh` 未跑），全部靠读代码审查 + 用户在 build 机器验证。

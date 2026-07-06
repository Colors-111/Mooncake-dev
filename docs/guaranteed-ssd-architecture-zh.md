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
  │  │  - GetReplicaList: 读时续期 (Phase 3 Task 6, 待做)      │    │
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

## 4. 4-Phase 路线图

| Phase | 目标 | 状态 | 关键机制 |
|------|------|------|---------|
| 1 | 确保写入 SSD | ✅ 已实现验证 | 独立 guaranteed offload 队列（无 limit）+ PutEnd 总 offload + NACK 重试，`enable_guaranteed_cache` 门控 |
| 2 | SSD 副本驱逐保护 | ✅ 已实现验证 | `BucketMetadata.guaranteed`（`YLT_REFL` 持久）+ `OffloadObjects` 分同质 bucket + `SelectEvictionCandidate` 跳过（FIFO/LRU） |
| 3 | SSD 副本 TTL 管理 | 进行中 | master-driven downgrade：TTL 在 master，到期下发降级列表，worker 翻转 bool 后现成 LRU 回收 |
| 4 | SGLang HiCache 集成 | 未开始 | HiCache 写回判断 cache_control → write_through + `guaranteed_until_ms`；读取续期；Router 解析 |

### Phase 3 细分（master-driven downgrade）

| 子任务 | 状态 | 依赖 |
|------|------|------|
| Task 1：`guaranteed_until_` 时间戳升级（master-only） | ✅ 已实现验证（4 单测） | 无 |
| Task 2：`PollDowngradeKeys` RPC（泛化 #2676 `PollRemoveAll`） | ⏳ 阻塞 PR #2676 | #2676 合入 |
| Task 3：周期到期扫描 `DispatchGuaranteedExpiry` | ⏳ 阻塞 #2676 | Task 1+2 |
| Task 4：worker `DowngradeKeys` 延迟桶级翻转 → 现成 LRU 回收 | ⏳ 阻塞 #2676 | Task 2 |
| Task 5：`BatchExpireGuaranteed` 显式 ops 失效（HTTP+RPC） | ⏳ 阻塞 #2676 | Task 1+2 |
| Task 6：读时续期 `GetReplicaList`（config 门控，默认关） | ⏳ 待做 | 无（纯 master） |

## 5. 关键设计决策

1. **guaranteed 生命周期由 SSD 管**（非内存）：写入 SSD（P1）→ 保护 SSD 副本（P2）→ TTL 回收（P3）。内存副本写完即普通。
2. **Phase 3 = master-driven downgrade**（非 client-side TTL）：TTL 只在 master（object 级，类似 lease，不进 HA），
   到期下发一次性降级列表，worker 翻转 bucket bool 后由**现成 LRU 路径**回收。消除跨节点 TTL 同步与
   `SelectEvictionCandidate` 热路径改动。
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

## 6. 与父文档（explicit_context_cache_design.md）的差异

父文档是设计**起点**（2026-07-02），其 §7（Mooncake Store 修改）/§11（实施分阶段）描述的是早期父设计，**已过时**：

- 父 §11 "IsHardPinned() 增加时间戳检查" → 本架构**不改 IsHardPinned**（决策 3）。
- 父 §11 "guaranteed_until_ 落 client StorageObjectMetadata + GrantLease guaranteed_ttl" → **master-driven downgrade**
  推翻（TTL 只在 master，不向 worker 同步，决策 2）。
- 父 §11 "guaranteed_memory_used_/limit 容量检查" → **不要容量限制**（弱语义，决策 4）。
- 父 §11 Phase 划分（1 基础设施/2 HiCache/3 端点） → 本架构 4-Phase（1 写入/2 驱逐/3 TTL/4 HiCache）。

**父文档仍有效部分**：§1 目标、§2 与阿里百炼定位、§3 分层职责、§4 cache_control 解析、§5 请求处理流程
（SGLang 侧重，Phase 4 集成时用）。SGLang 侧零新增状态、radix tree 节点无新增字段——这些设计原则保留。

## 7. 文档导航

| 文档 | 用途 |
|------|------|
| 本文档 | 总领架构（4-Phase 路线、关键决策、与父文档差异） |
| [superpowers/README.md](superpowers/README.md) | 实施文档索引 + 状态总览 |
| [superpowers/specs/2026-07-02-guaranteed-ssd-offload-priority-design-zh.md](superpowers/specs/2026-07-02-guaranteed-ssd-offload-priority-design-zh.md) | Phase 1 详细设计（数据模型、PutEnd、PushOffloadingQueue、测试计划、§14 维护性） |
| [superpowers/plans/](superpowers/plans/) | 各 Phase 实施计划（Phase 1/2 ✅，Phase 3 master-driven-downgrade 进行中） |
| [explicit_context_cache_design.md](../explicit_context_cache_design.md) | 显式上下文缓存总设计（SGLang + Mooncake，早期版本，§1-§5 仍有效） |

## 8. 测试

| 测试 target | Phase | 内容 |
|------------|------|------|
| `guaranteed_offload_test` | 1 | 12 测试（含补充用例 5&9） |
| `guaranteed_eviction_test` | 2 | 4 测试（字段 + FIFO/LRU 跳过） |
| `guaranteed_downgrade_test` | 3 Task 1 | 4 单测（`guaranteed_until_` 设置/降级/flag off/missing key） |

本机 build 环境坏（缺 msgpack + yalantinglibs，`dependencies.sh` 未跑），全部靠读代码审查 + 用户在 build 机器验证。

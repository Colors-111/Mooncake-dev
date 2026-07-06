# Guaranteed SSD Offload Priority — 文档索引

> 本目录是 Mooncake Store "guaranteed 对象 SSD 生命周期管理"特性的设计与实施文档。
> 特性分 4 个 Phase，依赖链 Phase 1 → 2 → 3 → 4。**整个特性无法合入社区 main**，需长期分支定期
> `git merge main`（非 rebase），见 spec §14 维护性策略。
>
> **总领架构文档**：[../guaranteed-ssd-architecture-zh.md](../guaranteed-ssd-architecture-zh.md)（4-Phase 路线、
> 关键设计决策、与父文档 explicit_context_cache_design.md 的差异、关键架构约束）。先读它，再深入本目录的 spec/plans。


## 当前状态总览（2026-07-06）

| Phase | 状态 | 说明 |
|------|------|------|
| Phase 1：确保写入 SSD | ✅ 已实现并验证 | master 侧，12 测试通过 |
| Phase 2：SSD 副本驱逐保护 | ✅ 已实现并验证 | client 侧，4 测试通过 |
| Phase 3 Task 1：`guaranteed_until_` 时间戳 | ✅ 已实现并验证 | 4 单测通过，无 #2676 依赖 |
| Phase 3 Task 2-5：降级派发链路 | ⏳ 阻塞于 PR #2676 | `PollDowngradeKeys` 通道依赖 #2676 合入 |
| Phase 3 Task 6：读时续期 | ⏳ 待做 | 纯 master 侧，无 #2676 依赖 |
| Phase 4：SGLang HiCache 集成 | ⏳ 未开始 | 端到端 |

## 设计 spec（根设计文档）

| 文件 | 语言 |
|------|------|
| [specs/2026-07-02-guaranteed-ssd-offload-priority-design-zh.md](specs/2026-07-02-guaranteed-ssd-offload-priority-design-zh.md) | 中文（主） |
| [specs/2026-07-02-guaranteed-ssd-offload-priority-design.md](specs/2026-07-02-guaranteed-ssd-offload-priority-design.md) | 英文 |

覆盖：分层职责、4 Phase 路线图、`ReplicateConfig.guaranteed_until_ms`、`ObjectMetadata.guaranteed_until_`、
PutEnd offload、`PushOffloadingQueue` 优先级、client 侧 bucket pin、SSD 生命周期管理、§14 维护性策略。

## 实施计划

### Phase 1 — 确保写入 SSD（✅ 已实现）
master 侧：独立 guaranteed offload 队列（无 limit）+ PutEnd 总 offload + NACK 重试，`enable_guaranteed_cache` 门控。

| 文件 | 语言 | 状态 |
|------|------|------|
| [plans/2026-07-03-guaranteed-ssd-offload-priority-phase1-zh.md](plans/2026-07-03-guaranteed-ssd-offload-priority-phase1-zh.md) | 中文 | ✅ 含 const-OffloadingTask 修复 + 用例 5&9 |
| [plans/2026-07-03-guaranteed-ssd-offload-priority-phase1.md](plans/2026-07-03-guaranteed-ssd-offload-priority-phase1.md) | 英文 | ✅ 已回填同等修复 |

### Phase 2 — SSD 副本驱逐保护（✅ 已实现）
client 侧：`BucketMetadata.guaranteed`（`YLT_REFL` 持久化）+ `OffloadObjects` 分同质 bucket +
`SelectEvictionCandidate` 跳过 guaranteed bucket（FIFO 前向扫描 / LRU 前向扫描不 erase）。

| 文件 | 语言 | 状态 |
|------|------|------|
| [plans/2026-07-03-guaranteed-ssd-offload-priority-phase2-zh.md](plans/2026-07-03-guaranteed-ssd-offload-priority-phase2-zh.md) | 中文 | ✅ 含 LRU 死循环修复 + 3 编译 bug 修复 |
| [plans/2026-07-03-guaranteed-ssd-offload-priority-phase2.md](plans/2026-07-03-guaranteed-ssd-offload-priority-phase2.md) | 英文 | ✅ 已回填同等修复 |

### Phase 3 — SSD 副本 TTL 管理（进行中）
方案：**master-driven downgrade**（TTL 只在 master，到期下发一次性降级列表，worker 翻转 bucket bool 后由现成 LRU 驱逐路径回收）。

| 文件 | 语言 | 状态 |
|------|------|------|
| [plans/2026-07-06-guaranteed-ssd-master-driven-downgrade-zh.md](plans/2026-07-06-guaranteed-ssd-master-driven-downgrade-zh.md) | 中文 | ✅ Task 1 完成；Task 2-5 阻塞 #2676；Task 6 待做 |
| [plans/2026-07-06-guaranteed-ssd-offload-priority-phase3-design-zh.md](plans/2026-07-06-guaranteed-ssd-offload-priority-phase3-design-zh.md) | 中文 | ⚠️ **已废弃**（client-side TTL 方案，被 master-driven-downgrade 推翻，仅留作对比） |

### Phase 4 — SGLang HiCache 集成（未开始）
HiCache Controller 写回路径判断 cache_control token 范围 → write_through + `guaranteed_until_ms`；
读取 L3 路径续期；Router 解析 `cache_control`。端到端。计划待 Phase 3 完成后写。

## 关键设计决策（沉淀）

1. **guaranteed 生命周期由 SSD 管**（非内存）：master 写入 SSD（Phase 1）→ client 保护 SSD 副本（Phase 2）→ TTL 回收（Phase 3）。
2. **Phase 3 = master-driven downgrade**（非 client-side TTL）：TTL 只在 master（object 级，类似 lease，不进 HA），到期下发降级列表，worker 翻转 bool 后现成 LRU 回收。消除跨节点 TTL 同步与 `SelectEvictionCandidate` 热路径改动。
3. **`IsHardPinned()` 不改**：闸的是内存驱逐，不是 offload；SSD 写入保证来自 `PushOffloadingQueue` 优先级，与 `IsHardPinned` 正交。
4. **`enable_guaranteed_cache` flag**（默认 false）门控全部行为——合并 main 时特性休眠，零行为变化。
5. **避开 `Serializer<Replica>` 格式 bump**（最高合并风险点）：TTL 在 `ObjectMetadata`（非 Replica，非 HA），client 侧用 `StorageObjectMetadata`/`BucketMetadata`（本地/YLT_REFL，非 HA Replica 序列化）。
6. **#2676 阻塞 Phase 3 Task 2-5**：`PollDowngradeKeys` 泛化 `PollRemoveAll` 通道模式，#2676 合入后做 follow-up。

## 已验证的代码事实（写计划时验证，避免凭推理踩坑）

- `ObjectMetadata.guaranteed_`（Phase 1 const bool）→ `guaranteed_until_`（time_point，非 const，不进 HA `SerializeMetadata`）
- `offloading_tasks` map mapped type 是 `const OffloadingTask`（不能原地赋值，必须 erase+emplace）
- `storage_backend_` 是基类 `StorageBackendInterface`，基类 `BatchOffload` + 全部 4 个子类 override 都要加参
- 无 master→client push RPC（48 handler 全在 master 侧），跨节点只能 heartbeat 挎带
- `prefix_hash` 是 HA OpLog 的全 key XXH32（非 blake3、非 key 前缀），`BatchExpireGuaranteed` 按 exact key 线性扫描
- LRU `SelectEvictionCandidate` 的 `while` 每次重置 `top_it=begin()`，朴素 `++top_it; continue` 死循环，需前向扫描

## 测试

| 测试 target | Phase | 内容 |
|------------|------|------|
| `guaranteed_offload_test` | 1 | 12 测试（含补充用例 5&9） |
| `guaranteed_eviction_test` | 2 | 4 测试（字段 + FIFO/LRU 跳过） |
| `guaranteed_downgrade_test` | 3 Task 1 | 4 单测（`guaranteed_until_` 设置/降级/flag off/missing key） |

## memory

`~/.claude/projects/-home-ruanzhao-WorkSpace-Mooncake-dev/memory/guaranteed_ssd_lifecycle_design.md` —
特性整体方向、关键约束、分阶段、Phase 3 master-driven 方向修订、维护性硬约束、已纠正的错误。

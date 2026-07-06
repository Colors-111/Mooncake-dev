# 显式上下文缓存设计 (cache_control) — SGLang + Mooncake

> **状态说明（2026-07-06）**：本文档是显式上下文缓存的**早期总设计**（2026-07-02）。§1 目标、§2 与阿里百炼定位、
> §3 分层职责（L1/L2/L3）、§4 cache_control 解析、§5 请求处理流程**仍有效**（架构背景）。但 §7（Mooncake Store 修改）
> 与 §11（实施分阶段）描述的是早期父设计，**已过时**——实际实施演进为 4-Phase 架构：
> IsHardPinned 不改、Phase 3 采用 master-driven downgrade（非 client-side TTL）、无容量限制、Phase 划分不同。
> Mooncake Store 侧的权威设计见 [guaranteed-ssd-architecture-zh.md](guaranteed-ssd-architecture-zh.md)（总领架构）
> + [superpowers/README.md](superpowers/README.md)（实施文档索引）。阅读本文档时，§7/§11 以那两份为准。

## 1. 目标

为 SGLang + Mooncake 集群实现显式上下文缓存：用户在 `messages` 中的 content 块添加 `"cache_control": {"type": "ephemeral"}` 标记，系统将标记位置对应的前缀 KV Cache 在 Mooncake Store 中标记为 guaranteed（TTL 内不可驱逐），过期后自动降级为普通对象。

**核心语义**：`cache_control` 标记 = "这段前缀很重要，别被驱逐"。前缀匹配由 SGLang radix tree 自动完成，不做任何窗口限制。

**驱逐保护机制**：采用 `guaranteed_until` 时间戳方案——对象在 PutStart 时携带绝对过期时间，BatchEvict 自然跳过未过期对象，过期后自动降级为普通对象，无需主动释放。零新增 RPC、一步写入、被动过期、崩溃后自然安全。

**设计原则**：guaranteed 保护在 L3（Master 侧），不在 SGLang 侧。SGLang 不追踪也不管理 guaranteed 缓存的生命周期，只在 I/O 边界上传递正确的参数。radix tree 节点无需新增任何字段。

## 2. 设计定位：与阿里百炼方案的区别

| 维度 | 阿里百炼（多租户云服务） | 本方案（单一集群自部署） |
|------|--------------------------|--------------------------|
| 匹配方式 | content 块级索引 + 20 块回溯窗口 | radix tree token 级全前缀匹配，无窗口限制 |
| 索引体系 | 独立 CacheIndex | 无独立索引，复用 radix tree 已有的 prefix hash chain |
| cache_control 语义 | 缓存断点（创建索引条目） | L3 guaranteed 标记（防止驱逐 + TTL 管理） |
| 生命周期管理 | SGLang 侧管理 | Master 侧管理，SGLang 不追踪 |
| 多租户隔离 | account_id 隔离 | 不需要（单一集群） |
| 计费 | 创建 125% / 命中 10% | 无计费需求 |
| 崩溃恢复 | 依赖 TTL 补丁或手动清理 | guaranteed_until 自然过期，无孤儿问题 |

## 3. 整体架构

### 3.1 分层职责

| 层级 | 位置 | 显式缓存中的角色 | 保证级别 |
|------|------|-----------------|---------|
| L1 (GPU) | SGLang Worker 本地 | 推理时临时持有，正常驱逐 | 不参与保证 |
| L2 (CPU) | SGLang Worker 本地 | 同引擎命中时加速（bonus） | 不参与保证 |
| L3 (Mooncake Store) | 分布式存储 | **确定性保证层** | 硬保证 |

L1/L2 不参与保证：请求可能路由到任何引擎，L2 只在本引擎可用。把保证绑定到 L2 就必须强制路由到特定引擎，打破负载均衡。L3 是所有引擎共享的，天然支持跨引擎。

### 3.2 架构图

```
  Client (带 cache_control 的 messages)
       │
       ▼
  ┌──────────────────────────────────────┐
  │  SGLang Router / API Server           │
  │  - 解析 cache_control → token 断点    │
  └──────────────┬───────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────┐
  │  SGLang Worker Instance               │
  │  ┌──────────────────────────────┐    │
  │  │ HiCache Controller (无改动)   │    │
  │  │ - radix tree 正常前缀匹配    │    │
  │  │ - 写入时判断 cache_control   │    │
  │  │   → write_through + 参数     │    │
  │  │ - 读取时判断 cache_control   │    │
  │  │   → GetReplicaList + 参数    │    │
  │  └──────────────┬───────────────┘    │
  │                 │                     │
  │  ┌──────────────▼───────────────┐    │
  │  │ HiRadixTree (L1+L2)         │    │
  │  │ - 本地 token→block 映射      │    │
  │  │ - 自动全前缀匹配             │    │
  │  │ - 无新增字段                 │    │
  │  └──────────────────────────────┘    │
  └──────────┬───────────────────────────┘
             │
  ┌──────────▼───────────────────────────┐
  │  Mooncake Store (L3)                │
  │  - guaranteed_until_ 时间戳保护     │
  │  - BatchEvict 跳过未过期对象        │
  │  - GrantLease 自动续期             │
  │  - RadixTree Index → prefix→keys   │
  │  - BatchExpireGuaranteed RPC       │
  │  - Lease + Eviction (existing)      │
  └─────────────────────────────────────┘
```

关键设计决策：SGLang 侧零新增状态，radix tree 节点无新增字段。所有 guaranteed 生命周期管理在 Master 侧。

## 4. cache_control 解析

### 4.1 从 messages 提取断点

```python
def extract_cache_breakpoints(messages: list[dict]) -> list[int]:
    """
    从 messages 中提取所有 cache_control 标记位置。
    返回字符空间中的偏移列表。
    """
    breakpoints = []
    char_offset = 0

    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            char_offset += len(content)
            continue
        if isinstance(content, list):
            for block in content:
                block_text = block.get("text", "") if isinstance(block, dict) else ""
                char_offset += len(block_text)
                if isinstance(block, dict) and "cache_control" in block:
                    cc = block["cache_control"]
                    if isinstance(cc, dict) and cc.get("type") == "ephemeral":
                        breakpoints.append(char_offset)
            continue
        char_offset += len(str(content))

    return breakpoints
```

### 4.2 字符断点 → token 断点

```python
def resolve_cache_control_offsets(messages: list[dict], tokenizer) -> list[int]:
    """
    从 messages 中的 cache_control 标记，解析出 token 空间中的断点位置。
    返回按大小排序的 token 偏移列表。
    仅在请求级使用，不存储到 radix tree 节点。
    """
    char_breakpoints = extract_cache_breakpoints(messages)

    if len(char_breakpoints) > 4:
        char_breakpoints = char_breakpoints[-4:]

    # 将字符偏移转换为 token 偏移
    token_offsets = map_char_positions_to_token_offsets(
        messages, char_breakpoints, tokenizer
    )

    return sorted(token_offsets)
```

## 5. 请求处理流程

### 5.1 主流程

```python
async def handle_chat_completion(request: ChatCompletionRequest):
    """
    处理带 cache_control 的 chat completion 请求。
    SGLang 侧零新增状态，仅在 I/O 边界传递参数。
    """

    # ========== Phase 0: 解析 cache_control 标记 ==========
    cc_token_offsets = resolve_cache_control_offsets(request.messages, tokenizer)

    if not cc_token_offsets:
        return await handle_standard_request(request)

    # ========== Phase 1: radix tree 正常前缀匹配 ==========
    match_result = sglang_worker.match_prefix(request.tokens)

    # ========== Phase 2: 执行推理 ==========
    # 将 cache_control 断点传给 Worker，控制写回策略
    response = await sglang_worker.chat_completions(
        model=request.model,
        messages=request.messages,
        _cc_token_offsets=cc_token_offsets,  # 请求级元数据
    )

    # ========== Phase 3: 组装响应 ==========
    cached_tokens = match_result.l2_hit + match_result.l3_hit
    response["usage"]["prompt_tokens_details"]["cached_tokens"] = cached_tokens
    return response
```

### 5.2 I/O 边界：写入路径

SGLang HiCache 在写回 L3 时，判断当前 block 是否在 `cache_control` 标记的 token 范围内：

```python
def determine_write_policy(self, token_idx, cc_token_offsets=None):
    """
    判断 block 写回策略。
    cache_control 范围内的 block → write_through + guaranteed_until_ms
    其他 block → 默认 write-back policy
    """
    if cc_token_offsets:
        for offset in cc_token_offsets:
            if token_idx < offset:
                return WritePolicy.WRITE_THROUGH

    return self.default_write_policy

def write_backup_storage(self, node, token_idx_start, token_idx_end, cc_token_offsets=None):
    """写回 L3，对 cache_control 范围内的 block 带 guaranteed 参数"""
    for token_idx in range(token_idx_start, token_idx_end):
        if self.determine_write_policy(token_idx, cc_token_offsets) == WritePolicy.WRITE_THROUGH:
            config = ReplicateConfig()
            config.with_soft_pin = True
            # 至少保证 1 分钟，避免请求处理时间长导致写入时 guaranteed 期已近过期
            config.guaranteed_until_ms = max(ttl_remaining_ms, 60000)

            key = self._make_key(node, token_idx)
            self.store.put(key, node.get_slice(token_idx), config)
```

### 5.3 I/O 边界：读取路径

Worker 从 L3 读取 KV cache 时，如果当前请求带有 `cache_control`，在 GetReplicaList 中传 `guaranteed_ttl_ms` 以续期：

```python
def fetch_from_l3(self, keys, cc_token_offsets=None):
    """从 L3 预取 KV cache blocks"""
    guaranteed_ttl_ms = 300000 if cc_token_offsets else 0  # 5min

    for key in keys:
        # GetReplicaList 时传 guaranteed_ttl_ms，Master 侧 GrantLease 自动续期
        replica_list = self.store.get_replica_list(key, guaranteed_ttl_ms=guaranteed_ttl_ms)
        # RDMA 预取...
```

### 5.4 关键路径说明

**初始写入（未命中场景）**：

1. 请求带了 `cache_control` → 解析出 token 断点
2. Worker 推理，新创建 KV cache blocks
3. HiCache 对 cache_control token 范围内的 blocks：write_through 写 L3 + 带 `guaranteed_until_ms`
4. Master 侧 `AllocateAndInsertMetadata` 转换为绝对时间戳存入 `guaranteed_until_`

**命中续期（已命中场景）**：

1. Worker 匹配到前缀（L1/L2/L3 均可），请求仍带 `cache_control`
2. Worker 读取 L3 数据时调 `GetReplicaList`，传入 `guaranteed_ttl_ms=300000`
3. Master 侧 `GrantLease` 自动续期 `guaranteed_until_ = max(当前值, now + 300s)`

**不带 cache_control 的请求读取 guaranteed 块**：

1. 请求没有 `cache_control`，但 L3 中有 guaranteed 块
2. Worker 正常读取（GetReplicaList 不带 guaranteed_ttl_ms）
3. Master 不续期 guaranteed_until_，guaranteed 期自然倒计时
4. 效果：这些块仍然可读（guaranteed 期未过不会被驱逐），但不会续期

**过期降级**：

1. `guaranteed_until_` 自然到期 → `IsHardPinned()` 返回 false
2. BatchEvict 正常将该对象纳入候选

## 6. 数据流详图

**首次请求（建立缓存）**：

```
客户端 → Router → SGLang Engine (messages 带 cache_control 标记)
                      │
                      ├─ tokenize messages → 解析 cache_control → cc_token_offsets
                      ├─ RadixAttention.match → matched_len (可能为 0)
                      ├─ Prefill（全部重计算）
                      ├─ Decode → 生成响应
                      ├─ HiCache.write_backup:
                      │   ├─ cc_token_offsets 范围内 → write_through 写 L3
                      │   │   config = {with_soft_pin=True, guaranteed_until_ms=max(remaining, 60000)}
                      │   │   → Master: PutStart(key, config) → guaranteed_until_ = now+5min
                      │   │   → Worker: RDMA 写入 → Master: PutEnd
                      │   └─ 其他 token → 按默认 write-back policy
                      └─ 返回响应 (cached_tokens=0)
```

**同引擎后续请求（L2 命中，请求仍带 cache_control）**：

```
客户端 → Router → 同一 SGLang Engine (messages 带 cache_control 标记)
                      │
                      ├─ RadixAttention.match → L2 直接命中
                      ├─ Prefill 只计算 suffix
                      ├─ Decode → 生成响应
                      ├─ HiCache.write_backup:
                      │   └─ 已写 L3 的 block → 跳过
                      ├─ L3 读取路径：GetReplicaList(guaranteed_ttl_ms=300000)
                      │   → GrantLease → guaranteed_until_ 续期到 now+5min
                      └─ 返回响应 (cached_tokens=matched_len)
```

**跨引擎后续请求（L3 预取命中，请求带 cache_control）**：

```
客户端 → Router → 另一台 SGLang Engine (messages 带 cache_control 标记)
                      │
                      ├─ RadixAttention.match → 本地未命中
                      ├─ batch_is_exist 查 L3 → 命中（guaranteed 保护）
                      ├─ RDMA 预取 KV cache 到 L2
                      │   → GetReplicaList(guaranteed_ttl_ms=300000)
                      │   → GrantLease → guaranteed_until_ 续期到 now+5min
                      ├─ Prefill 只计算 suffix
                      ├─ Decode → 生成响应
                      └─ 返回响应 (cached_tokens=l3_hit_len)
```

**5 分钟无带 cache_control 的请求（过期降级）**：

```
guaranteed_until_ 到期 → IsHardPinned() 返回 false
→ BatchEvict 正常驱逐 → 内存回收
→ 下次请求同前缀 → L3 未命中 → 需重计算 → 重新建立缓存
```

## 7. Mooncake Store 修改

### 7.1 ReplicateConfig 扩展

```cpp
// replica.h
struct ReplicateConfig {
    // 现有字段...
    int64_t guaranteed_until_ms{0};  // 0 = 无保证, >0 = 从 now 起的毫秒数
};
```

### 7.2 ObjectMetadata 扩展

```cpp
struct ObjectMetadata {
    // ... existing fields ...

    // NEW: 绝对过期时间戳，默认为 epoch（无保证）
    const time_point guaranteed_until_{};

    bool IsHardPinned() const {
        SpinLocker locker(&lock);
        if (hard_pinned) returntrue;
        // guaranteed_until 未过期也视为 pinned
        if (guaranteed_until_ > time_point{} &&
            guaranteed_until_ > SteadyClock::now()) {
            return true;
        }
        return false;
    }
};
```

使用 `SteadyClock`（单调时钟）而非 system_clock，避免系统时间回拨导致 guaranteed 期异常延长。

### 7.3 AllocateAndInsertMetadata 修改

```cpp
// 转换为绝对时间
time_point guaranteed_until = time_point{};
if (config.guaranteed_until_ms > 0) {
    guaranteed_until = SteadyClock::now() +
        std::chrono::milliseconds(config.guaranteed_until_ms);
}
ObjectMetadata meta(client_id, size, ..., guaranteed_until);
```

### 7.4 BatchEvict 修改——跳过 guaranteed 对象

```cpp
for (auto& [key, meta] : shard) {
    if (meta.IsHardPinned()) continue;
    if (!meta.IsLeaseExpired(now)) continue;

    // 跳过 guaranteed 期内的对象
    if (meta.guaranteed_until_ > now) continue;

    if (!meta.IsSoftPinned(now)) {
        no_pin_candidates.push_back({key, meta.lease_timeout});
    } else if (allow_evict_soft_pinned_objects_) {
        soft_pin_candidates.push_back({key, meta.lease_timeout});
    }
}
```

### 7.5 GrantLease 修改——仅续期，不创建

```cpp
void GrantLease(const uint64_t ttl, const uint64_t soft_ttl,
               const uint64_t guaranteed_ttl) const {
    SpinLocker locker(&lock);
    time_point now = SteadyClock::now();

    // 现有逻辑
    lease_timeout = std::max(lease_timeout,
        now + std::chrono::milliseconds(ttl));
    if (soft_pin_timeout) {
        *soft_pin_timeout = std::max(*soft_pin_timeout,
            now + std::chrono::milliseconds(soft_ttl));
    }

    // 新增：仅续期已 guaranteed 的对象，不创建新 guaranteed 期
    // 这样不带 cache_control 的请求读取 guaranteed 块不会意外续期
    if (guaranteed_ttl > 0 && guaranteed_until_ > now) {
        guaranteed_until_ = std::max(guaranteed_until_,
            now + std::chrono::milliseconds(guaranteed_ttl));
    }
}
```

关键区别：`guaranteed_until_ > now` 检查确保只有当前仍 guaranteed 的对象才会续期。如果 guaranteed 期已过（guaranteed_until_ <= now），即使传入 guaranteed_ttl > 0 也不会重新创建 guaranteed 期。

### 7.6 GetReplicaListRequest 扩展

```cpp
struct GetReplicaListRequest {
    std::string key;
    uint64_t guaranteed_ttl_ms{0};  // >0 表示续期 guaranteed_until
};
```

### 7.7 Guaranteed 容量限制

Master 维护 `guaranteed_memory_used_`（所有 guaranteed 对象占用的总容量），在 PutStart 时检查：

```cpp
if (config.guaranteed_until_ms > 0) {
    size_t guaranteed_total = guaranteed_memory_used_.load(std::memory_order_relaxed);
    if (guaranteed_total + value_length > guaranteed_memory_limit_) {
        return tl::make_unexpected(ErrorCode::GUARANTEED_CAPACITY_EXCEEDED);
    }
    guaranteed_memory_used_.fetch_add(value_length, std::memory_order_relaxed);
}
```

Worker 侧收到 `GUARANTEED_CAPACITY_EXCEEDED` 后降级为普通 PutStart（不带 guaranteed_until_ms），不影响写入。

驱逐 guaranteed 对象时递减：

```cpp
if (meta.guaranteed_until_ > time_point{}) {
    guaranteed_memory_used_.fetch_sub(meta.size, std::memory_order_relaxed);
}
```

### 7.8 SSD Offload 优先级

**前置问题**：当前 SSD offload 机制中，每个 client 的 offload 队列上限为 `offloading_queue_limit_`（默认 50000）。超限时 `PushOffloadingQueue` 返回 `KEYS_ULTRA_LIMIT`，在 `offload_force_evict_` 为 true 的情况下，MEMORY replica 会被直接驱逐而不写 SSD——数据丢失，guaranteed 语义失效。

**解决方案**：给 offload 队列加入优先级，guaranteed 对象必须成功写入 SSD，不能因为队列满而被丢弃。

**优先级定义**：

| 优先级 | 对象类型 | offload 失败时行为 |
|--------|---------|-------------------|
| HIGH | guaranteed（`guaranteed_until_ > now`） | 必须成功，队列满时驱逐 LOW 对象腾位 |
| NORMAL | soft-pinned | best-effort，队列满时可丢弃 |
| LOW | 普通对象（无 pin） | best-effort，队列满时可丢弃 |

**实现要点**：

1. **`PushOffloadingQueue` 改造**：入队时检查对象优先级。若为 HIGH 且队列已满，从队列中驱逐 LOW 优先级的待 offload 对象腾出位置；若无 LOW 可驱逐，则扩容队列（打破 50000 上限）。NORMAL/LOW 对象入队时队列已满则返回 `KEYS_ULTRA_LIMIT`，行为与现在一致。

2. **`try_evict_or_offload` 改造**：`BatchEvict` 中选择 offload 候选时，guaranteed 对象优先入队（抢占 LOW 对象的 offload 位），确保在内存压力下 guaranteed 对象先写 SSD 再释放内存。

3. **ObjectMetadata 新增优先级方法**：

```cpp
enum class OffloadPriority { LOW, NORMAL, HIGH };

OffloadPriority GetOffloadPriority() const {
    SpinLocker locker(&lock);
    auto now = SteadyClock::now();
    if (guaranteed_until_ > now) return OffloadPriority::HIGH;
    if (soft_pin_timeout && *soft_pin_timeout > now) return OffloadPriority::NORMAL;
    return OffloadPriority::LOW;
}
```

**关键保证**：guaranteed 对象的 offload 是"必须成功"语义——不能因为批次数组满了就丢弃，只有 SSD 本身写入失败或 SSD 空间不足才能拒绝。

### 7.9 主动失效：BatchExpireGuaranteed RPC

运营场景（更新 system prompt、RAG doc 错误、调试）需要立即失效，不能等 5 分钟自然过期。

```cpp
struct BatchExpireGuaranteedRequest {
    std::string prefix_hash;  // blake3 hex
};

struct BatchExpireGuaranteedResponse {
    uint32_t expired_count;
};
```

Master 遍历所有 shard，对 key 匹配 `starts_with(prefix_hash)` 的对象批量置 `guaranteed_until_` 为 epoch：

```cpp
auto MasterService::BatchExpireGuaranteed(const std::string& prefix_hash)
    -> tl::expected<uint32_t, ErrorCode> {
    uint32_t expired = 0;
    for (size_t i = 0; i < kNumShards; ++i) {
        MetadataShardAccessorRW accessor(this, i);
        for (auto& [key, meta] : accessor->metadata) {
            if (!key.starts_with(prefix_hash)) continue;
            SpinLocker locker(&meta.lock);
            if (meta.guaranteed_until_ > time_point{}) {
                guaranteed_memory_used_.fetch_sub(meta.size, std::memory_order_relaxed);
                meta.guaranteed_until_ = time_point{};
                expired++;
            }
        }
    }
    return expired;
}
```

注意：当前实现为 O(N) 全量扫描，若后续 Master 引入 prefix 索引可优化为 O(K)。

管理端点：

```
POST /v1/cache/evict
  Body: {"prefix_hash": "<blake3_hex>"}
  → {"expired_count": 3, "status": "ok"}
```

Router 收到请求后直接调 Master 的 `BatchExpireGuaranteed` RPC，不经过 Worker。

### 7.10 崩溃恢复

**Worker 崩溃**：SGLang 侧无 guaranteed 状态，无需清理。Master 侧 `guaranteed_until_` 自然到期后可驱逐。

**Master 崩溃重启**：`guaranteed_until_` 不持久化到 etcd（和 `hard_pinned` 一样是运行时状态），重启后所有对象无 guaranteed 期，可被驱逐。下次请求的 write_through 会重新带 `guaranteed_until_ms`。

**Worker + Master 同时崩溃重启**：干净状态，安全。

### 7.11 向后兼容性

- `guaranteed_until_ms` 默认值 0 → 所有现有 PutStart 调用行为不变
- `guaranteed_until_` 默认值为 epoch → `IsHardPinned(now)` 返回 false → BatchEvict 行为不变
- `guaranteed_ttl` 默认值 0 → GrantLease 不修改 guaranteed_until_ → 现有 GetReplicaList 行为不变

**零回归风险。**

## 8. 统计与观测

SGLang 侧无 guaranteed 状态，不提供 guaranteed 相关统计。

Master 侧可提供 guaranteed 对象的总量统计：

```cpp
struct GuaranteedStats {
    size_t guaranteed_object_count;
    size_t guaranteed_memory_bytes;
    size_t guaranteed_memory_limit_bytes;
    double guaranteed_utilization_ratio;  // used / limit
};
```

通过 Master 现有的 metrics 端点暴露。

## 9. 并发控制

### 9.1 竞态: 同一前缀的并发写入

两个相同内容的请求同时到达，可能同时尝试 write_through 同一个 block 到 L3。

**解决方案**: PutStart 是幂等的。两个请求都带 `guaranteed_until_ms` 写同一个 key，先到者设置 `guaranteed_until_`，后到者覆盖为相同值。不会出错。

### 9.2 竞态: guaranteed_until_ 过期与续期同时发生

**解决方案**: GrantLease 在 SpinLock 下操作，且先检查 `guaranteed_until_ > now`。如果刚好过期，续期条件不满足，不会续期。后果：该 block 的 guaranteed 保护自然结束，下次带 `cache_control` 的请求会重新 write_through 建立 guaranteed。

### 9.3 不存在"SGLang 与 Master 状态不一致"问题

SGLang 不持有 guaranteed 状态，Master 是唯一的 guaranteed 生命周期管理者。不存在双端同步问题。

## 10. 需要修改的关键文件

| 文件 | 修改内容 |
|------|----------|
| `mooncake-store/include/replica.h` | ReplicateConfig 增加 `guaranteed_until_ms` |
| `mooncake-store/include/master_service.h` | ObjectMetadata 增加 `guaranteed_until_`, `IsHardPinned()` 增加时间戳检查, `GrantLease` 增加 `guaranteed_ttl` 参数（仅续期不创建）, 新增 `guaranteed_memory_used_` 原子变量和 `guaranteed_memory_limit_` 配置, 新增 `BatchExpireGuaranteed` 方法, 新增 `GetOffloadPriority()` 方法 |
| `mooncake-store/src/master_service.cpp` | AllocateAndInsertMetadata 处理 guaranteed_until 转换及容量检查, BatchEvict 增加 guaranteed 跳过判断及 guaranteed_memory_used_ 递减, GrantLease 修改（仅续期）, BatchExpireGuaranteed 实现, PushOffloadingQueue 优先级队列改造, try_evict_or_offload guaranteed 优先入队 |
| `mooncake-store/include/rpc_types.h` | GetReplicaListRequest 增加 `guaranteed_ttl_ms` 字段, 新增 BatchExpireGuaranteedRequest/Response |
| SGLang HiCache Controller | 写回策略感知 cache_control → write_through + guaranteed_until_ms; 读取 L3 时根据请求 cache_control 传 guaranteed_ttl_ms |
| SGLang Router / API Server | cache_control 解析 → token 断点; `/v1/cache/evict` 端点 → 调 Master BatchExpireGuaranteed |

**注意**：SGLang RadixTreeNode **无需修改**。

## 11. 实施分阶段

### Phase 1: Mooncake Store 基础设施
- SSD Offload 优先级改造（前置依赖）：`PushOffloadingQueue` 优先级队列、`try_evict_or_offload` guaranteed 优先入队、`ObjectMetadata::GetOffloadPriority()`
- ReplicateConfig 增加 `guaranteed_until_ms`
- ObjectMetadata 增加 `guaranteed_until_`，`IsHardPinned()` 增加时间戳检查
- AllocateAndInsertMetadata 处理 guaranteed_until 转换
- BatchEvict 增加 guaranteed 跳过判断
- GetReplicaListRequest 增加 `guaranteed_ttl_ms`
- GrantLease 增加 `guaranteed_ttl` 参数（仅续期，不创建）
- 新增 `guaranteed_memory_used_` / `guaranteed_memory_limit_`（按 L3 内存+SSD 总容量设），PutStart 容量检查
- 新增 `BatchExpireGuaranteed` RPC
- 单元测试：驱逐行为、时间过期、容量超限、续期逻辑、BatchExpireGuaranteed、SSD offload 优先级行为

### Phase 2: HiCache 集成
- HiCache Controller 写回路径判断 cache_control token 范围 → write_through + guaranteed_until_ms
- HiCache Controller 读取 L3 路径：请求带 cache_control 时传 guaranteed_ttl_ms
- GUARANTEED_CAPACITY_EXCEEDED 降级为普通 PutStart
- Router 侧 cache_control 解析 → cc_token_offsets
- 端到端集成测试

### Phase 3: 管理端点 (可选)
- `/v1/cache/evict` → 调 Master BatchExpireGuaranteed
- `/v1/cache/status` → 查 Master guaranteed stats
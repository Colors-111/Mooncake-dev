# Guaranteed Object SSD-Write Guarantee — Design Spec

**Date:** 2026-07-02
**Parent design:** `docs/explicit_context_cache_design.md` (Phase 1, item 1 — narrowed)
**Branch:** `remove_disk_data_in_ssd_offload`

## 1. Goal

Guarantee that a guaranteed object is **successfully enqueued into the offload queue** at `PutEnd` (async SSD
write), **retried on SSD write failure**, and treated as a normal object for memory eviction once the SSD copy
completes. **Core guarantee:**

> **A guaranteed object is eventually written to SSD** — never rejected for a full offload queue (independent
> guaranteed queue, no limit), never abandoned on SSD write failure (retry). After the SSD copy completes, memory is released normally
> and the data persists on SSD.

**Design intent**: the guaranteed object's **lifecycle is managed by SSD** — which is exactly why "ensure write to
SSD" is fundamental: SSD is the lifecycle carrier; without writing, there is no object to protect. Memory only
passes the guaranteed marker to the offload queue (HIGH enqueue) and treats the object as normal once written
(weak semantics, no long-lived memory pin). TTL/renewal/eviction-protection/active-invalidation lifecycle logic
will act on the **SSD replica** in a follow-up slice (§13) — this slice is its **write prerequisite**: ensure the
data reaches SSD first, then SSD-side lifecycle management becomes meaningful.

Current pain point: `PushOffloadingQueue` returns `KEYS_ULTRA_LIMIT` when
`offloading_objects.size() >= offloading_queue_limit_` (default 50000); with `offload_force_evict_=true` the
MEMORY replica is deleted without an SSD copy — data loss. This spec closes the "queue-full rejects enqueue" and
"write-failure drops guarantee" holes.

## 2. Scope (substantially narrowed)

The core of Phase 1 item 1 + minimal marker + failure retry. **Eviction is untouched.**

| Item | Status |
|------|--------|
| `PushOffloadingQueue` independent guaranteed queue (no limit/no preemption, core) | ✅ in scope |
| `OffloadTaskItem.guaranteed` flag (master→client propagation, carried on drain) | ✅ in scope |
| `ReplicateConfig.guaranteed_until_ms` marker | ✅ in scope |
| `ObjectMetadata.guaranteed_` boolean marker | ✅ in scope |
| client-side `BuildBucket` splitting by `guaranteed` into homogeneous buckets | ⏭️ Phase 2 (see §6.4) |
| `AllocateAndInsertMetadata` records marker | ✅ in scope |
| `PutEnd` guaranteed always offloads + HIGH priority | ✅ in scope |
| `NotifyOffloadSuccess` guaranteed failure retry | ✅ in scope |
| `IsHardPinned` modification | ❌ no (memory protected by refcount pin, not hard-pin) |
| `BatchEvict` deletion exclusion / force-evict bypass | ❌ no (eviction treats it as a normal object) |
| `GrantLease` renewal + `GetReplicaListWithGuaranteed` RPC | ❌ no (lifetime not maintained via renewal) |
| `guaranteed_memory_used_` / capacity limits | ❌ no (not long-lived in memory) |
| `guaranteed_until_` timestamp / TTL semantics | ❌ not this slice (Phase 2) |
| `BatchExpireGuaranteed` RPC | ❌ deferred |

## 3. Guarantee semantics & known boundaries

Verified technical facts:

**① ✓ refcount pin protects the memory replica until SSD write completes** — after `PushOffloadingQueue`
succeeds, `inc_refcnt()` is called ([3075](../../../mooncake-store/src/master_service.cpp),
[6824](../../../mooncake-store/src/master_service.cpp), [7069](../../../mooncake-store/src/master_service.cpp));
all memory eviction guards check `get_refcnt()==0` ([6761](../../../mooncake-store/src/master_service.cpp) et al.).
SSD write done → `NotifyOffloadSuccess` releases the pin ([4858](../../../mooncake-store/src/master_service.cpp))
→ memory replica becomes a normal evictable object. **This chain is already supported; no `IsHardPinned` needed.**

**② ⚠ SSD replica is evictable by fifo/lru (known boundary, accepted)** — LOCAL_DISK replicas have no pin
protection; SSD-full client-side fifo/lru evicts them (`PrepareEviction`,
[storage_backend.cpp:2227](../../../mooncake-store/src/storage_backend.cpp)). So guaranteed's "guarantee" is
**at-least-once write to SSD**, not SSD-resident-forever. Under SSD capacity pressure a guaranteed replica may
still be evicted from SSD — consistent with "eviction treats it as a normal object"; this slice accepts that.
SSD-replica protection (strong semantics) is future work.

**③ ✅ SSD write-failure retry (fixed this slice)** — currently `NotifyOffloadSuccess` on NACK (`data_size<0`,
[4813](../../../mooncake-store/src/master_service.cpp)) releases the pin + erases the task, no re-queue →
guarantee lost. This slice re-enqueues guaranteed objects on failure (`PushOffloadingQueue` + `inc_refcnt`).

**④ ⚠ offload-task 600s TTL expiry (known boundary, accepted)** — task expiry releases the pin
([5602](../../../mooncake-store/src/master_service.cpp)); under pathological SSD latency (>600s) the guaranteed
memory replica may become evictable early, and retry also fails (task already erased). This slice does not touch
the TTL (user choice). Documented as a known limitation.

**Degradation**: when `enable_offload_=false` (offload globally off) or client `enable_offloading=false`,
`PushOffloadingQueue` returns `UNABLE_OFFLOADING`, guaranteed cannot enqueue → degrades to a normal object (no
guarantee). Reasonable (no SSD, no guarantee).

## 4. Data model

### 4.1 `ReplicateConfig` ([mooncake-store/include/replica.h:81-144](../../../mooncake-store/include/replica.h))

Trailing field (parent-design form retained for Phase 2 TTL compatibility):

```cpp
int64_t guaranteed_until_ms{0};  // 0 = not guaranteed; >0 = guaranteed (this slice only uses >0)
```

Backward-compatible: aggregate, no `YLT_REFL`, struct_pack default config tolerates trailing fields. **This slice
does not use the time value — only `>0` as the "this object is guaranteed" boolean; TTL semantics (renewal/expiry
eviction) are Phase 2.**

### 4.2 `ObjectMetadata` ([mooncake-store/include/master_service.h:851-1154](../../../mooncake-store/include/master_service.h))

Boolean marker (no timestamp for now — YAGNI; upgrade to `time_point` when Phase 2 adds TTL):

```cpp
const bool guaranteed_{false};  // true = PutEnd guarantees offload to SSD
```

Constructor gains a trailing param `bool guaranteed = false` (default false → existing callers unchanged).
`IsHardPinned()` is NOT modified.

### 4.3 `OffloadTaskItem` ([mooncake-store/include/types.h:263-273](../../../mooncake-store/include/types.h))

Add a `guaranteed` flag (snapshot from object `guaranteed_` at enqueue; the client uses it in `BuildBucket` to split
into homogeneous buckets):

```cpp
struct OffloadTaskItem {
    std::string tenant_id;
    std::string key;
    int64_t size;
    bool guaranteed{false};  // snapshot from object guaranteed_
    // ... operator==, YLT_REFL updated to include guaranteed
};
```

Trailing bool field, backward-compatible. **No `OffloadPriority` enum** — the independent queue itself is the
classification; the enqueue side only needs `bool guaranteed` to pick which map, and the task-side `guaranteed` flag
is for client-side bucket splitting.

## 5. `PutEnd` path: guaranteed always offloads

Current ([master_service.cpp:3064-3084](../../../mooncake-store/src/master_service.cpp)): offloads all completed
memory replicas only when `offload_on_evict_=false` (default). With `offload_on_evict_=true`, PutEnd does not
offload (deferred to eviction).

Change: guaranteed objects **ignore `offload_on_evict_`** and always offload at PutEnd:

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

Non-guaranteed under `offload_on_evict_=true` still does not PutEnd-offload (unchanged).

## 6. `PushOffloadingQueue`: independent guaranteed queue

Guaranteed and normal split into two per-client maps; guaranteed has no limit and no preemption. Simpler than
"shared queue + HIGH preempts LOW", with existing precedent.

### 6.1 Data structure

Each client's `LocalDiskSegment` already has `offloading_objects` (normal) and `promotion_objects`
(promotion-on-hit, [master_service.cpp:5004](../../../mooncake-store/src/master_service.cpp) precedent). Add
`guaranteed_offloading_objects` (per-client `unordered_map<string, OffloadTaskItem>`), parallel to
`offloading_objects`, guarded by the same `offloading_mutex_`.

### 6.2 Enqueue logic

`PushOffloadingQueue` gains a trailing `bool guaranteed = false` param (default false, backward-compatible):

- `guaranteed=true` → into `guaranteed_offloading_objects`, **no limit check**, never returns `KEYS_ULTRA_LIMIT`.
- `guaranteed=false` → into `offloading_objects`, returns `KEYS_ULTRA_LIMIT` when
  `size() >= offloading_queue_limit_` (unchanged).

No preemption, no expansion. The guaranteed queue is unbounded but implicitly bounded: a task in it corresponds to a
memory replica still in memory (pinned awaiting offload), so queue length ≤ in-memory guaranteed objects awaiting
offload ≤ memory capacity.

### 6.3 drain

`OffloadObjectHeartbeat` ([master_service.cpp:4708-4717](../../../mooncake-store/src/master_service.cpp)) is extended:
on drain it merges `guaranteed_offloading_objects` and `offloading_objects` into the return (signature unchanged, still
`vector<OffloadTaskItem>`), with each task carrying the `guaranteed` flag. The disabled branch likewise cleans both
maps + refcounts (reusing the copy-then-release at [4719-4749](../../../mooncake-store/src/master_service.cpp)).

### 6.4 client-side bucket splitting (⏭️ Phase 2)

The client's `BuildBucket` splits by `OffloadTaskItem.guaranteed`: guaranteed tasks go into guaranteed buckets, normal
tasks into normal buckets. This pairs with Phase 2 bucket pin (guaranteed buckets un-evictable).

**Moved to Phase 2 (not Phase 1)**: under Phase 1 (no client-side bucket pin), `BuildBucket` splitting has no observable
effect (it doesn't change whether data reaches SSD, only the homogeneity of keys within buckets). And `BuildBucket` lives
in `storage_backend.cpp` (a different module), requiring separate exploration of `AllocateOffloadingBuckets`. Phase 1
already propagates the flag to the client via `OffloadTaskItem.guaranteed` (§4.3) + `OffloadObjectHeartbeat` drain
(§6.3); Phase 2 simply consumes it in `BuildBucket`.

## 7. `NotifyOffloadSuccess`: guaranteed failure retry

Current NACK branch (`data_size < 0`, [master_service.cpp:4813-4830](../../../mooncake-store/src/master_service.cpp)):
releases pin + erases task, no re-queue.

Change: guaranteed re-enqueues on failure to **wait for the next drain batch**, with the **pin untouched** (no
`inc_refcnt`, no `dec_refcnt`). The refcount set at PutEnd stays at 1 until SSD write succeeds. Re-enqueue only
refreshes the task's `start_time` to reset the offload-task TTL (preventing the 600s reaper from erasing the task
and making the memory replica evictable prematurely):

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
                // guaranteed: re-enqueue, wait for next drain. Pin untouched (no inc/dec).
                auto result = PushOffloadingQueue(request_object_id, *source,
                                                   /*guaranteed=*/true);
                if (result || result.error() == ErrorCode::OBJECT_ALREADY_EXISTS) {
                    // enqueued, or key still queued (NACK before drain) — refresh start_time, wait for next batch
                    task_it->second = OffloadingTask{task_it->second.source_id,
                                                     std::chrono::system_clock::now()};
                    continue;  // skip dec_refcnt/erase; pin retained
                }
                // re-enqueue failed (e.g. UNABLE_OFFLOADING) → degrade
            }
            // non-guaranteed or degraded: existing behavior, dec_refcnt + erase task
            if (source != nullptr) source->dec_refcnt();
            tenant_state.offloading_tasks.erase(task_it);
        }
    }
    continue;
}
```

**Why refcount is untouched**: PutEnd `inc_refcnt` once (refcnt=1) → drain to client → write fails → NACK →
re-enqueue → next drain → write again… refcnt stays 1 through the entire retry loop until SSD write succeeds and
`NotifyOffloadSuccess` takes the success branch (`dec_refcnt`, [4858](../../../mooncake-store/src/master_service.cpp)).
No inc, no dec — no double-inc risk. The pin protects the memory replica from eviction until SSD write completes.

**No retry-count cap**: guaranteed means "must be written". Under persistent SSD failure a guaranteed object stays
memory-pinned waiting for the next batch — the cost of the guarantee. Persistent failure is an operational concern
(SSD fault). Optional backoff/cap is future work.

**drain interaction**: at NACK time the client has already drained the key from the master queue via
`OffloadObjectHeartbeat` (snapshot-and-clear), so re-`PushOffloadingQueue` won't conflict; if NACK arrives before
the drain (key still queued), it returns `OBJECT_ALREADY_EXISTS` — also "already queued, wait for next batch",
just retain the pin.

**TTL interaction**: refreshing `start_time` resets the 600s offload-task TTL, preventing the reaper from erasing
the task. If the TTL already expired (task erased by the reaper), `task_it == end` → the retry branch is not
entered, existing degradation runs, guaranteed degrades to a normal object. This is boundary ④; accepted.

## 8. Concurrency & lock order

- Lock order stays **Shard Lock → `offloading_mutex_`**. Failure retry calls `PushOffloadingQueue` (taking
  `offloading_mutex_`) while holding the shard lock — correct order.
- `PushOffloadingQueue`'s enqueue (pick map + emplace) completes atomically under `offloading_mutex_`; no
  copy-then-release needed (independent queue, no preemption, no preempted items to clean up).
- `NotifyOffloadSuccess` retry is already inside the shard lock (via accessor); re-entering
  `PushOffloadingQueue` is safe.
- `guaranteed_` is `const` (immutable after construction) — no locking needed.

## 9. Backward compatibility

- `ReplicateConfig.guaranteed_until_ms` defaults 0 → existing PutStart unchanged.
- `ObjectMetadata.guaranteed_` defaults false → `IsHardPinned()` unchanged, offload behavior unchanged.
- `OffloadTaskItem.guaranteed` defaults false + trailing-tolerant → old/new client/server interoperate.
- `PutEnd` condition `(!offload_on_evict_ || metadata.guaranteed_)`: with `guaranteed_=false` reduces to
  `!offload_on_evict_` (original).
- `PushOffloadingQueue` new `bool guaranteed=false` param: existing callers (eviction-path
  [6821](../../../mooncake-store/src/master_service.cpp), [7066](../../../mooncake-store/src/master_service.cpp)) omit
  it (defaults false), go to `offloading_objects`, behavior unchanged.
- `NotifyOffloadSuccess` NACK branch: `guaranteed_=false` takes the original path (release pin + erase task).
- **Zero regression.**

## 10. Testing plan

Mirror `tests/offload_on_evict_test.cpp` patterns (public API: `PutStart`/`PutEnd`/`OffloadObjectHeartbeat`).
New file `tests/guaranteed_offload_test.cpp`. Cases:

1. **guaranteed exempt from normal-queue limit** (default async path) — fill `offloading_objects` to
   `offloading_queue_limit` with normal objects, Put a guaranteed object (`guaranteed_until_ms>0`), assert guaranteed
   goes to `guaranteed_offloading_objects` at `PutEnd` (not `KEYS_ULTRA_LIMIT`); `OffloadObjectHeartbeat` return
   contains guaranteed (with the `guaranteed` flag).
2. **normal queue still rejects when full** — `offloading_objects` full + non-guaranteed Put →
   `KEYS_ULTRA_LIMIT` (unchanged), and the guaranteed queue is unaffected.
3. **two queues independent** — with normal queue full, guaranteed still enqueues; with many guaranteed tasks in
   the guaranteed queue, the normal queue's limit quota is not consumed.
4. **guaranteed retries on SSD write failure** — mock SSD write failure (NACK), assert guaranteed re-enqueued
   (task retained/refreshed, pin not released), memory replica not evicted.
5. **guaranteed becomes normally evictable after SSD success** — guaranteed SSD write succeeds, pin released,
   trigger memory eviction → memory replica deleted, LOCAL_DISK replica retained.
6. **offload_on_evict=true: guaranteed still PutEnd-offloads** — `offload_on_evict_=true`, Put guaranteed →
   enqueued at `PutEnd` (not deferred); contrast non-guaranteed which does not PutEnd-offload.
7. **non-guaranteed under offload_on_evict=true does not PutEnd-offload** — unchanged (backward compat).
8. **default zero behavior change** — with `enable_guaranteed_cache=false` (default) and `guaranteed_until_ms=0`,
   behaves identically to today (hard guarantee via the flag gate — lets you verify upstream behavior is unchanged
   right after a main merge).
9. **enable_offload=false: guaranteed degrades** — guaranteed not enqueued, treated as normal object (no
    guarantee, no error).

> Case 10 (client-side `BuildBucket` splitting into homogeneous buckets) moves to **Phase 2** with §6.4 and is out of
> the Phase 1 test scope.

## 11. Overall phased roadmap

**Design spine**: the guaranteed object's **lifecycle is managed by SSD**. This requires three steps: (1) first ensure
data is written to SSD; (2) protect the guaranteed SSD replica from eviction; (3) manage the SSD replica's TTL
(renewal + expiry downgrade + active invalidation). This spec covers all of it, implemented in phases — each
independently testable and deliverable.

```
Phase 1: Ensure write to SSD (this spec's deliverable slice)
  ├─ 1.A marker + data model
  ├─ 1.B PutEnd guarantee enqueue + independent guaranteed queue
  └─ 1.C failure retry
Phase 2: SSD replica eviction protection (guaranteed not deleted by fifo/lru after SSD write)
  ├─ 2.A marker propagated to client + per-key metadata
  └─ 2.B client-side SelectEvictionCandidate skips guaranteed
Phase 3: SSD replica TTL management (lifecycle on SSD)
  ├─ 3.A guaranteed_until on SSD replica + renewal
  ├─ 3.B expiry downgrade (SSD replica evictable after TTL)
  └─ 3.C BatchExpireGuaranteed active-invalidation RPC
Phase 4: SGLang HiCache integration (end-to-end)
```

### Phase 1: Ensure write to SSD (**detailed in this spec, implement now**)

The guarantee lands at PutEnd enqueue + failure retry; memory becomes normal once written (weak semantics). See
§4–§10 for detail.

| Sub-phase | Changes | Tests |
|------|------|------|
| **1.A Marker** | config/gflag `enable_guaranteed_cache` (default false); `ReplicateConfig.guaranteed_until_ms` (only `>0` checked); `ObjectMetadata.guaranteed_` bool; `AllocateAndInsertMetadata` records marker (all gated under `enable_guaranteed_cache_`) | case 8 |
| **1.B PutEnd + independent queue** | `PutEnd` guaranteed ignores `offload_on_evict_` and always offloads; add `guaranteed_offloading_objects` per-client map (`promotion_objects` precedent); `PushOffloadingQueue` gains `bool guaranteed`, guaranteed goes to the independent queue with no limit, never `KEYS_ULTRA_LIMIT`; `OffloadObjectHeartbeat` merges both maps on drain, `OffloadTaskItem.guaranteed` flag propagated to client | cases 1–3, 6, 7 |
| **1.C Failure retry** | `NotifyOffloadSuccess` NACK branch: guaranteed re-enqueues into the independent queue, pin retained, waits for next drain | cases 4, 5 |

### Phase 2: SSD replica eviction protection (follow-up slice)

**Problem**: after Phase 1 a guaranteed object is surely written to SSD, but the SSD replica is evictable by
client-side fifo/lru (boundary ②). Phase 2 closes this, so a guaranteed SSD replica is not evicted within its TTL.

**Key architectural constraint** (verified):
- SSD eviction is **client-side** in `BucketStorageBackend`: `PrepareEviction`→`SelectEvictionCandidate`→
  `FinalizeEviction` deletes files ([storage_backend.cpp:1314-1326](../../../mooncake-store/src/storage_backend.cpp)).
- master `BatchEvictDiskReplica` is a **post-hoc notification**; returning an error **does not stop the client from
  deleting files** (`FinalizeEviction` deletes unconditionally, the handler ignores the return value,
  [file_storage.cpp:481-489](../../../mooncake-store/src/file_storage.cpp)).
- So protection **must land client-side at `SelectEvictionCandidate`**
  ([storage_backend.cpp:2181](../../../mooncake-store/src/storage_backend.cpp)), not master-side rejection.

**Sub-phases:**

**2.A Marker propagated to client + per-key metadata**
- `OffloadTaskItem` gains `bool guaranteed` (trailing, backward-compatible), set by master at enqueue.
- client-side `StorageObjectMetadata` ([types.h:542-550](../../../mooncake-store/include/types.h)) gains
  `bool guaranteed` (or `guaranteed_until_ms`); `BuildBucket` threads it from `OffloadTaskItem` into
  `object_bucket_map_` ([storage_backend.h:964](../../../mooncake-store/include/storage_backend.h)).
- The client eviction path can then look up each key's guaranteed flag.

**2.B client-side SelectEvictionCandidate skips guaranteed (bucket-level pin, chosen)**
- **Difficulty: eviction is bucket-granular** (`SelectEvictionCandidate` returns a whole bucket,
  [storage_backend.cpp:2181](../../../mooncake-store/src/storage_backend.cpp)). A bucket holds multiple keys; if one
  is guaranteed, skipping the whole bucket wastes space, evicting it would wrongly delete the guaranteed one.
- **Chosen: bucket-level pin**. After `SelectEvictionCandidate` picks a candidate bucket, check whether any key in it
  is guaranteed (maintain a `guaranteed_key_count` on `BucketMetadata` — see §14 maintainability, avoids scanning).
  If so, skip to the next candidate. Simple; the cost is a guaranteed bucket is fully un-evictable within its TTL
  (space fragmentation), reclaimed via Phase 3 TTL expiry or 3.C active invalidation.
- Rejected bucket-splitting (changes `BuildBucket`/`AllocateOffloadingBuckets` allocation — higher maintenance risk).
- TTL arrives in Phase 3; Phase 2 is "guaranteed SSD replica permanently protected".

### Phase 3: SSD replica TTL management (follow-up slice — the core of "lifecycle on SSD")

**Problem**: Phase 2's guaranteed SSD replica is permanently protected; TTL is needed for automatic reclamation. TTL
acts on the SSD replica, but renewal is on the master and eviction decisions are on the client — cross-node state sync
is the hard part.

**Sub-phases:**

**3.A guaranteed_until on client-side metadata + renewal**
- **TTL storage location (maintainability-critical decision)**: `guaranteed_until` lives on the **client-side
  `StorageObjectMetadata`** (`object_bucket_map_`, [storage_backend.h:964](../../../mooncake-store/include/storage_backend.h)),
  **not** in `LocalDiskReplicaData`/`LocalDiskDescriptor`. Reason: the latter is HA-snapshot-serialized
  (`Serializer<Replica>` hard-codes LOCAL_DISK as a 3-element array, [serializer.cpp:709](../../../mooncake-store/src/serializer.cpp),
  version-pinned); changing it is the **highest merge-risk point**. `StorageObjectMetadata` is client-local state, not
  HA-serialized — adding a field has zero serialization risk. Master-side `guaranteed_until_` still lives on
  `ObjectMetadata` (runtime state, reset on HA restart per parent design 7.10), used only for renewal decisions.
- `NotifyOffloadSuccess` success branch ([master_service.cpp:4832](../../../mooncake-store/src/master_service.cpp)):
  the object's `guaranteed_until_` is propagated to the client — via `OffloadTaskItem` carrying `guaranteed_until_ms`
  (Phase 2.A's `bool guaranteed` upgraded to a timestamp); the client's `BuildBucket` writes it into
  `object_bucket_map_`'s `StorageObjectMetadata`.
- **Renewal point**: `GetReplicaList` ([master_service.cpp:2496-2504](../../../mooncake-store/src/master_service.cpp)),
  on reading a LOCAL_DISK replica, extends the master-side `guaranteed_until_`.
- **Cross-node sync (confirmed approach)**: the client gets the refreshed `guaranteed_until` on the
  `GetReplicaList`/read path and updates its local `object_bucket_map_`. `SelectEvictionCandidate` compares the local
  `guaranteed_until > now` to decide whether to skip (holds the storage mutex, does not query the master — matching the
  chosen approach).

**3.B Expiry downgrade**
- After a guaranteed SSD replica's `guaranteed_until` expires, it becomes a normal evictable replica.
- client-side `SelectEvictionCandidate` protects only when `guaranteed_until > now`; once expired, normal eviction.
- master-side: once `guaranteed_until_` expires, the corresponding guard (e.g. a future `IsHardPinned` check) or
  eviction-candidate logic relaxes.
- **TTL clock skew**: master and client `system_clock` may diverge; use a loose comparison (client-side grace period)
  to avoid evicting right after a master renewal due to clock skew.

**3.C BatchExpireGuaranteed active-invalidation RPC** (parent design 7.9)
- Ops scenarios (updated system prompt, wrong RAG doc, debugging) need immediate invalidation, not waiting for TTL.
- master endpoint scans shards, matches keys by `prefix_hash`, sets the SSD replica's `guaranteed_until` to epoch.
- **Cross-node**: after master sets expiry, it must notify the client to lift protection (clear the guaranteed flag
  in the client's local `object_bucket_map_`), else the client still protects it as un-expired. Use a new RPC or piggyback
  the invalidated key list on the existing heartbeat.
- Endpoint: `POST /v1/cache/evict` Body `{"prefix_hash": "<blake3_hex>"}` → `{"expired_count": N}`.

### Phase 4: SGLang HiCache integration (end-to-end)
- HiCache Controller write-back path detects cache_control token range → write_through + `guaranteed_until_ms`.
- HiCache L3 read path: when the request carries cache_control, call `GetReplicaListWithGuaranteed` for renewal.
- GUARANTEED_CAPACITY_EXCEEDED downgrades to a plain PutStart (Phase 1 has no capacity limit; add in Phase 3/4 if
  needed).
- Router-side cache_control parsing → cc_token_offsets.
- End-to-end integration tests.

## 12. Phase 1 key files

| File | Changes |
|------|---------|
| [mooncake-store/include/replica.h](../../../mooncake-store/include/replica.h) | `ReplicateConfig.guaranteed_until_ms` (trailing) |
| [mooncake-store/include/master_service.h](../../../mooncake-store/include/master_service.h) | `ObjectMetadata.guaranteed_` member + ctor param; `LocalDiskSegment` gains `guaranteed_offloading_objects` per-client map; `PushOffloadingQueue` signature adds `bool guaranteed` (`IsHardPinned` unchanged); `enable_guaranteed_cache_` member |
| [mooncake-store/include/types.h](../../../mooncake-store/include/types.h) | `OffloadTaskItem.guaranteed` + `YLT_REFL` |
| [mooncake-store/src/master_service.cpp](../../../mooncake-store/src/master_service.cpp) | read `enable_guaranteed_cache_` from config (default false, gates all guaranteed behavior); `AllocateAndInsertMetadata` records marker; `PutEnd` condition + guaranteed; `PushOffloadingQueue` picks map by guaranteed (guaranteed has no limit); `OffloadObjectHeartbeat` merges both maps on drain; `NotifyOffloadSuccess` guaranteed retry into the independent queue; eviction-path call sites unchanged (default false) |
| [mooncake-store/tests/guaranteed_offload_test.cpp](../../../mooncake-store/tests/) | new test file |

## 13. Phase dependencies & boundaries

**Dependency chain**: Phase 1 (write) → Phase 2 (SSD protection) → Phase 3 (TTL) → Phase 4 (integration). Each phase
is the prerequisite of the next, but each delivers value independently.

**Cross-phase consistency**:
- `guaranteed_until_ms` is introduced in Phase 1 (only `>0` checked) and upgraded to real TTL semantics in Phase 3
  (object-level `guaranteed_until_` timestamp + SSD-replica-level `guaranteed_until`). Phase 1's boolean `guaranteed_`
  is a temporary simplification, replaced by a timestamp in Phase 3 — YAGNI.
- `OffloadTaskItem.priority` (Phase 1) and `OffloadTaskItem.guaranteed` (Phase 2.A) can be merged into one field
  (priority already implies guaranteed); decide at Phase 2 design time.

**Per-phase accepted boundaries**:
- **Phase 1**: boundary ② (SSD replica evictable) accepted — guarantees "at-least-once write to SSD".
- **Phase 2**: guaranteed SSD replica permanently protected (no TTL), reclaimed via Phase 3.C active invalidation —
  needs ops intervention or waiting for Phase 3.
- **Phase 3**: TTL cross-node clock skew mitigated by grace period; the 600s offload-task TTL window (boundary ④)
  remains but is eased by Phase 1.C retry's `start_time` refresh.
- **All phases**: with `enable_offload=false`, guaranteed degrades to a normal object (no SSD, no guarantee).

## 14. Maintainability: easing periodic merges with community main

**Constraint**: the entire explicit-cache (guaranteed) feature **cannot be merged into community main**; it must live
on a long-lived branch and be periodically `git merge main`. Thus **minimizing merge conflicts** is a hard design
constraint spanning all phases. The pain of each main merge is determined by "how invasive the change is into core hot
paths" — `master_service.cpp` (~9500 lines), `storage_backend.cpp`, `serializer.cpp` are upstream high-churn files; the
deeper the edit, the worse.

### 14.1 Design principles

1. **Additive over invasive** — add fields/functions/RPCs/config rather than modify existing function bodies. New
   fields are always **trailing + defaulted** (aggregate struct_pack tolerates trailing; `group_ids` is precedent).
2. **Defaulted params keep existing callers unchanged** — `PushOffloadingQueue(..., bool guaranteed = false)`
   etc.; new callers pass `guaranteed=true`, existing callers are untouched, upstream-added callers pick the default
   false without breakage.
3. **Isolate logic into named helpers** — guaranteed-specific logic is extracted into named methods (e.g.
   `MaybeEnqueueGuaranteedOffload(...)`, `HandleGuaranteedOffloadFailure(...)`); hot paths gain **one call site**
   rather than inline logic. When main edits the hot function body, the merge conflict is on one call line, not a whole
   logic block.
4. **Avoid serialization format bumps** — the **highest-risk point** is `Serializer<Replica>` (LOCAL_DISK hard-coded
   3-element, [serializer.cpp:709](../../../mooncake-store/src/serializer.cpp), version-pinned). Phase 3 stores TTL on
   the client-side `StorageObjectMetadata` (local, not HA-serialized) rather than `LocalDiskDescriptor`, **fully
   avoiding** serialization changes. `ObjectMetadata`'s `guaranteed_`/`guaranteed_until_` are runtime state (reset on
   HA), reconstructed as defaults at snapshot load, not added to `Serializer` (verify the snapshot path during planning).
5. **New code in new files** — tests in a new file (planned); consider a `guaranteed_manager` helper module to
   centralize guaranteed logic and pull edits out of `master_service.cpp`.
6. **Feature flag gating** — add config `enable_guaranteed_cache` (default false) + gflag. All guaranteed behavior is
   gated under `enable_guaranteed_cache_`. When merging upstream the feature is **dormant** (default off), zero
   behavior change, safer merges, easier bisect, can be disabled anytime. Cost: a few `if`s in hot paths (minor).

### 14.2 Per-phase risk & mitigation

| Phase | Change site | Merge risk | Mitigation |
|------|--------|---------|------|
| Phase 1 | `ReplicateConfig` trailing field | low | trailing + default 0 |
| Phase 1 | `PushOffloadingQueue` signature | low | default param `guaranteed=false` |
| Phase 1 | `PutEnd`/`NotifyOffloadSuccess`/`AllocateAndInsertMetadata` hot paths | medium | extract helper, single call site |
| Phase 1 | `ObjectMetadata` snapshot serialization | medium | guaranteed_ runtime state, reconstruct default false |
| Phase 2 | `StorageObjectMetadata`/`OffloadTaskItem` fields | low | trailing + YLT_REFL |
| Phase 2 | `SelectEvictionCandidate` | medium | helper + `guaranteed_key_count` |
| Phase 3 | `GetReplicaList` renewal | medium | helper call site |
| Phase 3 | ~~`Serializer<Replica>` bump~~ | ~~high~~ | **avoided**: TTL on client-local |

### 14.3 Branch strategy

- Long-lived feature branch, periodic **`git merge main` (not rebase)** to sync — preserves history, no force-push,
  team-shareable.
- Design docs (this spec, the parent design) live in `docs/` and merge with the branch so rationale is not lost.
- Prefer **additive conflict resolution**: keep upstream changes + this feature's additions; avoid deleting either side.
- Feature flag defaults off; after a merge, first verify upstream behavior is unchanged, then enable the flag to test
  this feature.

### 14.4 Verification guardrails

- After every main merge, run `scripts/run_ci_test.sh` (mooncake-ci-local skill) to ensure no regression.
- With `enable_guaranteed_cache=false`, all existing tests must pass unchanged (hard guarantee of zero behavior
  change).

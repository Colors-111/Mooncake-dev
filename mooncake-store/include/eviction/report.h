#pragma once

// What one eviction run did, object by object (Effect) and in total (Report).

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "eviction/policy.h"
#include "replica.h"

namespace mooncake {
namespace eviction {

// What reclaiming an object's memory came to.
enum class Outcome : uint8_t {
    // The object lost its last valid replica and goes.
    kDropped,
    // The object lost memory and lives on through its other replicas.
    kDemoted,
    // One memory replica was queued for offload to local disk and is freed by
    // a later run once the copy lands; the rest were released now.
    kOffloadQueued,
    // Nothing was released: the object waits for a later run.
    kKept,
};
inline constexpr size_t kOutcomeCount = 4;

// Why an object went without the offload that was due for it.
enum class DropReason : uint8_t {
    kNone,
    // The run had queued as many offloads as it may.
    kOffloadCap,
    // The offload could not be queued.
    kOffloadRejected,
};

// What reclaiming one object did. Move-only: it carries the replicas taken off
// the object, which must outlive the object's lock.
struct Effect {
    Outcome outcome = Outcome::kKept;
    DropReason reason = DropReason::kNone;
    // Memory taken off the object: freed now, or once the OpLog entry that
    // records it is durable.
    uint64_t released_bytes = 0;
    // Memory pinned for an offload, freed by a later run.
    uint64_t offload_bytes = 0;
    // The object holds nothing valid any more and must be torn down.
    bool needs_teardown = false;
    // The replicas taken off the object, freed when this is destroyed.
    std::vector<Replica> detached;

    Effect() = default;
    Effect(Effect&&) = default;
    Effect& operator=(Effect&&) = default;
    Effect(const Effect&) = delete;
    Effect& operator=(const Effect&) = delete;
};

// Why the policy found nothing to release for an object, as far as its
// replicas tell. The run classifies only objects it already skipped with
// SkipReason::kNothingToRelease, so no healthy object pays for this.
enum class NothingCause : uint8_t {
    // Every COMPLETE memory replica lost its allocator or segment lifetime.
    kPermanentlyUnavailable,
    // A COMPLETE memory replica still holds an allocator, but none of them can
    // serve a read right now (unreadable or awaiting remount).
    kTemporarilyUnavailable,
    // Every serveable COMPLETE memory replica is still held by a reference
    // count.
    kRefcntBusy,
    // The object holds no COMPLETE memory replica at all.
    kNoCompleteMemory,
    kOther,
};
inline constexpr size_t kNothingCauseCount = 5;

// A set of durations, summed and maxed over the objects a run sampled.
struct Timing {
    uint64_t samples = 0;
    uint64_t sum_ns = 0;
    uint64_t max_ns = 0;

    void Add(uint64_t elapsed_ns) {
        ++samples;
        sum_ns += elapsed_ns;
        if (elapsed_ns > max_ns) {
            max_ns = elapsed_ns;
        }
    }
};

// What a run did in total.
struct Report {
    // Which round the caller runs this for, from its own sequence; 0 if none.
    uint64_t round_id = 0;
    // Keys the eviction indexes had filed when the run began.
    uint64_t filed = 0;
    // Keys the walk tried to hold, and those it could not because the object
    // was gone or replaced; attempted == missing + visited.
    uint64_t attempted = 0;
    uint64_t missing = 0;
    uint64_t visited = 0;
    // Objects that released memory, and the memory they released.
    uint64_t evicted_objects = 0;
    uint64_t released_bytes = 0;
    // Memory pinned for offloads, freed by a later run.
    uint64_t offload_bytes = 0;
    std::array<uint64_t, kOutcomeCount> outcomes{};
    // Objects dropped without the offload due for them, by DropReason.
    uint64_t offload_cap_drops = 0;
    uint64_t offload_rejected_drops = 0;
    // Keys left alone, by SkipReason.
    std::array<uint64_t, static_cast<size_t>(SkipReason::kCount)> skips{};
    // Objects the policy found nothing to release for, by cause.
    std::array<uint64_t, kNothingCauseCount> nothing{};
    // Objects the index took out of its candidates for good after finding
    // nothing to release.
    uint64_t suppressed_permanently_unavailable = 0;
    // How reading the index's buckets went, and the segments of a sampled
    // object's processing. Sample counts bound every timing.
    uint64_t index_bucket_reads = 0;
    uint64_t index_read_ns = 0;
    uint64_t index_max_read_ns = 0;
    Timing route_acquire;
    Timing reclaim;
    Timing guard_release;
    Timing detached_free;
    // An OpLog write failed and cut the run short.
    bool stopped = false;

    [[nodiscard]] uint64_t Count(Outcome outcome) const {
        return outcomes[static_cast<size_t>(outcome)];
    }

    [[nodiscard]] uint64_t SkipCount(SkipReason reason) const {
        return skips[static_cast<size_t>(reason)];
    }

    [[nodiscard]] uint64_t NothingCount(NothingCause cause) const {
        return nothing[static_cast<size_t>(cause)];
    }

    // Something was released or queued for offload.
    [[nodiscard]] bool MadeProgress() const {
        return evicted_objects > 0 || Count(Outcome::kOffloadQueued) > 0;
    }

    void Add(const Effect& effect) {
        ++outcomes[static_cast<size_t>(effect.outcome)];
        released_bytes += effect.released_bytes;
        offload_bytes += effect.offload_bytes;
        if (effect.released_bytes > 0) {
            ++evicted_objects;
        }
        if (effect.reason == DropReason::kOffloadCap) {
            ++offload_cap_drops;
        } else if (effect.reason == DropReason::kOffloadRejected) {
            ++offload_rejected_drops;
        }
    }
};

}  // namespace eviction
}  // namespace mooncake

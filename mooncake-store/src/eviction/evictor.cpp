#include "eviction/evictor.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "random.h"

namespace mooncake {
namespace eviction {
namespace {

// How often a long run tells the caller its cumulative progress.
constexpr auto kProgressInterval = std::chrono::seconds(5);

// A ratio above 1 asks for every eligible object; one that is negative or not
// a number asks for none.
double Sanitized(const char* name, double ratio) {
    if (std::isnan(ratio) || ratio < 0) {
        LOG(ERROR) << name << "=" << ratio << ", error=invalid_params";
        return 0;
    }
    return std::min(ratio, 1.0);
}

uint64_t ElapsedNs(std::chrono::steady_clock::time_point started) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
}

// Adds one duration to `timing` on destruction, or does nothing when the
// object was not sampled.
class ScopeTimer {
   public:
    explicit ScopeTimer(Timing* timing)
        : timing_(timing),
          started_(timing != nullptr
                       ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point{}) {}
    ~ScopeTimer() {
        if (timing_ != nullptr) {
            timing_->Add(ElapsedNs(started_));
        }
    }
    ScopeTimer(const ScopeTimer&) = delete;
    ScopeTimer& operator=(const ScopeTimer&) = delete;

   private:
    Timing* const timing_;
    const std::chrono::steady_clock::time_point started_;
};

// What the object's replicas say about a policy that found nothing to release;
// only objects already skipped that way are classified.
NothingCause ClassifyNothing(const ObjectMetadata& metadata) {
    bool has_complete_memory = false;
    bool has_live_allocator = false;
    bool has_servable = false;
    bool has_free_servable = false;
    for (const Replica& replica : metadata.GetAllReplicas()) {
        if (!replica.is_memory_replica() || !replica.is_completed()) {
            continue;
        }
        has_complete_memory = true;
        if (replica.has_invalid_mem_handle()) {
            continue;
        }
        has_live_allocator = true;
        if (!replica.is_available()) {
            continue;
        }
        has_servable = true;
        if (replica.get_refcnt() == 0) {
            has_free_servable = true;
        }
    }
    if (!has_complete_memory) {
        return NothingCause::kNoCompleteMemory;
    }
    if (!has_live_allocator) {
        return NothingCause::kPermanentlyUnavailable;
    }
    if (!has_servable) {
        return NothingCause::kTemporarilyUnavailable;
    }
    if (!has_free_servable) {
        return NothingCause::kRefcntBusy;
    }
    return NothingCause::kOther;
}

}  // namespace

// Either goal, as a run holds each namespace to it.
struct Evictor::Aim {
    // Whether a namespace that filed `filed` keys, and has given up `evicted`
    // objects and `released` bytes, has met its share of the target, or for
    // the fallback of the minimum. A share is rounded up, as BatchEvict rounds
    // its pool's. A group goes all or none, so a namespace may give up more.
    [[nodiscard]] bool Met(uint64_t filed, uint64_t evicted, uint64_t released,
                           bool fallback) const {
        if (in_bytes) {
            return released >= target_bytes;
        }
        const double ratio = fallback ? min_ratio : target_ratio;
        return evicted >= static_cast<uint64_t>(std::ceil(filed * ratio));
    }

    // Set for a ByteGoal, which uses `target_bytes` alone.
    bool in_bytes = false;
    double target_ratio = 0;
    double min_ratio = 0;
    uint64_t target_bytes = 0;
};

// One namespace's part of a run: the clock leases are judged against, the aim
// and the report it adds to, the namespace's cursor, the report as the
// namespace's part began, and the objects its policy deferred.
class Evictor::Pass {
   public:
    Pass(const Evictor& evictor, metadata::Namespace& ns,
         std::unique_ptr<metadata::EvictionIndex::Cursor> cursor,
         const Aim& aim, Report& report, Clock::time_point now)
        : evictor_(evictor),
          ns_(ns),
          cursor_(std::move(cursor)),
          filed_(cursor_->Total()),
          evicted_before_(report.evicted_objects),
          released_before_(report.released_bytes),
          aim_(aim),
          report_(report),
          now_(now) {}

    // Walks the cursor until the namespace's share of the target is met, then
    // the objects the policy deferred until its share of the minimum is; each
    // also ends when an OpLog write fails or nothing is left. A group's
    // members come in one entry and go together, one key lock at a time, in
    // the walk and in the fallback alike.
    void Run(const ProgressFn* progress) {
        progress_ = progress;
        while (!report_.stopped && !Met(/*fallback=*/false)) {
            std::optional<Entry> entry = cursor_->Next();
            if (!entry) {
                break;
            }
            deferred_.emplace_back();
            for (const std::string_view key : entry->keys) {
                if (report_.stopped) {
                    break;
                }
                Evict(key, &*entry);
            }
            if (deferred_.back().empty()) {
                deferred_.pop_back();
            }
        }
        for (const std::vector<route::ObjectRef>& refs : deferred_) {
            if (report_.stopped || Met(/*fallback=*/true)) {
                break;
            }
            for (const route::ObjectRef& ref : refs) {
                if (report_.stopped) {
                    break;
                }
                // Only the publication the policy deferred: a replacement was
                // never judged.
                Evict(ref, nullptr);
            }
        }
        // How the index's own bucket reads went, for the diagnostics.
        const auto timing = cursor_->Timing();
        report_.index_bucket_reads += timing.bucket_reads;
        report_.index_read_ns += timing.read_ns;
        report_.index_max_read_ns =
            std::max(report_.index_max_read_ns, timing.max_read_ns);
    }

   private:
    using Entry = metadata::EvictionIndex::Entry;

    [[nodiscard]] bool Met(bool fallback) const {
        return aim_.Met(filed_, report_.evicted_objects - evicted_before_,
                        report_.released_bytes - released_before_, fallback);
    }

    // The key or the publication held exclusively, timing the acquisition
    // when `sampled`. A guard is move-constructible but not assignable, so the
    // optional is built from it directly.
    template <typename Key>
    std::optional<route::WriteGuard> Acquire(const Key& key, bool sampled) {
        if (!sampled) {
            return ns_.objects.Write(key);
        }
        const auto started = std::chrono::steady_clock::now();
        auto hold = ns_.objects.Write(key);
        report_.route_acquire.Add(ElapsedNs(started));
        return hold;
    }

    // Holds the object under `key` or `ref`, judges it, and frees what
    // reclaiming took once the lock is let go.
    void Evict(std::string_view key, const Entry* entry) {
        ++report_.attempted;
        const bool sampled = evictor_.diagnostics_.sample_percent > 0 &&
                             randomUniform<uint32_t>(1, 100) <=
                                 evictor_.diagnostics_.sample_percent;
        auto hold = Acquire(key, sampled);
        if (!hold) {
            // Gone, or replaced since the index handed the key out.
            ++report_.missing;
            MaybeReportProgress();
            return;
        }
        ++report_.visited;
        std::vector<Replica> detached = EvictLocked(*hold, entry, sampled);
        if (sampled) {
            auto started = std::chrono::steady_clock::now();
            hold.reset();
            report_.guard_release.Add(ElapsedNs(started));
            started = std::chrono::steady_clock::now();
            detached.clear();
            report_.detached_free.Add(ElapsedNs(started));
        } else {
            hold.reset();
        }
        MaybeReportProgress();
    }

    void Evict(const route::ObjectRef& ref, const Entry* entry) {
        ++report_.attempted;
        const bool sampled = evictor_.diagnostics_.sample_percent > 0 &&
                             randomUniform<uint32_t>(1, 100) <=
                                 evictor_.diagnostics_.sample_percent;
        auto hold = Acquire(ref, sampled);
        if (!hold) {
            ++report_.missing;
            MaybeReportProgress();
            return;
        }
        ++report_.visited;
        std::vector<Replica> detached = EvictLocked(*hold, entry, sampled);
        if (sampled) {
            auto started = std::chrono::steady_clock::now();
            hold.reset();
            report_.guard_release.Add(ElapsedNs(started));
            started = std::chrono::steady_clock::now();
            detached.clear();
            report_.detached_free.Add(ElapsedNs(started));
        } else {
            hold.reset();
        }
        MaybeReportProgress();
    }

    void MaybeReportProgress() {
        if (progress_ == nullptr || !*progress_) {
            return;
        }
        // Check the clock once every 256 objects, not once per object, so the
        // healthy path never pays a clock call per key.
        if ((report_.attempted & 0xFFu) != 0) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now < next_progress_) {
            return;
        }
        (*progress_)(report_);
        next_progress_ = now + kProgressInterval;
    }

    // Reclaims the object `hold` holds if it qualifies, tears it down if that
    // left it nothing valid, and returns the replicas it took. `sampled` times
    // the policy, reclaim and teardown together.
    std::vector<Replica> EvictLocked(const route::WriteGuard& hold,
                                     const Entry* entry, bool sampled) {
        ScopeTimer timer(sampled ? &report_.reclaim : nullptr);
        // Filed ahead of its real rank: letting the lock go files it there,
        // where the walk meets it in its turn.
        if (entry != nullptr && !cursor_->StillAt(hold, *entry)) {
            Skip(hold, SkipReason::kMoved);
            return {};
        }
        auto plan = Judge(hold, /*fallback=*/entry == nullptr);
        if (!plan) {
            return {};
        }
        auto effect = evictor_.reclaimer_.Apply(ns_, hold, *plan, report_);
        if (!effect) {
            report_.stopped = true;
            return {};
        }
        report_.Add(*effect);
        const ObjectView view{ns_.id(), hold.key(), hold.metadata()};
        for (EvictionObserver* observer : evictor_.observers_) {
            observer->OnApplied(view, *effect);
        }
        if (effect->needs_teardown) {
            evictor_.reclaimer_.TearDown(ns_, hold);
        }
        return std::move(effect->detached);
    }

    // The plan for the object `hold` holds, or why it is skipped. An object
    // the policy defers on the walk is kept for the fallback. A policy that
    // finds nothing to release opens the one abnormal branch that inspects the
    // replicas: it classifies why and asks the index to suppress a candidate
    // that can never release memory. No healthy object pays for either.
    tl::expected<Plan, SkipReason> Judge(const route::WriteGuard& hold,
                                         bool fallback) {
        const ObjectMetadata& metadata = hold.metadata();
        // Not const: ObjectMetadata::IsLeaseExpired takes it by reference.
        Clock::time_point now = now_;
        SkipReason reason;
        if (metadata.IsHardPinned()) {
            reason = SkipReason::kHardPinned;
        } else if (!metadata.IsLeaseExpired(now)) {
            // A surviving shared group lease keeps every member alive.
            reason = SkipReason::kLeaseActive;
        } else {
            auto plan = evictor_.policy_.Decide(
                Candidate{ns_.id(), hold.key(), metadata, now_, fallback});
            if (plan) {
                return plan;
            }
            reason = plan.error();
            if (reason == SkipReason::kDeferred && !fallback) {
                deferred_.back().push_back(hold.ref());
            } else if (reason == SkipReason::kNothingToRelease) {
                ++report_
                      .nothing[static_cast<size_t>(ClassifyNothing(metadata))];
                if (ns_.eviction_index().SuppressPermanentlyUnavailable(hold)) {
                    ++report_.suppressed_permanently_unavailable;
                }
            }
        }
        Skip(hold, reason);
        return tl::make_unexpected(reason);
    }

    void Skip(const route::WriteGuard& hold, SkipReason reason) {
        ++report_.skips[static_cast<size_t>(reason)];
        const ObjectView view{ns_.id(), hold.key(), hold.metadata()};
        for (EvictionObserver* observer : evictor_.observers_) {
            observer->OnSkipped(view, reason);
        }
    }

    const Evictor& evictor_;
    metadata::Namespace& ns_;
    const std::unique_ptr<metadata::EvictionIndex::Cursor> cursor_;
    // The keys the index filed when the run began.
    const uint64_t filed_;
    const uint64_t evicted_before_;
    const uint64_t released_before_;
    const Aim& aim_;
    Report& report_;
    const Clock::time_point now_;
    // In the order the walk met them, one list for each entry that had any.
    std::vector<std::vector<route::ObjectRef>> deferred_;
    // Set by Run for a caller that asked to hear of long progress.
    const ProgressFn* progress_ = nullptr;
    std::chrono::steady_clock::time_point next_progress_{};
};

Evictor::Evictor(metadata::NamespaceTable& namespaces,
                 const EvictionPolicy& policy, Reclaimer& reclaimer,
                 std::vector<EvictionObserver*> observers,
                 Diagnostics diagnostics)
    : namespaces_(namespaces),
      policy_(policy),
      reclaimer_(reclaimer),
      observers_(std::move(observers)),
      diagnostics_(diagnostics) {}

Report Evictor::Run(const Scope& scope, const ObjectGoal& goal,
                    uint64_t round_id, ProgressFn progress) {
    Aim aim{.target_ratio = Sanitized("evict_ratio_target", goal.target_ratio),
            .min_ratio = Sanitized("evict_ratio_lowerbound", goal.min_ratio)};
    if (aim.min_ratio > aim.target_ratio) {
        LOG(ERROR) << "evict_ratio_target=" << aim.target_ratio
                   << ", evict_ratio_lowerbound=" << aim.min_ratio
                   << ", error=invalid_params";
        aim.min_ratio = aim.target_ratio;
    }
    return RunFor(scope, aim, round_id, progress);
}

Report Evictor::Run(const TenantId& tenant_id, const ByteGoal& goal,
                    uint64_t round_id, ProgressFn progress) {
    return RunFor(Scope::Of(tenant_id),
                  Aim{.in_bytes = true, .target_bytes = goal.target_bytes},
                  round_id, progress);
}

Report Evictor::RunFor(const Scope& scope, const Aim& aim, uint64_t round_id,
                       const ProgressFn& progress) {
    std::vector<metadata::Namespace*> namespaces;
    if (const TenantId* tenant_id = scope.tenant_id()) {
        // A tenant with no namespace owns no object.
        if (metadata::Namespace* ns = namespaces_.Lookup(*tenant_id)) {
            namespaces.push_back(ns);
        }
    } else {
        namespaces_.Visit([&](const TenantId&, metadata::Namespace& ns) {
            namespaces.push_back(&ns);
        });
    }

    Report report;
    report.round_id = round_id;
    const Clock::time_point now = Clock::now();
    std::vector<std::unique_ptr<metadata::EvictionIndex::Cursor>> cursors;
    cursors.reserve(namespaces.size());
    for (metadata::Namespace* ns : namespaces) {
        cursors.push_back(ns->eviction_index().Scan(now));
        report.filed += cursors.back()->Total();
    }

    // A run an OpLog write cuts short leaves the namespaces after that one to
    // the next run, which starts one namespace further on.
    const size_t start = next_start_.fetch_add(1, std::memory_order_relaxed);
    for (size_t n = 0; n < namespaces.size() && !report.stopped; ++n) {
        const size_t i = (start + n) % namespaces.size();
        Pass(*this, *namespaces[i], std::move(cursors[i]), aim, report, now)
            .Run(&progress);
    }

    for (EvictionObserver* observer : observers_) {
        observer->OnRunEnd(scope, report);
    }
    return report;
}

}  // namespace eviction
}  // namespace mooncake

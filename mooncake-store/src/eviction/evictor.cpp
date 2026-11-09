#include "eviction/evictor.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace mooncake {
namespace eviction {
namespace {

// A ratio above 1 asks for every eligible object; one that is negative or not
// a number asks for none.
double Sanitized(const char* name, double ratio) {
    if (std::isnan(ratio) || ratio < 0) {
        LOG(ERROR) << name << "=" << ratio << ", error=invalid_params";
        return 0;
    }
    return std::min(ratio, 1.0);
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
         std::unique_ptr<metadata::EvictionIndex::Cursor> cursor, Aim aim,
         Report& report, Clock::time_point now,
         const std::function<bool()>* stop)
        : evictor_(evictor),
          ns_(ns),
          cursor_(std::move(cursor)),
          filed_(cursor_->Total()),
          evicted_before_(report.evicted_objects),
          released_before_(report.released_bytes),
          aim_(std::move(aim)),
          report_(report),
          now_(now),
          stop_(stop) {}

    // Walks the cursor until the namespace's share of the target is met, then
    // the objects the policy deferred until its share of the minimum is; each
    // also ends when an OpLog write fails or nothing is left. A group's
    // members come in one entry and go together, one key lock at a time, in
    // the walk and in the fallback alike.
   public:
    enum class StepKind { kMore, kDone };

    // Walks until the aim is met, the cursor is spent, or `max_candidates`
    // keys were judged; a group finishes before the slice yields. Calling it
    // again resumes where it stopped.
    StepKind Step(size_t max_candidates) {
        size_t judged = 0;
        while (!walk_done_) {
            if (report_.stopped || Met(/*fallback=*/false)) {
                walk_done_ = true;
                break;
            }
            if (max_candidates != kUnbounded && judged >= max_candidates) {
                return StepKind::kMore;
            }
            std::optional<Entry> entry = cursor_->Next();
            if (!entry) {
                walk_done_ = true;
                break;
            }
            deferred_.emplace_back();
            for (const std::string_view key : entry->keys) {
                if (report_.stopped) {
                    break;
                }
                Evict(ns_.objects.Write(key), &*entry);
                ++judged;
                ++judged_since_check_;
            }
            if (deferred_.back().empty()) {
                deferred_.pop_back();
            }
            // Asked only once the entry (a whole group, at most) is done, so
            // a stop never leaves half a group evicted.
            if (ShouldHalt()) {
                return StepKind::kDone;
            }
        }
        while (!fallback_done_) {
            if (report_.stopped || Met(/*fallback=*/true)) {
                fallback_done_ = true;
                break;
            }
            if (max_candidates != kUnbounded && judged >= max_candidates) {
                return StepKind::kMore;
            }
            if (fallback_entry_ >= deferred_.size()) {
                fallback_done_ = true;
                break;
            }
            for (const route::ObjectRef& ref : deferred_[fallback_entry_]) {
                if (report_.stopped) {
                    break;
                }
                // Only the publication the policy deferred: a replacement was
                // never judged.
                Evict(ns_.objects.Write(ref), /*entry=*/nullptr);
                ++judged;
                ++judged_since_check_;
            }
            ++fallback_entry_;
            if (ShouldHalt()) {
                return StepKind::kDone;
            }
        }
        return StepKind::kDone;
    }

    void Run() {
        while (Step(kUnbounded) == StepKind::kMore) {
        }
    }

   private:
   private:
    using Entry = metadata::EvictionIndex::Entry;

    // The caller's stop predicate, asked every batch of judged keys, so a
    // healthy round pays one counter increment per key and nothing else. A
    // halt is not an OpLog failure: the report keeps what this round released.
    bool ShouldHalt() {
        if (stop_ == nullptr || !*stop_) {
            return false;
        }
        if (judged_since_check_ < kStopCheckInterval) {
            return false;
        }
        judged_since_check_ = 0;
        if (!(*stop_)()) {
            return false;
        }
        halted_ = true;
        report_.interrupted = true;
        return true;
    }

    static constexpr size_t kStopCheckInterval = 4096;

    [[nodiscard]] bool Met(bool fallback) const {
        return aim_.Met(filed_, report_.evicted_objects - evicted_before_,
                        report_.released_bytes - released_before_, fallback);
    }

    // Reclaims the object `hold` holds, met on the walk in `entry` or, with
    // no entry, in the fallback, and frees what that took once the lock is
    // let go.
    void Evict(std::optional<route::WriteGuard> hold, const Entry* entry) {
        if (!hold) {
            return;
        }
        // One increment per judged key: cheap enough for the healthy path,
        // and it is what tells a slow round from a short one.
        ++report_.visited;
        // Freed as this returns, after the lock is let go.
        const std::vector<Replica> detached = EvictLocked(*hold, entry);
        hold.reset();
    }

    // Reclaims the object `hold` holds if it qualifies, tears it down if that
    // left it nothing valid, and returns the replicas it took.
    std::vector<Replica> EvictLocked(const route::WriteGuard& hold,
                                     const Entry* entry) {
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
    // the policy defers on the walk is kept for the fallback.
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
            }
        }
        Skip(hold, reason);
        return tl::make_unexpected(reason);
    }

    void Skip(const route::WriteGuard& hold, SkipReason reason) {
        if (reason == SkipReason::kNothingToRelease) {
            ++report_.nothing_to_release;
        }
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
    // Owned by value: a cursor outlives the call that built its aim.
    const Aim aim_;
    Report& report_;
    const Clock::time_point now_;
    const std::function<bool()>* const stop_;
    bool halted_ = false;
    size_t judged_since_check_ = 0;
    // In the order the walk met them, one list for each entry that had any.
    std::vector<std::vector<route::ObjectRef>> deferred_;
    // Slice bounds: where the walk and the fallback stopped.
    static constexpr size_t kUnbounded = std::numeric_limits<size_t>::max();
    bool walk_done_{false};
    bool fallback_done_{false};
    size_t fallback_entry_{0};
};

Evictor::TenantCursor::TenantCursor(const Evictor& evictor, TenantId tenant,
                                    metadata::Namespace* ns,
                                    uint64_t index_instance,
                                    Clock::time_point now, const ByteGoal& goal)
    : evictor_(evictor),
      tenant_(std::move(tenant)),
      ns_(ns),
      index_instance_(index_instance),
      now_(now),
      goal_(goal) {}

Evictor::TenantCursor::~TenantCursor() = default;

std::unique_ptr<Evictor::TenantCursor> Evictor::BeginTenantWork(
    const TenantId& tenant_id, const ByteGoal& goal, Clock::time_point now) {
    metadata::Namespace* ns = namespaces_.Lookup(tenant_id);
    if (ns == nullptr) {
        return nullptr;
    }
    auto cursor = std::unique_ptr<TenantCursor>(new TenantCursor(
        *this, tenant_id, ns, ns->eviction_index().instance_id(), now, goal));
    // What this tenant's index held when the work began, for the period's
    // measurements; the pass counts its own baseline for the aim.
    cursor->report_.filed = ns->eviction_index().Scan(now)->Total();
    return cursor;
}

Evictor::WorkStep Evictor::StepTenantWork(TenantCursor& cursor,
                                          size_t max_candidates) {
    if (namespaces_.Lookup(cursor.tenant_) != cursor.ns_ ||
        cursor.ns_->eviction_index().instance_id() != cursor.index_instance_) {
        return WorkStep::kInvalidated;
    }
    if (!cursor.pass_) {
        const Aim aim{.in_bytes = true,
                      .target_bytes = cursor.goal_.target_bytes};
        cursor.pass_ = std::make_unique<Pass>(
            *this, *cursor.ns_, cursor.ns_->eviction_index().Scan(cursor.now_),
            aim, cursor.report_, cursor.now_, /*stop=*/nullptr);
    }
    return cursor.pass_->Step(max_candidates) == Pass::StepKind::kDone
               ? WorkStep::kDone
               : WorkStep::kMore;
}

const Report& Evictor::WorkReport(const TenantCursor& cursor) const {
    return cursor.report_;
}

Evictor::Evictor(metadata::NamespaceTable& namespaces,
                 const EvictionPolicy& policy, Reclaimer& reclaimer,
                 std::vector<EvictionObserver*> observers)
    : namespaces_(namespaces),
      policy_(policy),
      reclaimer_(reclaimer),
      observers_(std::move(observers)) {}

Report Evictor::Run(const Scope& scope, const ObjectGoal& goal,
                    std::function<bool()> stop) {
    Aim aim{.target_ratio = Sanitized("evict_ratio_target", goal.target_ratio),
            .min_ratio = Sanitized("evict_ratio_lowerbound", goal.min_ratio)};
    if (aim.min_ratio > aim.target_ratio) {
        LOG(ERROR) << "evict_ratio_target=" << aim.target_ratio
                   << ", evict_ratio_lowerbound=" << aim.min_ratio
                   << ", error=invalid_params";
        aim.min_ratio = aim.target_ratio;
    }
    const std::function<bool()>* stop_ptr = stop ? &stop : nullptr;
    return RunFor(scope, aim, stop_ptr);
}

Report Evictor::Run(const TenantId& tenant_id, const ByteGoal& goal) {
    return RunFor(Scope::Of(tenant_id),
                  Aim{.in_bytes = true, .target_bytes = goal.target_bytes},
                  /*stop=*/nullptr);
}

Report Evictor::RunFor(const Scope& scope, const Aim& aim,
                       const std::function<bool()>* stop) {
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
    // An interrupted round must not start on the next namespace: the caller
    // is switching modes, not asking for the rest of this round.
    for (size_t n = 0;
         n < namespaces.size() && !report.stopped && !report.interrupted; ++n) {
        const size_t i = (start + n) % namespaces.size();
        Pass(*this, *namespaces[i], std::move(cursors[i]), aim, report, now,
             stop)
            .Run();
    }

    for (EvictionObserver* observer : observers_) {
        observer->OnRunEnd(scope, report);
    }
    return report;
}

}  // namespace eviction
}  // namespace mooncake

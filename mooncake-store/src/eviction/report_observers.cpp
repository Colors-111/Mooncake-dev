#include "eviction/report_observers.h"

#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

#include "master_metric_manager.h"

namespace mooncake {
namespace eviction {
namespace {

int64_t Saturated(uint64_t value) {
    return static_cast<int64_t>(std::min<uint64_t>(
        value, static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
}

const char* SkipName(SkipReason reason) {
    switch (reason) {
        case SkipReason::kMoved:
            return "moved";
        case SkipReason::kHardPinned:
            return "hard_pinned";
        case SkipReason::kLeaseActive:
            return "lease_active";
        case SkipReason::kDeferred:
            return "deferred";
        case SkipReason::kSoftPinned:
            return "soft_pinned";
        case SkipReason::kExternalEvict:
            return "external_evict";
        case SkipReason::kNothingToRelease:
            return "nothing_to_release";
        case SkipReason::kCount:
            break;
    }
    return "unknown";
}

const char* NothingName(NothingCause cause) {
    switch (cause) {
        case NothingCause::kPermanentlyUnavailable:
            return "permanently_unavailable";
        case NothingCause::kTemporarilyUnavailable:
            return "temporarily_unavailable";
        case NothingCause::kRefcntBusy:
            return "refcnt_busy";
        case NothingCause::kNoCompleteMemory:
            return "no_complete_memory";
        case NothingCause::kOther:
            return "other";
    }
    return "unknown";
}

// A timing as microseconds, with its sample count so a sampled figure is not
// read as a full one.
void AppendTiming(std::ostream& out, const char* name, const Timing& timing) {
    out << " " << name << "_samples=" << timing.samples << " " << name
        << "_avg_us="
        << (timing.samples == 0 ? 0 : timing.sum_ns / timing.samples / 1000)
        << " " << name << "_max_us=" << timing.max_ns / 1000;
}

// Whether a per-run diagnostic summary is due at INFO, at most once a second;
// the rounds in between still get one VLOG line each. A pool under pressure
// can run a round every few milliseconds, and a line per round would swamp
// the log.
bool DiagnosticLogDue() {
    static std::atomic<int64_t> last_ms{0};
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
    int64_t previous = last_ms.load(std::memory_order_relaxed);
    if (now_ms - previous < 1000) {
        return false;
    }
    return last_ms.compare_exchange_strong(previous, now_ms);
}

}  // namespace

void MetricsObserver::OnRunEnd(const Scope& scope, const Report& report) {
    const TenantId* tenant_id = scope.tenant_id();
    if (tenant_id != nullptr && report.released_bytes > 0) {
        MasterMetricManager::instance().inc_tenant_evict_bytes(
            tenant_id->value(), Saturated(report.released_bytes));
    }
}

void LogObserver::OnRunEnd(const Scope& scope, const Report& report) {
    const uint64_t offloads = report.Count(Outcome::kOffloadQueued);
    const TenantId* tenant_id = scope.tenant_id();
    if (tenant_id == nullptr) {
        if (report.filed == 0) {
            VLOG(1) << "[EVICT-DIAG] eviction_base=0 "
                       "(no evictable memory objects)";
        } else {
            const double actual_evict_ratio =
                static_cast<double>(report.evicted_objects) / report.filed;
            VLOG(1) << "action=evict_objects"
                    << ", evicted_count=" << report.evicted_objects
                    << ", offload_deferred=" << offloads
                    << ", offload_cap_forced=" << report.offload_cap_drops
                    << ", offload_push_failed_forced="
                    << report.offload_rejected_drops
                    << ", total_freed_size=" << report.released_bytes
                    << ", eviction_base=" << report.filed
                    << ", actual_evict_ratio=" << actual_evict_ratio;
            LOG(INFO) << "[EVICT-RESULT] evicted_count="
                      << report.evicted_objects
                      << ", eviction_base=" << report.filed
                      << ", actual_evict_ratio=" << actual_evict_ratio;
        }
    }

    const std::string prefix =
        tenant_id == nullptr ? "[EVICT] " : "[TENANT-EVICT] ";
    const std::string whose = tenant_id == nullptr
                                  ? std::string()
                                  : " for tenant " + tenant_id->value();
    if (report.released_bytes == 0 && offloads > 0) {
        LOG(WARNING) << prefix << "No memory freed" << whose << "; " << offloads
                     << " object(s) deferred for disk offload. "
                        "Consider lowering eviction_high_watermark_ratio.";
    }
    if (report.offload_cap_drops > 0) {
        LOG(WARNING) << prefix << "Offload cap (" << offload_cap_ << ") reached"
                     << whose << "; force-evicted " << report.offload_cap_drops
                     << " object(s) without disk offload.";
    }
    if (report.offload_rejected_drops > 0) {
        LOG(WARNING) << prefix << "PushOffloadingQueue failed" << whose
                     << " for " << report.offload_rejected_drops
                     << " object(s); force-evicted without disk offload "
                        "(offload_force_evict=true).";
    }

    // The diagnostic summary of one run, without a line per key: what the walk
    // tried, what it could not hold, how every skip split, and what the
    // nothing-to-release branch found. `visited` covers the keys actually
    // judged; `attempted = missing + visited`. The lines are rate limited, so
    // a pool running rounds every few milliseconds does not swamp the log.
    const bool due = DiagnosticLogDue();
    const auto emit = [due](const std::string& line) {
        if (due) {
            LOG(INFO) << line;
        } else {
            VLOG(1) << line;
        }
    };
    if (report.attempted > 0 || report.filed > 0) {
        std::ostringstream line;
        line << "[EVICT-PROGRESS] round_id=" << report.round_id
             << " scope=" << (tenant_id == nullptr ? "pool" : "tenant")
             << " attempted=" << report.attempted
             << " missing=" << report.missing << " visited=" << report.visited
             << " evicted=" << report.evicted_objects
             << " released_bytes=" << report.released_bytes
             << " filed=" << report.filed << " stopped=" << report.stopped;
        emit(line.str());
    }
    uint64_t skipped = 0;
    for (const uint64_t count : report.skips) {
        skipped += count;
    }
    if (skipped > 0) {
        std::ostringstream line;
        line << "[EVICT-SKIP] round_id=" << report.round_id
             << " scope=" << (tenant_id == nullptr ? "pool" : "tenant");
        for (size_t i = 0; i < report.skips.size(); ++i) {
            line << " " << SkipName(static_cast<SkipReason>(i)) << "="
                 << report.skips[i];
        }
        emit(line.str());
    }
    uint64_t nothing = 0;
    for (const uint64_t count : report.nothing) {
        nothing += count;
    }
    if (nothing > 0 || report.suppressed_permanently_unavailable > 0) {
        std::ostringstream line;
        line << "[EVICT-NOTHING] round_id=" << report.round_id
             << " scope=" << (tenant_id == nullptr ? "pool" : "tenant");
        for (size_t i = 0; i < report.nothing.size(); ++i) {
            line << " " << NothingName(static_cast<NothingCause>(i)) << "="
                 << report.nothing[i];
        }
        line << " suppressed=" << report.suppressed_permanently_unavailable;
        emit(line.str());
    }
    if (report.index_bucket_reads > 0 || report.route_acquire.samples > 0) {
        std::ostringstream line;
        line << "[EVICT-TIMING] round_id=" << report.round_id
             << " scope=" << (tenant_id == nullptr ? "pool" : "tenant")
             << " index_bucket_reads=" << report.index_bucket_reads
             << " index_avg_us="
             << (report.index_bucket_reads == 0
                     ? 0
                     : report.index_read_ns / report.index_bucket_reads / 1000)
             << " index_max_us=" << report.index_max_read_ns / 1000;
        AppendTiming(line, "route_acquire", report.route_acquire);
        AppendTiming(line, "reclaim", report.reclaim);
        AppendTiming(line, "guard_release", report.guard_release);
        AppendTiming(line, "detached_free", report.detached_free);
        emit(line.str());
    }
}

}  // namespace eviction
}  // namespace mooncake

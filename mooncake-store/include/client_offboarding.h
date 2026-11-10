#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "client_liveness.h"
#include "metadata/object_route.h"
#include "tenant_id.h"
#include "types.h"
#include "unmount_log.h"

namespace mooncake {

namespace test {
class MasterServiceTest;
}

// A segment whose unmount this job prepared and therefore must commit.
struct PendingSegmentOffboarding {
    UUID segment_id;
    std::string segment_name;
    std::string transport_endpoint;
};

struct PreparedSegmentOffboarding {
    UUID segment_id;
    std::string segment_name;
    std::string transport_endpoint;
    size_t metrics_dec_capacity{0};
    // Held for the rest of the job so a retry still sees that the unmount
    // entry is logged after the master's index dropped it.
    std::shared_ptr<SegmentUnmountLogState> unmount_log;
};

// What one metadata cleanup pass over the namespaces is doing. Published to
// the metrics scrape so a management thread can tell where a long job is and
// how long it has been there when no per-object log is due.
enum class ClientOffboardingStage : int {
    kIdle = 0,
    kPrepare = 1,
    kCollect = 2,
    kClean = 3,
    // Waiting for an affiliated PROCESSING replica to finish, or for a segment
    // prepare to settle, before the sweep may finish.
    kLifecycleWait = 4,
    // Waiting for memory eviction, as offboarding yields to it.
    kEvictionWait = 5,
    kCommit = 6,
};

// What one ProcessClientOffboardingJob call asks of the worker.
enum class ClientOffboardingDisposition {
    // The job converged.
    kComplete,
    // The job made progress and may run again at once.
    kContinue,
    // Memory eviction holds priority; recheck after a short slice or a wakeup.
    kPressureWait,
    // A segment prepare or an affiliated PROCESSING replica is still settling.
    kLifecycleWait,
    // A real error: keep the failed position and back off.
    kError,
};

// Why the worker was last asked to wait, for the logs.
enum class ClientOffboardingWaitReason {
    kNone,
    kPressure,
    kNoProgressMaintenance,
    kLifecycle,
};

// Process-local residual work for one terminal Client incarnation. The job is
// intentionally not serializable: snapshots stay behind the pending-work
// barrier until the residual work converges. It names the incarnation only:
// every attempt enumerates the segments the client still owns, which is stable
// because an OFFLINE session rejects registration until the job removes it.
struct ClientOffboardingJob {
    UUID client_id;
    ClientSessionSharedPtr retired_session;
    std::vector<PendingSegmentOffboarding> pending_prepare_segments;
    std::vector<PreparedSegmentOffboarding> prepared_segments;
    bool metadata_cleanup_accepted{false};
    bool local_ssd_unregistered{false};
    uint64_t retry_count{0};
    std::chrono::steady_clock::time_point next_attempt_at{
        std::chrono::steady_clock::now()};
    std::chrono::steady_clock::time_point enqueued_at{
        std::chrono::steady_clock::now()};
    // Requeued only to wait for memory eviction: the round-end notification
    // makes it due at once instead of at its next wait slice. An error
    // backoff or a plain continue leaves it false, so a notification never
    // shortens a real backoff.
    bool wake_on_pressure_notification{false};

    // --- Resumable metadata sweep -------------------------------------------
    //
    // Once every segment is prepared, the job snapshots the tenant ids and
    // walks them one stripe at a time. It keeps only ObjectRefs and counters
    // across a wait: no iterator, namespace pointer or lock outlives a unit.
    bool cleanup_snapshot_taken{false};
    std::vector<TenantId> cleanup_tenants;
    size_t cleanup_tenant_index{0};
    size_t cleanup_stripe_index{0};
    std::vector<route::ObjectRef> cleanup_candidates;
    size_t cleanup_candidate_index{0};
    // The tenant the candidates were collected from. The cursor advances as
    // collection finishes a stripe, so the cleaning batch must not re-derive
    // the tenant from the cursor: the last stripe of a tenant may already have
    // moved it to the next one, or past the end.
    TenantId cleanup_candidate_tenant;
    // The restore lifecycle generation the sweep's cursors and snapshot belong
    // to. A restore that crosses a batch discards them.
    uint64_t sweep_lifecycle_epoch{0};
    // Whether the sweep open now met an affiliated PROCESSING replica. A
    // complete sweep that saw one reopens instead of declaring the metadata
    // clean.
    bool sweep_saw_processing{false};
    // The barrier sequence this sweep started at: everything published later
    // keeps its own responsibility for the next sweep.
    uint64_t sweep_barrier_seq{0};
    bool sweep_barrier_started{false};
    // The eviction stall episode whose one maintenance batch was already
    // spent, so a stalled eviction cannot be bypassed more than once per
    // episode.
    uint64_t consumed_eviction_stall_episode{0};
    // When the next bounded maintenance unit may run while eviction has
    // nothing it can take.
    std::chrono::steady_clock::time_point next_maintenance_at{};

    // --- Accounting and progress --------------------------------------------
    uint64_t attempt_id{0};
    ClientOffboardingStage stage{ClientOffboardingStage::kIdle};
    ClientOffboardingWaitReason wait_reason{ClientOffboardingWaitReason::kNone};
    bool waiting{false};
    std::chrono::steady_clock::time_point wait_started{};
    std::chrono::steady_clock::time_point last_summary_at{};
    uint64_t collect_units{0};
    uint64_t collect_visited{0};
    uint64_t collect_candidates{0};
    uint64_t collect_deferred{0};
    uint64_t collect_deferred_ns{0};
    uint64_t collect_max_unit_ns{0};
    uint64_t clean_units{0};
    uint64_t clean_attempted{0};
    uint64_t clean_skipped{0};
    uint64_t clean_requests{0};
    uint64_t clean_removed_replicas{0};
    uint64_t clean_ha_enqueued{0};
    uint64_t clean_ha_failed{0};
    uint64_t wait_count{0};
    uint64_t wait_ms{0};
    uint64_t collect_ms{0};
    uint64_t clean_ms{0};
    uint64_t prepare_ms{0};
    uint64_t commit_ms{0};
};

// Runs offboarding jobs on its own thread and retries, with backoff, the ones
// that have not converged. What a job does is the processor's business; the
// snapshot barrier is not tracked here either: it is the OFFLINE session
// itself, which stays registered until its job removes it.
class ClientOffboardingWorker {
   public:
    // Returns whether the job converged; false reschedules it. The processor
    // runs on the worker thread and may update the job between attempts. Its
    // owner must outlive Stop.
    using Processor =
        std::function<ClientOffboardingDisposition(ClientOffboardingJob&)>;

    explicit ClientOffboardingWorker(Processor process)
        : process_(std::move(process)) {}
    ~ClientOffboardingWorker();

    ClientOffboardingWorker(const ClientOffboardingWorker&) = delete;
    ClientOffboardingWorker& operator=(const ClientOffboardingWorker&) = delete;

    // Stop drops the jobs that are still queued or backing off.
    void Start();
    void Stop();
    // Requires a started worker.
    void Schedule(ClientOffboardingJob job);
    // Wakes the worker so a job waiting for memory eviction rechecks its
    // priority at once instead of at the end of its wait slice.
    void NotifyPending();
    [[nodiscard]] bool HasPending() const {
        return pending_jobs_.load(std::memory_order_acquire) != 0;
    }

    // How long a job waits before rechecking a condition that may clear on its
    // own. A wait is not a failure and does not count as a retry.
    static constexpr uint64_t kWaitSliceMs = 100;

   private:
    friend class test::MasterServiceTest;

    void ThreadFunc();
    void CompleteJob(const ClientOffboardingJob& job);
    void DropJob(const ClientOffboardingJob& job, const char* reason);
    static std::chrono::seconds RetryDelay(uint64_t retry_count);
    static bool ShouldAlert(uint64_t retry_count) {
        return retry_count >= kAlertRetryThreshold;
    }

    static constexpr uint64_t kAlertRetryThreshold = 10;

    const Processor process_;
    std::thread thread_;
    bool running_{false};
    std::deque<ClientOffboardingJob> jobs_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    // Set by NotifyPending, consumed by the worker: jobs waiting for memory
    // eviction are due at once, not at their next slice. Guarded by mutex_.
    bool pressure_notification_pending_{false};
    std::atomic<size_t> pending_jobs_{0};
};

}  // namespace mooncake

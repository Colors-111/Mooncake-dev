#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "client_liveness.h"
#include "metadata/object_route.h"
#include "types.h"

namespace mooncake {

class MasterService;
namespace test {
class MasterServiceTest;
}

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
};

// Result of one offboarding attempt.
enum class ClientOffboardingOutcome {
    kDone,      // the job finished
    kContinue,  // a bounded slice finished with work left; requeue at once
    kWait,      // nothing to do right now, or a real error; retry with backoff
};

// Process-local residual work for one terminal Client incarnation. The job is
// intentionally not serializable: snapshots stay behind the pending-work
// barrier until the residual work converges.
struct ClientOffboardingJob {
    UUID client_id;
    std::shared_ptr<ClientLivenessRecord> liveness;
    std::vector<PendingSegmentOffboarding> pending_prepare_segments;
    std::vector<PreparedSegmentOffboarding> prepared_segments;
    bool metadata_cleanup_accepted{false};
    // Resumable metadata sweep. The tenant set is captured once, when the walk
    // begins: VisitNamespaces filters empty tenants and iterates an unordered
    // container, so re-enumerating with a saved index could skip a tenant.
    // The current stripe's candidates and how far they were cleaned are kept
    // too, so a slice can end inside a stripe rather than only at its end.
    std::vector<TenantId> sweep_tenants;
    bool sweep_tenants_captured{false};
    size_t sweep_tenant_index{0};
    size_t sweep_stripe{0};
    std::vector<route::ObjectRef> sweep_candidates;
    size_t sweep_candidate_offset{0};
    bool unfinished_processing_seen{false};
    bool local_ssd_unregistered{false};
    uint64_t retry_count{0};
    std::chrono::steady_clock::time_point next_attempt_at{
        std::chrono::steady_clock::now()};
    std::chrono::steady_clock::time_point enqueued_at{
        std::chrono::steady_clock::now()};
};

class ClientOffboardingWorker {
   public:
    explicit ClientOffboardingWorker(MasterService* service)
        : service_(service) {}
    ~ClientOffboardingWorker();

    ClientOffboardingWorker(const ClientOffboardingWorker&) = delete;
    ClientOffboardingWorker& operator=(const ClientOffboardingWorker&) = delete;

    void Start();
    void Stop();
    [[nodiscard]] bool HasPending() const {
        return pending_jobs_.load(std::memory_order_acquire) != 0;
    }

   private:
    friend class MasterService;
    friend class test::MasterServiceTest;

    void ReserveJob();
    void ScheduleReserved(ClientOffboardingJob job);
    void ThreadFunc();
    void CompleteJob(const ClientOffboardingJob& job);
    void DropJob(const ClientOffboardingJob& job, const char* reason);
    static std::chrono::seconds RetryDelay(uint64_t retry_count);
    static bool ShouldAlert(uint64_t retry_count) {
        return retry_count >= kAlertRetryThreshold;
    }

    static constexpr uint64_t kAlertRetryThreshold = 10;

    MasterService* service_;
    std::thread thread_;
    bool running_{false};
    std::deque<ClientOffboardingJob> jobs_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<size_t> pending_jobs_{0};
};

}  // namespace mooncake

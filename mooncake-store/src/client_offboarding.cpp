#include "client_offboarding.h"

#include <algorithm>
#include <iterator>
#include <utility>

#include <glog/logging.h>

#include "master_metric_manager.h"
#include "master_service.h"

namespace mooncake {

ClientOffboardingWorker::~ClientOffboardingWorker() { Stop(); }

void ClientOffboardingWorker::Start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_) {
        return;
    }
    running_ = true;
    try {
        thread_ = std::thread(&ClientOffboardingWorker::ThreadFunc, this);
    } catch (...) {
        running_ = false;
        throw;
    }
}

void ClientOffboardingWorker::Stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_ && !thread_.joinable()) {
            return;
        }
        running_ = false;
    }
    cv_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void ClientOffboardingWorker::NotifyPending() {
    // Take the mutex so a wakeup cannot be lost between the worker's
    // predicate check and its wait, and record that a job waiting for memory
    // eviction may recheck at once.
    std::lock_guard<std::mutex> lock(mutex_);
    pressure_notification_pending_ = true;
    cv_.notify_all();
}

void ClientOffboardingWorker::ReserveJob() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        CHECK(running_);
        pending_jobs_.fetch_add(1, std::memory_order_release);
        MasterMetricManager::instance().inc_client_offboarding_queue_depth();
    }
}

void ClientOffboardingWorker::ScheduleReserved(ClientOffboardingJob job) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        CHECK(running_);
        jobs_.push_back(std::move(job));
    }
    cv_.notify_one();
}

std::chrono::seconds ClientOffboardingWorker::RetryDelay(uint64_t retry_count) {
    constexpr uint64_t kBackoffs[] = {1, 2, 4, 8, 16, 30};
    const size_t index = static_cast<size_t>(
        std::min<uint64_t>(retry_count - 1, std::size(kBackoffs) - 1));
    return std::chrono::seconds(kBackoffs[index]);
}

void ClientOffboardingWorker::CompleteJob(const ClientOffboardingJob& job) {
    pending_jobs_.fetch_sub(1, std::memory_order_acq_rel);
    MasterMetricManager::instance().dec_client_offboarding_queue_depth();
    const auto duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - job.enqueued_at)
            .count();
    MasterMetricManager::instance().observe_client_offboarding_duration_ms(
        duration_ms);
    LOG(INFO) << "client_id=" << job.client_id
              << ", action=client_offboarding_complete"
              << ", retries=" << job.retry_count
              << ", duration_ms=" << duration_ms;
}

void ClientOffboardingWorker::DropJob(ClientOffboardingJob& job,
                                      const char* reason) {
    // A dropped job keeps no window open: the sweep never converges, so the
    // release must happen here too. It is idempotent with the one at
    // convergence.
    service_->ReleaseOffboardingCleanup(job);
    pending_jobs_.fetch_sub(1, std::memory_order_acq_rel);
    MasterMetricManager::instance().dec_client_offboarding_queue_depth();
    LOG(ERROR) << "client_id=" << job.client_id
               << ", action=client_offboarding_dropped"
               << ", reason=" << reason << ", retry_count=" << job.retry_count;
}

void ClientOffboardingWorker::ThreadFunc() {
    LOG(INFO) << "Client offboarding worker started";
    while (true) {
        ClientOffboardingJob job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            while (running_ && jobs_.empty()) {
                cv_.wait(lock);
            }
            if (!running_) {
                auto dropped = std::move(jobs_);
                jobs_.clear();
                lock.unlock();
                for (auto& queued : dropped) {
                    DropJob(queued, "master_stopping");
                }
                break;
            }

            // A memory eviction round just ended: a job that was waiting for
            // it rechecks now instead of finishing its 100 ms slice. Jobs in
            // an error backoff keep their time.
            if (pressure_notification_pending_) {
                pressure_notification_pending_ = false;
                const auto now = std::chrono::steady_clock::now();
                for (ClientOffboardingJob& queued : jobs_) {
                    if (queued.wake_on_pressure_notification) {
                        queued.next_attempt_at = now;
                    }
                }
            }

            const auto next = std::min_element(
                jobs_.begin(), jobs_.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.next_attempt_at < rhs.next_attempt_at;
                });
            const auto now = std::chrono::steady_clock::now();
            if (next->next_attempt_at > now) {
                cv_.wait_until(lock, next->next_attempt_at);
                continue;
            }
            job = std::move(*next);
            jobs_.erase(next);
        }

        const auto disposition = service_->ProcessClientOffboardingJob(job);
        if (disposition == ClientOffboardingDisposition::kComplete) {
            CompleteJob(job);
            continue;
        }

        // A wait is not an error: the job keeps its position and is requeued
        // after a short slice, or earlier when eviction ends and wakes it. An
        // error keeps the failed position too, but backs off.
        bool requeued = false;
        if (disposition == ClientOffboardingDisposition::kError) {
            ++job.retry_count;
            MasterMetricManager::instance().inc_client_offboarding_retry();
            const bool alert = ShouldAlert(job.retry_count);
            if (alert) {
                MasterMetricManager::instance().inc_client_offboarding_alert();
            }
            LOG(ERROR) << "client_id=" << job.client_id
                       << ", action=client_offboarding_retry"
                       << ", retry_count=" << job.retry_count
                       << ", alert=" << (alert ? "true" : "false")
                       << ", pending_prepare_segments="
                       << job.pending_prepare_segments.size()
                       << ", prepared_segments=" << job.prepared_segments.size()
                       << ", metadata_cleanup_accepted="
                       << (job.metadata_cleanup_accepted ? "true" : "false")
                       << ", cleanup_tenant_index=" << job.cleanup_tenant_index
                       << ", cleanup_stripe_index=" << job.cleanup_stripe_index
                       << ", cleanup_candidate_offset="
                       << job.cleanup_candidate_index;
            std::lock_guard<std::mutex> lock(mutex_);
            if (running_) {
                // A backoff keeps its full time: no wakeup may shorten it.
                job.wake_on_pressure_notification = false;
                job.next_attempt_at = std::chrono::steady_clock::now() +
                                      RetryDelay(job.retry_count);
                jobs_.push_back(std::move(job));
                requeued = true;
            }
        } else {
            const auto delay =
                disposition == ClientOffboardingDisposition::kContinue
                    ? std::chrono::milliseconds(0)
                    : std::chrono::milliseconds(kWaitSliceMs);
            std::lock_guard<std::mutex> lock(mutex_);
            if (running_) {
                // To the tail: a job that continues does not hold the worker.
                // Only a pressure wait makes a round-end notification useful.
                job.wake_on_pressure_notification =
                    disposition == ClientOffboardingDisposition::kPressureWait;
                job.next_attempt_at = std::chrono::steady_clock::now() + delay;
                jobs_.push_back(std::move(job));
                requeued = true;
            }
        }
        if (requeued) {
            cv_.notify_one();
        } else {
            DropJob(job, "master_stopping_after_attempt");
        }
    }
    LOG(INFO) << "Client offboarding worker stopped";
}

}  // namespace mooncake

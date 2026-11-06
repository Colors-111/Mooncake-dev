#pragma once

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace mooncake {

// Runs a callback on a dedicated thread when scheduled. Multiple Schedule()
// calls made before the worker consumes the pending request are coalesced into
// one callback invocation. Start() and Stop() must be called serially by the
// owner; Schedule() may be called concurrently from other threads.
class BackgroundWorker {
   public:
    using Callback = std::function<void()>;

    explicit BackgroundWorker(Callback callback)
        : callback_(std::move(callback)) {}

    ~BackgroundWorker() { Stop(); }

    BackgroundWorker(const BackgroundWorker&) = delete;
    BackgroundWorker& operator=(const BackgroundWorker&) = delete;

    void Start() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (running_) {
            return;
        }
        running_ = true;
        requested_ = false;
        thread_ = std::thread(&BackgroundWorker::ThreadFunc, this);
    }

    void Stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!running_ && !thread_.joinable()) {
                return;
            }
            running_ = false;
            requested_ = false;
        }
        cv_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void Schedule() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!running_) {
                return;
            }
            // An immediate request wins over a pending delay.
            requested_ = true;
            deadline_.reset();
        }
        cv_.notify_one();
    }

    // Runs the callback no earlier than `delay` from now. Coalesced with
    // Schedule() and other ScheduleAfter() calls: the earliest deadline wins,
    // and an immediate Schedule() overrides it.
    void ScheduleAfter(std::chrono::milliseconds delay) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!running_) {
                return;
            }
            const auto when = std::chrono::steady_clock::now() + delay;
            if (!requested_ || !deadline_ || when < *deadline_) {
                requested_ = true;
                deadline_ = when;
            }
        }
        cv_.notify_one();
    }

   private:
    void ThreadFunc() {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                for (;;) {
                    if (!running_) {
                        return;
                    }
                    if (!requested_) {
                        cv_.wait(lock);
                        continue;
                    }
                    if (!deadline_ ||
                        std::chrono::steady_clock::now() >= *deadline_) {
                        break;
                    }
                    cv_.wait_until(lock, *deadline_);
                }
                requested_ = false;
                deadline_.reset();
            }
            callback_();
        }
    }

    Callback callback_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    bool running_{false};
    bool requested_{false};
    std::optional<std::chrono::steady_clock::time_point> deadline_;
};

}  // namespace mooncake

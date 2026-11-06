#include "background_worker.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

#include <gtest/gtest.h>

namespace mooncake::test {

TEST(BackgroundWorkerTest, CoalescesPendingSchedules) {
    std::mutex mutex;
    std::condition_variable cv;
    size_t callback_count = 0;
    bool release_first_callback = false;

    BackgroundWorker worker([&] {
        std::unique_lock<std::mutex> lock(mutex);
        ++callback_count;
        cv.notify_all();
        if (callback_count == 1) {
            cv.wait(lock, [&] { return release_first_callback; });
        }
    });
    worker.Start();
    worker.Schedule();

    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(1),
                                [&] { return callback_count == 1; }));
    }
    for (size_t i = 0; i < 10; ++i) {
        worker.Schedule();
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        release_first_callback = true;
    }
    cv.notify_all();

    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(1),
                                [&] { return callback_count == 2; }));
    }
    worker.Stop();
    EXPECT_EQ(callback_count, 2);
}

TEST(BackgroundWorkerTest, ScheduleIsIgnoredWhileStopped) {
    std::mutex mutex;
    size_t callback_count = 0;
    BackgroundWorker worker([&] {
        std::lock_guard<std::mutex> lock(mutex);
        ++callback_count;
    });

    worker.Schedule();
    worker.Start();
    worker.Stop();
    worker.Schedule();

    std::lock_guard<std::mutex> lock(mutex);
    EXPECT_EQ(callback_count, 0);
}

TEST(BackgroundWorkerTest, ScheduleAfterWaitsForTheDelay) {
    std::atomic<int> calls{0};
    BackgroundWorker worker([&] { ++calls; });
    worker.Start();
    worker.ScheduleAfter(std::chrono::milliseconds(200));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(calls.load(), 0);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (calls.load() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(calls.load(), 1);
    worker.Stop();
}

TEST(BackgroundWorkerTest, ImmediateScheduleWinsOverAPendingDelay) {
    std::atomic<int> calls{0};
    BackgroundWorker worker([&] { ++calls; });
    worker.Start();
    worker.ScheduleAfter(std::chrono::seconds(5));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    // An immediate request must not wait behind the delayed one.
    worker.Schedule();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (calls.load() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(calls.load(), 1);
    worker.Stop();
}

}  // namespace mooncake::test

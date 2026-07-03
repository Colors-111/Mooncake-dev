#include "storage_backend.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "allocator.h"
#include "types.h"
#include "utils.h"

namespace fs = std::filesystem;
namespace mooncake::test {

class GuaranteedEvictionTest : public ::testing::Test {
   protected:
    void SetUp() override {
        google::InitGoogleLogging("GuaranteedEvictionTest");
        FLAGS_logtostderr = true;
        data_path_ = "/tmp/guaranteed_eviction_test_" +
                     std::to_string(getpid());
        fs::remove_all(data_path_);
        fs::create_directories(data_path_);
    }

    void TearDown() override {
        google::ShutdownGoogleLogging();
        std::error_code ec;
        fs::remove_all(data_path_, ec);
    }

    std::string data_path_;
};

// A complete_handler that just accepts the write (records nothing). Free
// function (not a fixture member) so OffloadBatch can call it.
ErrorCode NoopComplete(const std::vector<std::string>&,
                      std::vector<StorageObjectMetadata>&) {
    return ErrorCode::OK;
}

// Task 1: BucketMetadata defaults to non-guaranteed.
TEST_F(GuaranteedEvictionTest, BucketMetadataDefaultsNonGuaranteed) {
    BucketMetadata meta;
    EXPECT_FALSE(meta.guaranteed);
    meta.guaranteed = true;
    EXPECT_TRUE(meta.guaranteed);
}

// Helper: offload a batch and return the bucket_id.
tl::expected<int64_t, ErrorCode> OffloadBatch(
    BucketStorageBackend& backend,
    const std::vector<std::pair<std::string, std::string>>& kv,
    bool guaranteed) {
    std::unordered_map<std::string, std::vector<Slice>> batch;
    // buffers must outlive the call; leak is acceptable in a short test, but
    // better to keep them alive until the test ends — return them via a static
    // pool to avoid use-after-free. Simpler: allocate with new[] and don't free
    // (test process exits shortly).
    static std::vector<std::unique_ptr<char[]>> pool;
    for (const auto& [key, value] : kv) {
        auto buf = std::make_unique<char[]>(value.size());
        std::memcpy(buf.get(), value.data(), value.size());
        batch.emplace(key, std::vector<Slice>{Slice{buf.get(), value.size()}});
        pool.push_back(std::move(buf));
    }
    return backend.BatchOffload(batch, NoopComplete,
                                /*eviction_handler=*/nullptr, guaranteed);
}

// Task 4A (FIFO): a guaranteed bucket is NOT evicted even under quota pressure,
// while a normal bucket IS evicted to make room.
TEST_F(GuaranteedEvictionTest, FifoSkipsGuaranteedBucket) {
    FileStorageConfig config;
    config.storage_filepath = data_path_;
    BucketBackendConfig bucket_config;
    bucket_config.eviction_policy = BucketEvictionPolicy::FIFO;
    bucket_config.max_total_size = 1024;  // tiny cap => eviction on 2nd offload
    BucketStorageBackend backend(config, bucket_config);
    ASSERT_TRUE(backend.Init());

    // First offload: a guaranteed bucket (well under cap).
    ASSERT_TRUE(OffloadBatch(backend, {{"g1", std::string(256, 'g')}}, true));
    // Second offload: a normal bucket whose size exceeds the cap => must evict.
    // The guaranteed bucket must NOT be selected; the normal write must either
    // succeed (evicting nothing because only the guaranteed bucket exists, and
    // it's protected) or fail with ENOSPC — but the guaranteed bucket survives.
    auto res = OffloadBatch(backend, {{"n1", std::string(768, 'n')}}, false);

    // The guaranteed key must still be present (not evicted).
    auto g_exist = backend.IsExist("g1");
    ASSERT_TRUE(g_exist);
    EXPECT_TRUE(g_exist.value())
        << "guaranteed bucket must survive eviction pressure";
}

// Task 4B (FIFO): a normal bucket IS evicted when pressure comes from another
// normal offload (regression: normal eviction still works).
TEST_F(GuaranteedEvictionTest, FifoEvictsNormalBucket) {
    FileStorageConfig config;
    config.storage_filepath = data_path_;
    BucketBackendConfig bucket_config;
    bucket_config.eviction_policy = BucketEvictionPolicy::FIFO;
    bucket_config.max_total_size = 1024;
    BucketStorageBackend backend(config, bucket_config);
    ASSERT_TRUE(backend.Init());

    ASSERT_TRUE(OffloadBatch(backend, {{"n1", std::string(256, 'a')}}, false));
    // Force eviction of n1 by writing a second normal batch over the cap.
    ASSERT_TRUE(OffloadBatch(backend, {{"n2", std::string(768, 'b')}}, false));

    auto n1_exist = backend.IsExist("n1");
    ASSERT_TRUE(n1_exist);
    EXPECT_FALSE(n1_exist.value())
        << "normal (non-guaranteed) bucket must be evicted under pressure";
}

// Task 4C (LRU): a guaranteed bucket is NOT evicted under LRU pressure.
TEST_F(GuaranteedEvictionTest, LruSkipsGuaranteedBucket) {
    FileStorageConfig config;
    config.storage_filepath = data_path_;
    BucketBackendConfig bucket_config;
    bucket_config.eviction_policy = BucketEvictionPolicy::LRU;
    bucket_config.max_total_size = 1024;
    BucketStorageBackend backend(config, bucket_config);
    ASSERT_TRUE(backend.Init());

    ASSERT_TRUE(OffloadBatch(backend, {{"g1", std::string(256, 'g')}}, true));
    // Touch g1 to update its LRU timestamp (so it's "most recently used"), then
    // write a normal batch that exceeds the cap. Under LRU the oldest normal
    // bucket should be evicted, not g1.
    std::unordered_map<std::string, StorageObjectMetadata> meta;
    ASSERT_TRUE(backend.BatchQuery(std::vector<std::string>{"g1"}, meta));
    ASSERT_TRUE(OffloadBatch(backend, {{"n1", std::string(768, 'n')}}, false));

    auto g_exist = backend.IsExist("g1");
    ASSERT_TRUE(g_exist);
    EXPECT_TRUE(g_exist.value()) << "guaranteed bucket must survive LRU eviction";
}

}  // namespace mooncake::test

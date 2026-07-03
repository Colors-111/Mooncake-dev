#include "master_service.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "types.h"

namespace mooncake::test {

class GuaranteedOffloadTest : public ::testing::Test {
   protected:
    void SetUp() override {
        google::InitGoogleLogging("GuaranteedOffloadTest");
        FLAGS_logtostderr = true;
    }

    void TearDown() override { google::ShutdownGoogleLogging(); }

    static constexpr size_t kDefaultSegmentBase = 0x300000000;

    Segment MakeSegment(std::string name, size_t base, size_t size) const {
        Segment segment;
        segment.id = generate_uuid();
        segment.name = std::move(name);
        segment.base = base;
        segment.size = size;
        segment.te_endpoint = segment.name;
        return segment;
    }

    struct MountedSegmentContext {
        UUID segment_id;
        UUID client_id;
    };

    MountedSegmentContext PrepareSegment(MasterService& service,
                                         std::string name, size_t base,
                                         size_t size) const {
        Segment segment = MakeSegment(std::move(name), base, size);
        UUID client_id = generate_uuid();
        auto mount_result = service.MountSegment(segment, client_id);
        EXPECT_TRUE(mount_result.has_value());
        return {.segment_id = segment.id, .client_id = client_id};
    }

    // Put an object and complete it. guaranteed_until_ms>0 marks it guaranteed.
    void PutObject(MasterService& service, const UUID& client_id,
                   const std::string& key, int64_t guaranteed_until_ms = 0,
                   size_t size = 1024) {
        ReplicateConfig config;
        config.replica_num = 1;
        config.guaranteed_until_ms = guaranteed_until_ms;
        auto put_start =
            service.PutStart(client_id, key, "default", size, config);
        ASSERT_TRUE(put_start.has_value()) << "PutStart failed for key=" << key;
        auto put_end =
            service.PutEnd(client_id, key, "default", ReplicaType::MEMORY);
        ASSERT_TRUE(put_end.has_value()) << "PutEnd failed for key=" << key;
    }

    // Drain the offload queue via OffloadObjectHeartbeat.
    std::vector<OffloadTaskItem> DrainOffloadQueue(MasterService& service,
                                                    const UUID& client_id) {
        auto res = service.OffloadObjectHeartbeat(client_id, true);
        if (!res) {
            return {};
        }
        return res.value();
    }

    template <typename Predicate>
    void WaitUntil(Predicate&& predicate,
                   std::chrono::milliseconds timeout = std::chrono::milliseconds(4000),
                   std::chrono::milliseconds interval =
                       std::chrono::milliseconds(50)) const {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (predicate()) {
                return;
            }
            std::this_thread::sleep_for(interval);
        }
        EXPECT_TRUE(predicate());
    }
};

// Task 1: OffloadTaskItem carries a guaranteed flag and compares on it.
TEST_F(GuaranteedOffloadTest, OffloadTaskItemCarriesGuaranteedFlag) {
    OffloadTaskItem normal{.tenant_id = "default", .key = "k", .size = 10,
                           .guaranteed = false};
    OffloadTaskItem guar{.tenant_id = "default", .key = "k", .size = 10,
                         .guaranteed = true};
    EXPECT_EQ(normal, (OffloadTaskItem{.tenant_id = "default", .key = "k",
                                       .size = 10, .guaranteed = false}));
    EXPECT_NE(normal, guar)
        << "Items differing only in guaranteed must not be equal";
}

// Task 2: ReplicateConfig carries guaranteed_until_ms (Phase 1: only >0 is checked).
TEST_F(GuaranteedOffloadTest, ReplicateConfigCarriesGuaranteedUntilMs) {
    ReplicateConfig config;
    EXPECT_EQ(config.guaranteed_until_ms, 0);  // default: no guarantee
    config.guaranteed_until_ms = 60000;
    EXPECT_GT(config.guaranteed_until_ms, 0);
}

// Task 3: enable_guaranteed_cache defaults false and is settable on MasterServiceConfig.
TEST_F(GuaranteedOffloadTest, EnableGuaranteedCacheConfigField) {
    MasterServiceConfig config;
    EXPECT_FALSE(config.enable_guaranteed_cache);  // default off
    config.enable_guaranteed_cache = true;
    EXPECT_TRUE(config.enable_guaranteed_cache);
}

// Task 7A: guaranteed object is enqueued even when the normal offload queue is full.
TEST_F(GuaranteedOffloadTest, GuaranteedExemptFromNormalQueueLimit) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = true;
    config.offloading_queue_limit = 4;  // small limit so we can fill it
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true).has_value());

    // Fill the normal offload queue to the limit (no drain yet).
    for (int i = 0; i < 4; ++i) {
        PutObject(*service, ctx.client_id, "normal_" + std::to_string(i));
    }
    // A guaranteed object must still be enqueued despite the full normal queue.
    PutObject(*service, ctx.client_id, "guar", /*guaranteed_until_ms=*/60000);

    auto drained = DrainOffloadQueue(*service, ctx.client_id);
    bool found_guar = false;
    for (const auto& t : drained) {
        if (t.key == "guar") {
            found_guar = true;
            EXPECT_TRUE(t.guaranteed) << "guaranteed key must carry the flag";
        }
    }
    EXPECT_TRUE(found_guar) << "guaranteed object must be enqueued even when "
                               "the normal queue is full";

    service->RemoveAll();
}

// Task 7B: a normal object past the limit is NOT enqueued (regression guard).
TEST_F(GuaranteedOffloadTest, NormalStillRespectsQueueLimit) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = true;
    config.offloading_queue_limit = 4;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true).has_value());

    for (int i = 0; i < 4; ++i) {
        PutObject(*service, ctx.client_id, "normal_" + std::to_string(i));
    }
    PutObject(*service, ctx.client_id, "overflow");  // 5th normal -> KEYS_ULTRA_LIMIT

    auto drained = DrainOffloadQueue(*service, ctx.client_id);
    bool found_overflow = false;
    for (const auto& t : drained) {
        if (t.key == "overflow") found_overflow = true;
    }
    EXPECT_FALSE(found_overflow)
        << "5th normal object must be rejected (queue full)";

    service->RemoveAll();
}

// Task 7C: under offload_on_evict=true, guaranteed STILL offloads at PutEnd.
TEST_F(GuaranteedOffloadTest, GuaranteedOffloadsAtPutEndEvenWhenOffloadOnEvict) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = true;
    config.offload_on_evict = true;  // normally defers normal offload to eviction
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true).has_value());

    PutObject(*service, ctx.client_id, "guar", /*guaranteed_until_ms=*/60000);
    PutObject(*service, ctx.client_id, "normal");

    auto drained = DrainOffloadQueue(*service, ctx.client_id);
    bool found_guar = false, found_normal = false;
    for (const auto& t : drained) {
        if (t.key == "guar") found_guar = true;
        if (t.key == "normal") found_normal = true;
    }
    EXPECT_TRUE(found_guar)
        << "guaranteed must offload at PutEnd even under offload_on_evict=true";
    EXPECT_FALSE(found_normal)
        << "normal object must NOT PutEnd-offload under offload_on_evict=true";

    service->RemoveAll();
}

// Task 7D: with enable_guaranteed_cache=false, a guaranteed_until_ms>0 Put
// degrades to normal (no guaranteed offload under offload_on_evict=true).
TEST_F(GuaranteedOffloadTest, FlagOffDegradesGuaranteedToNormal) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = false;  // flag off
    config.offload_on_evict = true;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true).has_value());

    PutObject(*service, ctx.client_id, "would_be_guar", /*guaranteed_until_ms=*/60000);

    auto drained = DrainOffloadQueue(*service, ctx.client_id);
    EXPECT_TRUE(drained.empty())
        << "with flag off + offload_on_evict, guaranteed_until_ms>0 must NOT "
           "offload at PutEnd (degrades to normal)";

    service->RemoveAll();
}

// Task 8A: drain returns both guaranteed and normal tasks, each with its flag.
TEST_F(GuaranteedOffloadTest, DrainReturnsBothQueuesWithFlags) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = true;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true).has_value());

    PutObject(*service, ctx.client_id, "normal");
    PutObject(*service, ctx.client_id, "guar", /*guaranteed_until_ms=*/60000);

    auto drained = DrainOffloadQueue(*service, ctx.client_id);
    ASSERT_EQ(drained.size(), 2u);
    int guar_count = 0, normal_count = 0;
    for (const auto& t : drained) {
        if (t.key == "guar") {
            EXPECT_TRUE(t.guaranteed);
            ++guar_count;
        }
        if (t.key == "normal") {
            EXPECT_FALSE(t.guaranteed);
            ++normal_count;
        }
    }
    EXPECT_EQ(guar_count, 1);
    EXPECT_EQ(normal_count, 1);

    service->RemoveAll();
}

// Task 8B: when offloading is disabled, a queued guaranteed task is cleaned up
// (refcnt decremented, task erased) — the next enable-drain returns empty.
TEST_F(GuaranteedOffloadTest, DisableCleansGuaranteedQueue) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = true;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true).has_value());

    PutObject(*service, ctx.client_id, "guar", /*guaranteed_until_ms=*/60000);
    // Do NOT drain — leave it queued, then disable (clears + refcnt-decs).
    auto disable_res = service->OffloadObjectHeartbeat(ctx.client_id, false);
    ASSERT_TRUE(disable_res.has_value());
    EXPECT_TRUE(disable_res.value().empty());  // disabled branch returns {}

    // Re-enable: the guaranteed task was cleaned, not re-enqueued.
    auto drained = DrainOffloadQueue(*service, ctx.client_id);
    EXPECT_TRUE(drained.empty())
        << "disabled cleanup must clear the guaranteed queue";

    service->RemoveAll();
}

// Task 9: on SSD-write NACK, a guaranteed object is re-enqueued (retry).
TEST_F(GuaranteedOffloadTest, GuaranteedReenqueuedOnNack) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = true;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true).has_value());

    PutObject(*service, ctx.client_id, "guar", /*guaranteed_until_ms=*/60000);

    // Drain: master hands the task to the client (simulated).
    auto first_drain = DrainOffloadQueue(*service, ctx.client_id);
    ASSERT_EQ(first_drain.size(), 1u);
    ASSERT_EQ(first_drain[0].key, "guar");

    // Simulate an SSD-write failure: client NACKs via NotifyOffloadSuccess.
    std::vector<OffloadTaskItem> tasks = first_drain;
    std::vector<StorageObjectMetadata> metas(tasks.size());
    metas[0].data_size = -1;  // NACK sentinel
    auto nack_res = service->NotifyOffloadSuccess(ctx.client_id, tasks, metas);
    ASSERT_TRUE(nack_res.has_value());

    // The guaranteed object must have been re-enqueued for retry.
    auto second_drain = DrainOffloadQueue(*service, ctx.client_id);
    bool reenqueued = false;
    for (const auto& t : second_drain) {
        if (t.key == "guar") reenqueued = true;
    }
    EXPECT_TRUE(reenqueued)
        << "guaranteed object must be re-enqueued on NACK for retry";

    service->RemoveAll();
}

// Task 5: after SSD write success, a guaranteed object's pin is released and a
// LOCAL_DISK replica is recorded — it has transitioned from "guaranteed, pinned,
// in-queue" to "offloaded, LOCAL_DISK present, no in-flight task". (Spec §10
// case 5: guaranteed becomes normally evictable after SSD success.)
TEST_F(GuaranteedOffloadTest, GuaranteedBecomesEvictableAfterSsdSuccess) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = true;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true).has_value());

    PutObject(*service, ctx.client_id, "guar", /*guaranteed_until_ms=*/60000);

    // Drain the guaranteed queue: master hands the task to the client.
    auto drained = DrainOffloadQueue(*service, ctx.client_id);
    ASSERT_EQ(drained.size(), 1u);
    ASSERT_EQ(drained[0].key, "guar");

    // Simulate a successful SSD write: client reports back via
    // NotifyOffloadSuccess with a non-negative data_size + transport_endpoint.
    std::vector<OffloadTaskItem> tasks = drained;
    std::vector<StorageObjectMetadata> metas(tasks.size());
    metas[0].data_size = 1024;  // success (non-negative)
    metas[0].transport_endpoint = "tcp://fake_endpoint:1234";
    auto ack_res = service->NotifyOffloadSuccess(ctx.client_id, tasks, metas);
    ASSERT_TRUE(ack_res.has_value()) << "NotifyOffloadSuccess should succeed";

    // After success: a LOCAL_DISK replica must be recorded (offload completed).
    auto replica_list = service->GetReplicaList("guar", "default");
    ASSERT_TRUE(replica_list.has_value()) << "GetReplicaList should succeed";
    bool has_local_disk = std::any_of(
        replica_list->replicas.begin(), replica_list->replicas.end(),
        [](const Replica::Descriptor& d) { return d.is_local_disk_replica(); });
    EXPECT_TRUE(has_local_disk)
        << "after SSD success, the object must have a LOCAL_DISK replica";

    // The in-flight offloading task must be cleared (no longer pinned/queued).
    auto second_drain = DrainOffloadQueue(*service, ctx.client_id);
    EXPECT_TRUE(second_drain.empty())
        << "after success, no offloading task should remain in the queue";

    service->RemoveAll();
}

// Task 9 (enable_offload=false): with offload entirely disabled, a guaranteed
// Put degrades to a normal object — it is NOT enqueued for offload and incurs no
// error. (Distinct from FlagOffDegradesGuaranteedToNormal, which covers the
// enable_guaranteed_cache=false direction.)
TEST_F(GuaranteedOffloadTest, GuaranteedDegradesWhenOffloadDisabled) {
    MasterServiceConfig config;
    config.enable_offload = false;  // offload entirely off
    config.enable_guaranteed_cache = true;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(service->MountLocalDiskSegment(ctx.client_id, true).has_value());

    // A guaranteed_until_ms>0 Put must succeed (no error) even though offload is
    // disabled — it just degrades to a normal in-memory object.
    PutObject(*service, ctx.client_id, "guar", /*guaranteed_until_ms=*/60000);

    // Nothing should be queued for offload (enable_offload_ gates PutEnd enqueue).
    auto drained = DrainOffloadQueue(*service, ctx.client_id);
    EXPECT_TRUE(drained.empty())
        << "with enable_offload=false, a guaranteed Put must NOT offload";

    service->RemoveAll();
}

}  // namespace mooncake::test

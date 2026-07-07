#include "master_service.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "types.h"

namespace mooncake::test {

class GuaranteedDowngradeTest : public ::testing::Test {
   protected:
    void SetUp() override {
        google::InitGoogleLogging("GuaranteedDowngradeTest");
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

    // Construct a MasterService with enable_guaranteed_cache=true and the
    // given master-level guaranteed_until_ms.
    std::unique_ptr<MasterService> MakeMaster(
        int64_t guaranteed_until_ms = 60000) {
        MasterServiceConfig config;
        config.enable_offload = true;
        config.enable_guaranteed_cache = true;
        config.guaranteed_until_ms = guaranteed_until_ms;
        config.default_kv_lease_ttl = 2000;
        return std::make_unique<MasterService>(config);
    }
};

// Task 1: PutEnd with guaranteed_until_ms sets guaranteed_until_ ~ now+TTL.
TEST_F(GuaranteedDowngradeTest, GuaranteedUntilSetOnPutEnd) {
    auto master = MakeMaster(/*guaranteed_until_ms=*/60000);
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*master, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(master->MountLocalDiskSegment(ctx.client_id, true).has_value());

    auto t0 = std::chrono::system_clock::now();
    PutObject(*master, ctx.client_id, "k1", /*guaranteed_until_ms=*/60000);
    auto t1 = std::chrono::system_clock::now();

    // guaranteed_until_ ~ t0 + 60s (tolerance covers t0..t1 ctor overhead).
    auto gu = master->GetGuaranteedUntilForTesting("k1", "default");
    ASSERT_TRUE(gu.has_value());
    EXPECT_GT(*gu, t0 + std::chrono::seconds(59));
    EXPECT_LT(*gu, t1 + std::chrono::seconds(61));

    master->RemoveAll();
}

// Task 1: non-guaranteed Put (guaranteed_until_ms=0) leaves guaranteed_until_
// = epoch.
TEST_F(GuaranteedDowngradeTest, NonGuaranteedPutLeavesEpoch) {
    auto master = MakeMaster();
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*master, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(master->MountLocalDiskSegment(ctx.client_id, true).has_value());

    PutObject(*master, ctx.client_id, "normal");  // guaranteed_until_ms=0
    auto gu = master->GetGuaranteedUntilForTesting("normal", "default");
    ASSERT_TRUE(gu.has_value());
    EXPECT_EQ(*gu, std::chrono::system_clock::time_point{});

    master->RemoveAll();
}

// Task 1: enable_guaranteed_cache=false -> guaranteed_until_ stays epoch even
// with TTL>0 (flag gates the per-request TTL).
TEST_F(GuaranteedDowngradeTest, FlagOffLeavesEpochEvenWithTtl) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_guaranteed_cache = false;  // flag off
    config.guaranteed_until_ms = 60000;
    config.default_kv_lease_ttl = 2000;
    MasterService master(config);
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(master, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(master.MountLocalDiskSegment(ctx.client_id, true).has_value());

    PutObject(master, ctx.client_id, "k1", /*guaranteed_until_ms=*/60000);
    auto gu = master.GetGuaranteedUntilForTesting("k1", "default");
    ASSERT_TRUE(gu.has_value());
    EXPECT_EQ(*gu, std::chrono::system_clock::time_point{});

    master.RemoveAll();
}

// Task 1: key absent -> nullopt.
TEST_F(GuaranteedDowngradeTest, MissingKeyReturnsNullopt) {
    auto master = MakeMaster();
    auto gu = master->GetGuaranteedUntilForTesting("nope", "default");
    EXPECT_FALSE(gu.has_value());
}

// Task 6: request-level renewal. Only GetReplicaList calls that explicitly pass
// renew_guaranteed_ttl_ms > 0 renew guaranteed_until_; plain calls leave it
// unchanged.
TEST_F(GuaranteedDowngradeTest, RenewOnlyWhenRequestParamSet) {
    auto master = MakeMaster(/*guaranteed_until_ms=*/10000);  // 10s TTL
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*master, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(master->MountLocalDiskSegment(ctx.client_id, true).has_value());
    PutObject(*master, ctx.client_id, "k1", /*guaranteed_until_ms=*/10000);
    auto before = master->GetGuaranteedUntilForTesting("k1", "default").value();

    // No renew_guaranteed_ttl_ms -> no renewal.
    auto r1 = master->GetReplicaList("k1", "default");
    ASSERT_TRUE(r1.has_value());
    auto after_plain = master->GetGuaranteedUntilForTesting("k1", "default").value();
    EXPECT_EQ(after_plain, before);  // unchanged

    // renew_guaranteed_ttl_ms=30000 -> push to now+30s, never shrink.
    auto r2 = master->GetReplicaList("k1", "default",
                                     /*renew_guaranteed_ttl_ms=*/30000);
    ASSERT_TRUE(r2.has_value());
    auto after_renew = master->GetGuaranteedUntilForTesting("k1", "default").value();
    EXPECT_GE(after_renew, before);  // never shrink
    auto now = std::chrono::system_clock::now();
    EXPECT_GT(after_renew, now + std::chrono::seconds(25));  // ~now+30s

    master->RemoveAll();
}

// Task 6: renewal is a no-op on non-guaranteed (epoch) objects ("only renew,
// never create").
TEST_F(GuaranteedDowngradeTest, RenewNoOpOnNonGuaranteed) {
    auto master = MakeMaster();
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*master, "seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(master->MountLocalDiskSegment(ctx.client_id, true).has_value());
    PutObject(*master, ctx.client_id, "normal");  // non-guaranteed (epoch)
    auto r = master->GetReplicaList("normal", "default",
                                    /*renew_guaranteed_ttl_ms=*/30000);
    ASSERT_TRUE(r.has_value());
    auto after = master->GetGuaranteedUntilForTesting("normal", "default").value();
    EXPECT_EQ(after, std::chrono::system_clock::time_point{});  // still epoch

    master->RemoveAll();
}

}  // namespace mooncake::test

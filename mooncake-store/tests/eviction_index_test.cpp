#include "metadata/lease_eviction_index.h"

#include <chrono>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "allocator.h"
#include "client_liveness.h"
#include "metadata/namespace.h"
#include "object_test_helpers.h"

namespace mooncake {
namespace {

using Clock = std::chrono::system_clock;
using metadata::EvictionIndex;
using std::chrono::seconds;

// A namespace whose writes the lease eviction index follows, and an allocator
// that hands out the memory replicas its objects hold.
class LeaseEvictionIndexTest : public ::testing::Test {
   protected:
    // Publishes `key` holding one memory replica in `status`, with its lease
    // ending at `deadline`.
    void Publish(const std::string& key, Clock::time_point deadline,
                 ReplicaStatus status = ReplicaStatus::COMPLETE,
                 bool hard_pinned = false) {
        auto guard = ns_.objects.WriteKey(key);
        PublishIn(guard, key, std::string(), status, hard_pinned);
        SetDeadline(guard.metadata(), deadline);
    }

    // Publishes `key` in `group_id`, setting the lease its members share.
    void PublishInGroup(const std::string& key, const std::string& group_id,
                        Clock::time_point deadline) {
        auto guard = ns_.objects.WriteKey(key);
        PublishIn(guard, key, group_id);
        SetDeadline(guard.metadata(), deadline);
    }

    // Publishes under `guard`, which holds `key` and no object.
    void PublishIn(const route::WriteGuard& guard, const std::string& key,
                   const std::string& group_id,
                   ReplicaStatus status = ReplicaStatus::COMPLETE,
                   bool hard_pinned = false) {
        std::vector<Replica> replicas;
        replicas.emplace_back(allocator_->allocate(128), status);
        (void)guard.Publish(UUID{1, 2}, Clock::time_point{}, 128,
                            std::move(replicas), std::nullopt, hard_pinned,
                            ObjectDataType::UNKNOWN, group_id, TenantId(), key);
    }

    // Publishes `key` holding one memory replica whose buffer is bound to
    // `liveness`, as a mounted segment's buffers are.
    void PublishWithLiveness(
        const std::string& key,
        const std::shared_ptr<ClientLivenessRecord>& liveness,
        Clock::time_point deadline) {
        auto guard = ns_.objects.WriteKey(key);
        auto buffer = allocator_->allocate(128);
        buffer->bindClientLiveness(liveness);
        std::vector<Replica> replicas;
        replicas.emplace_back(std::move(buffer), ReplicaStatus::COMPLETE);
        (void)guard.Publish(UUID{1, 2}, Clock::time_point{}, 128,
                            std::move(replicas), std::nullopt,
                            /*hard_pinned=*/false, ObjectDataType::UNKNOWN,
                            std::string(), TenantId(), key);
        SetDeadline(guard.metadata(), deadline);
    }

    // A liveness record that never served: ACTIVE, then SUSPECTED, then
    // OFFLINE. A returning client registers a new record.
    static std::shared_ptr<ClientLivenessRecord> LivenessIn(
        ClientLivenessState state) {
        auto record = std::make_shared<ClientLivenessRecord>(
            ClientLivenessRecord::Clock::now());
        const auto now = ClientLivenessRecord::Clock::now();
        (void)record->Evaluate(now, seconds(0), seconds(0));
        if (state != ClientLivenessState::SUSPECTED) {
            (void)record->Evaluate(now, seconds(0), seconds(0));
        }
        return record;
    }

    static void SetDeadline(const ObjectMetadata& metadata,
                            Clock::time_point deadline) {
        SpinLocker locker(&metadata.lock);
        metadata.lease_->SetDeadline(deadline);
    }

    // An entry a walk handed out, with its keys copied off the cursor.
    struct WalkedEntry {
        std::vector<std::string> keys;
        EvictionIndex::Rank rank;
    };

    // Every entry a walk at `now` hands out, in order.
    std::vector<WalkedEntry> Walk(Clock::time_point now) const {
        auto cursor = ns_.eviction_index().Scan(now);
        std::vector<WalkedEntry> entries;
        while (auto entry = cursor->Next()) {
            entries.push_back(WalkedEntry{
                {entry->keys.begin(), entry->keys.end()}, entry->rank});
        }
        return entries;
    }

    // The keys a walk at `now` hands out, in order, one entry's together.
    std::vector<std::string> DueKeys(Clock::time_point now) const {
        std::vector<std::string> keys;
        for (const auto& entry : Walk(now)) {
            keys.insert(keys.end(), entry.keys.begin(), entry.keys.end());
        }
        return keys;
    }

    // The entries a walk at `now` hands out, each as the set of its keys.
    std::vector<std::set<std::string>> DueEntries(Clock::time_point now) const {
        std::vector<std::set<std::string>> entries;
        for (const auto& entry : Walk(now)) {
            entries.emplace_back(entry.keys.begin(), entry.keys.end());
        }
        return entries;
    }

    size_t Filed() const { return ns_.eviction_index().Scan(t0_)->Total(); }

    const Clock::time_point t0_{seconds(1000)};
    // Declared before the namespace, so it outlives the replicas it gave out.
    std::shared_ptr<OffsetBufferAllocator> allocator_ =
        std::make_shared<OffsetBufferAllocator>("segment", 0x100000000ULL,
                                                64 * 1024 * 1024, "segment");
    metadata::Namespace ns_;
};

TEST_F(LeaseEvictionIndexTest, FilesOnlyObjectsEvictionCanTakeMemoryFrom) {
    Publish("complete", t0_);
    Publish("processing", t0_, ReplicaStatus::PROCESSING);
    Publish("hard_pinned", t0_, ReplicaStatus::COMPLETE, /*hard_pinned=*/true);
    {
        // No replica at all.
        auto guard = ns_.objects.WriteKey("empty");
        (void)test::PublishEnvelope(guard);
    }

    EXPECT_EQ(Filed(), 1u);
    EXPECT_EQ(DueKeys(t0_ + seconds(1)), std::vector<std::string>{"complete"});
}

TEST_F(LeaseEvictionIndexTest, OfflineClientsKeysNeverEnterTheIndex) {
    PublishWithLiveness("offline", LivenessIn(ClientLivenessState::OFFLINE),
                        t0_);
    Publish("healthy", t0_);

    EXPECT_EQ(Filed(), 1u);
    EXPECT_EQ(DueKeys(t0_ + seconds(1)), std::vector<std::string>{"healthy"});
}

TEST_F(LeaseEvictionIndexTest, SuspectedClientsKeysStayFiled) {
    // SUSPECTED may still come back, so its keys remain evictable.
    PublishWithLiveness("suspected", LivenessIn(ClientLivenessState::SUSPECTED),
                        t0_);

    EXPECT_EQ(Filed(), 1u);
    EXPECT_EQ(DueKeys(t0_ + seconds(1)), std::vector<std::string>{"suspected"});
}

TEST_F(LeaseEvictionIndexTest, AKeyKeepsItsPlaceWhileOneMemoryReplicaServes) {
    {
        // The guard's release is what files the key, so it ends before the
        // assertions below.
        auto guard = ns_.objects.WriteKey("mixed");
        auto offline = allocator_->allocate(128);
        offline->bindClientLiveness(LivenessIn(ClientLivenessState::OFFLINE));
        auto healthy = allocator_->allocate(128);
        std::vector<Replica> replicas;
        replicas.emplace_back(std::move(offline), ReplicaStatus::COMPLETE);
        replicas.emplace_back(std::move(healthy), ReplicaStatus::COMPLETE);
        (void)guard.Publish(UUID{1, 2}, Clock::time_point{}, 128,
                            std::move(replicas), std::nullopt,
                            /*hard_pinned=*/false, ObjectDataType::UNKNOWN,
                            std::string(), TenantId(), "mixed");
        SetDeadline(guard.metadata(), t0_);
    }

    EXPECT_EQ(Filed(), 1u);
    EXPECT_EQ(DueKeys(t0_ + seconds(1)), std::vector<std::string>{"mixed"});
}

TEST_F(LeaseEvictionIndexTest, AnUnmountedSegmentsKeyLeavesOnItsFirstVisit) {
    Publish("on_segment", t0_);
    ASSERT_EQ(Filed(), 1u);

    // The segment is unmounted: the allocator the replica's buffer points at
    // is gone, so the replica can never serve again.
    allocator_.reset();

    // Eviction visits the key, takes its write lock and lets it go; the
    // release re-files it, and it no longer qualifies.
    {
        auto guard = ns_.objects.Write("on_segment");
        ASSERT_TRUE(guard.has_value());
    }
    EXPECT_EQ(Filed(), 0u);
    EXPECT_TRUE(DueKeys(t0_ + seconds(1)).empty());
}

TEST_F(LeaseEvictionIndexTest, WalksKeysOldestLeaseFirstUpToNow) {
    Publish("late", t0_ + seconds(5));
    Publish("middle", t0_ + seconds(1));
    Publish("early", t0_);

    // A bucket is due once it has begun, so "late" is not yet.
    EXPECT_EQ(DueKeys(t0_ + seconds(2)),
              (std::vector<std::string>{"early", "middle"}));
    EXPECT_EQ(DueKeys(t0_ + seconds(6)),
              (std::vector<std::string>{"early", "middle", "late"}));
}

TEST_F(LeaseEvictionIndexTest, KeysInOneBucketShareARank) {
    Publish("a", t0_);
    Publish("b", t0_);
    Publish("c", t0_ + seconds(1));

    const auto entries = Walk(t0_ + seconds(5));
    ASSERT_EQ(entries.size(), 3u);
    EXPECT_EQ(entries[0].rank, entries[1].rank);
    EXPECT_EQ(entries[2].keys, std::vector<std::string>{"c"});
    EXPECT_LT(entries[1].rank, entries[2].rank);
}

TEST_F(LeaseEvictionIndexTest, AReadLeavesTheKeyAndStillAtCatchesIt) {
    Publish("k", t0_);
    Publish("unmoved", t0_);

    // A read extends the lease without a write lock, so the key stays filed
    // at its old deadline.
    {
        auto guard = ns_.objects.Read("k");
        ASSERT_TRUE(guard.has_value());
        SetDeadline(guard->metadata(), t0_ + seconds(10));
    }
    // Under the write lock the cursor sees "k" moved on; releasing that lock
    // files it at its real deadline.
    auto cursor = ns_.eviction_index().Scan(t0_ + seconds(1));
    size_t entries = 0;
    while (auto entry = cursor->Next()) {
        ++entries;
        ASSERT_EQ(entry->keys.size(), 1u);
        const std::string key(entry->keys.front());
        auto guard = ns_.objects.Write(key);
        ASSERT_TRUE(guard.has_value());
        EXPECT_EQ(cursor->StillAt(*guard, *entry), key == "unmoved");
    }
    EXPECT_EQ(entries, 2u);
    EXPECT_EQ(DueKeys(t0_ + seconds(1)), std::vector<std::string>{"unmoved"});
    EXPECT_EQ(DueKeys(t0_ + seconds(11)),
              (std::vector<std::string>{"unmoved", "k"}));
}

TEST_F(LeaseEvictionIndexTest, KeyLeavesTheIndexWithItsObjectOrItsLastMemory) {
    Publish("torn_down", t0_);
    Publish("evicted", t0_);
    ASSERT_EQ(Filed(), 2u);

    {
        auto guard = ns_.objects.Write("torn_down");
        ASSERT_TRUE(guard.has_value());
        guard->TearDown();
    }
    {
        auto guard = ns_.objects.Write("evicted");
        ASSERT_TRUE(guard.has_value());
        (void)guard->metadata().PopReplicas();
    }

    EXPECT_EQ(Filed(), 0u);
    EXPECT_TRUE(DueKeys(t0_ + seconds(1)).empty());
}

// A group's members come in one entry, filed at the lease they share.
TEST_F(LeaseEvictionIndexTest, AGroupIsOneEntryAtItsSharedLease) {
    PublishInGroup("g1", "g", t0_);
    Publish("k", t0_ + seconds(1));
    // Sets the shared lease later, which moves g1 with it.
    PublishInGroup("g2", "g", t0_ + seconds(2));

    EXPECT_EQ(Filed(), 3u);
    EXPECT_EQ(DueEntries(t0_ + seconds(1)),
              (std::vector<std::set<std::string>>{{"k"}}));
    EXPECT_EQ(DueEntries(t0_ + seconds(3)),
              (std::vector<std::set<std::string>>{{"k"}, {"g1", "g2"}}));
}

// A member leaves the group's entry with its object or its last memory, and
// the entry goes with the last member.
TEST_F(LeaseEvictionIndexTest, AGroupEntryGoesWithItsLastMember) {
    PublishInGroup("g1", "g", t0_);
    PublishInGroup("g2", "g", t0_);
    {
        auto guard = ns_.objects.Write("g1");
        ASSERT_TRUE(guard.has_value());
        guard->TearDown();
    }
    EXPECT_EQ(Filed(), 1u);
    EXPECT_EQ(DueEntries(t0_ + seconds(1)),
              (std::vector<std::set<std::string>>{{"g2"}}));

    {
        auto guard = ns_.objects.Write("g2");
        ASSERT_TRUE(guard.has_value());
        (void)guard->metadata().PopReplicas();
    }
    EXPECT_EQ(Filed(), 0u);
    EXPECT_TRUE(Walk(t0_ + seconds(1)).empty());

    // A member published afterwards starts a new entry.
    PublishInGroup("g3", "g", t0_);
    EXPECT_EQ(DueEntries(t0_ + seconds(1)),
              (std::vector<std::set<std::string>>{{"g3"}}));
}

// A key published again in another group, or in none, under one write lock
// moves to where its new object is filed.
TEST_F(LeaseEvictionIndexTest, AKeyPublishedAgainMovesToItsNewGroup) {
    PublishInGroup("g1", "g", t0_);
    PublishInGroup("g2", "g", t0_);
    PublishInGroup("h1", "h", t0_ + seconds(1));
    {
        auto guard = ns_.objects.Write("g1");
        ASSERT_TRUE(guard.has_value());
        guard->TearDown();
        PublishIn(*guard, "g1", "h");
    }
    {
        auto guard = ns_.objects.Write("g2");
        ASSERT_TRUE(guard.has_value());
        guard->TearDown();
        PublishIn(*guard, "g2", std::string());
        SetDeadline(guard->metadata(), t0_ + seconds(2));
    }

    EXPECT_EQ(Filed(), 3u);
    EXPECT_EQ(DueEntries(t0_ + seconds(3)),
              (std::vector<std::set<std::string>>{{"g1", "h1"}, {"g2"}}));
}

// A key published in another group, or into a group, after the walk copied
// its entry no longer stands there, even at the same rank.
TEST_F(LeaseEvictionIndexTest, AKeyMovedToAnotherGroupIsNoLongerAtItsEntry) {
    PublishInGroup("g1", "g", t0_);
    PublishInGroup("g2", "g", t0_);
    Publish("k", t0_);
    // The walk copies the bucket as it starts.
    auto cursor = ns_.eviction_index().Scan(t0_ + seconds(1));
    std::optional<EvictionIndex::Entry> entry = cursor->Next();
    ASSERT_TRUE(entry.has_value());

    for (const std::string key : {"g2", "k"}) {
        auto guard = ns_.objects.Write(key);
        ASSERT_TRUE(guard.has_value());
        guard->TearDown();
        PublishIn(*guard, key, "h");
        SetDeadline(guard->metadata(), t0_);
    }
    size_t entries = 0;
    for (; entry.has_value(); entry = cursor->Next()) {
        ++entries;
        for (const std::string_view key : entry->keys) {
            auto guard = ns_.objects.Write(key);
            ASSERT_TRUE(guard.has_value());
            EXPECT_EQ(cursor->StillAt(*guard, *entry), key == "g1") << key;
        }
    }
    EXPECT_EQ(entries, 2u);
}

// A group whose members were all torn down after the walk copied its entry
// is gone; a key published again under the same id joins a new incarnation,
// which the old entry does not stand for.
TEST_F(LeaseEvictionIndexTest, AKeyInARecreatedGroupIsNoLongerAtItsEntry) {
    PublishInGroup("g1", "g", t0_);
    PublishInGroup("g2", "g", t0_);
    auto cursor = ns_.eviction_index().Scan(t0_ + seconds(1));
    std::optional<EvictionIndex::Entry> entry = cursor->Next();
    ASSERT_TRUE(entry.has_value());

    for (const std::string key : {"g1", "g2"}) {
        auto guard = ns_.objects.Write(key);
        ASSERT_TRUE(guard.has_value());
        guard->TearDown();
    }
    PublishInGroup("g1", "g", t0_);

    auto guard = ns_.objects.Write("g1");
    ASSERT_TRUE(guard.has_value());
    EXPECT_FALSE(cursor->StillAt(*guard, *entry));
}

// A group's last member torn down and published again under the same id
// within one write lock starts a new group with a new lease, which an entry
// copied before does not stand for.
TEST_F(LeaseEvictionIndexTest, AGroupRecreatedInPlaceIsANewIncarnation) {
    PublishInGroup("g1", "g", t0_);
    auto cursor = ns_.eviction_index().Scan(t0_ + seconds(1));
    std::optional<EvictionIndex::Entry> entry = cursor->Next();
    ASSERT_TRUE(entry.has_value());
    {
        auto guard = ns_.objects.Write("g1");
        ASSERT_TRUE(guard.has_value());
        guard->TearDown();
        PublishIn(*guard, "g1", "g");
        SetDeadline(guard->metadata(), t0_);
    }

    auto guard = ns_.objects.Write("g1");
    ASSERT_TRUE(guard.has_value());
    EXPECT_FALSE(cursor->StillAt(*guard, *entry));
}

// A member replaced in place while others keep the group stays in it.
TEST_F(LeaseEvictionIndexTest, AMemberReplacedInPlaceStaysInItsGroup) {
    PublishInGroup("g1", "g", t0_);
    PublishInGroup("g2", "g", t0_);
    auto cursor = ns_.eviction_index().Scan(t0_ + seconds(1));
    std::optional<EvictionIndex::Entry> entry = cursor->Next();
    ASSERT_TRUE(entry.has_value());
    {
        auto guard = ns_.objects.Write("g1");
        ASSERT_TRUE(guard.has_value());
        guard->TearDown();
        PublishIn(*guard, "g1", "g");
    }

    {
        auto guard = ns_.objects.Write("g1");
        ASSERT_TRUE(guard.has_value());
        EXPECT_TRUE(cursor->StillAt(*guard, *entry));
    }
    EXPECT_EQ(DueEntries(t0_ + seconds(1)),
              (std::vector<std::set<std::string>>{{"g1", "g2"}}));
}

// An index that counts what the namespace tells it.
class CountingIndex final : public EvictionIndex {
   public:
    CountingIndex(bool observes_reads, int* reads, int* writes)
        : EvictionIndex(observes_reads), reads_(reads), writes_(writes) {}

    void OnRead(const route::ReadGuard&) override { ++*reads_; }
    void OnWrite(const route::WriteGuard&) override { ++*writes_; }
    std::unique_ptr<Cursor> Scan(Clock::time_point) const override {
        return nullptr;
    }

   private:
    int* reads_;
    int* writes_;
};

TEST(EvictionIndexWiringTest, ReadsReachOnlyAnIndexThatObservesThem) {
    for (const bool observes_reads : {true, false}) {
        int reads = 0;
        int writes = 0;
        metadata::Namespace ns(
            TenantId(), nullptr, nullptr,
            std::make_unique<CountingIndex>(observes_reads, &reads, &writes));
        ASSERT_NE(test::PublishObject(ns.objects, "k"), 0u);
        EXPECT_EQ(writes, 1);

        ASSERT_TRUE(ns.objects.Read("k").has_value());
        EXPECT_EQ(reads, observes_reads ? 1 : 0);
        ASSERT_TRUE(ns.objects.Write("k").has_value());
        EXPECT_EQ(writes, 2);
    }
}

}  // namespace
}  // namespace mooncake

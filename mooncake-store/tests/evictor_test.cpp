#include "eviction/evictor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "allocator.h"
#include "eviction/offloading_reclaimer.h"
#include "eviction/report_observers.h"
#include "master_metric_manager.h"
#include "metadata/namespace_table.h"

namespace mooncake {
namespace {

using eviction::ByteGoal;
using eviction::Clock;
using eviction::NothingCause;
using eviction::ObjectGoal;
using eviction::Outcome;
using eviction::Scope;
using eviction::SkipReason;
using std::chrono::seconds;

bool InPlan(const eviction::Plan& plan, const Replica& replica) {
    return std::find(plan.release.begin(), plan.release.end(), replica.id()) !=
           plan.release.end();
}

// Every completed memory replica nobody holds goes. A soft-pinned key is
// deferred when `defer_soft_pinned` is set, and never taken otherwise.
class FakePolicy final : public eviction::EvictionPolicy {
   public:
    tl::expected<eviction::Plan, SkipReason> Decide(
        const eviction::Candidate& candidate) const override {
        if (!candidate.fallback && soft_pinned.count(candidate.key) > 0) {
            return tl::make_unexpected(defer_soft_pinned
                                           ? SkipReason::kDeferred
                                           : SkipReason::kSoftPinned);
        }
        eviction::Plan plan;
        for (const Replica& replica : candidate.metadata.GetAllReplicas()) {
            if (replica.is_memory_replica() && replica.is_completed() &&
                replica.get_refcnt() == 0) {
                plan.release.push_back(replica.id());
            }
        }
        if (plan.release.empty()) {
            return tl::make_unexpected(SkipReason::kNothingToRelease);
        }
        return plan;
    }

    bool defer_soft_pinned = false;
    std::set<std::string> soft_pinned;
};

// Takes the planned replicas off on the spot and records what it did; fails
// the `fail_at`-th Apply, counting from zero, as an OpLog write would. Runs
// `on_apply` first, as a writer of another key would meanwhile.
class FakeReclaimer final : public eviction::Reclaimer {
   public:
    tl::expected<eviction::Effect, ErrorCode> Apply(
        metadata::Namespace& ns, const route::WriteGuard& hold,
        const eviction::Plan& plan, const eviction::Report&) override {
        if (static_cast<int>(applies++) == fail_at) {
            return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
        }
        if (on_apply) {
            on_apply(ns, hold.key());
        }
        applied.push_back(hold.key());
        ObjectMetadata& metadata = hold.metadata();
        eviction::Effect effect;
        effect.detached = metadata.PopReplicas(
            [&plan](const Replica& replica) { return InPlan(plan, replica); });
        effect.released_bytes = metadata.TotalLength() * effect.detached.size();
        effect.needs_teardown = !metadata.IsValid();
        effect.outcome =
            effect.needs_teardown ? Outcome::kDropped : Outcome::kDemoted;
        return effect;
    }

    void TearDown(metadata::Namespace&,
                  const route::WriteGuard& hold) override {
        torn_down.push_back(hold.key());
        hold.TearDown();
    }

    int fail_at = -1;
    std::function<void(metadata::Namespace&, const std::string&)> on_apply;
    size_t applies = 0;
    std::vector<std::string> applied;
    std::vector<std::string> torn_down;
};

class RecordingObserver final : public eviction::EvictionObserver {
   public:
    void OnApplied(const eviction::ObjectView& view,
                   const eviction::Effect&) override {
        applied.push_back(view.key);
    }
    void OnSkipped(const eviction::ObjectView& view,
                   SkipReason reason) override {
        skipped.emplace_back(view.key, reason);
    }
    void OnRunEnd(const Scope&, const eviction::Report& report) override {
        ++runs;
        filed = report.filed;
    }

    std::vector<std::string> applied;
    std::vector<std::pair<std::string, SkipReason>> skipped;
    int runs = 0;
    uint64_t filed = 0;
};

class EvictorTest : public ::testing::Test {
   protected:
    // Publishes `key` in `tenant`'s namespace with `replicas` 128 B memory
    // replicas and its lease ending `age` ago.
    void Publish(const std::string& tenant, const std::string& key,
                 Clock::duration age, const std::string& group_id = {},
                 int replicas = 1) {
        metadata::Namespace& ns = namespaces_.GetOrCreate(TenantId(tenant));
        auto guard = ns.objects.WriteKey(key);
        std::vector<Replica> list;
        for (int i = 0; i < replicas; ++i) {
            list.emplace_back(allocator_->allocate(128),
                              ReplicaStatus::COMPLETE);
        }
        (void)guard.Publish(
            UUID{1, 2}, Clock::time_point{}, 128, std::move(list), std::nullopt,
            false, ObjectDataType::UNKNOWN, group_id, TenantId(tenant), key);
        SetDeadline(guard.metadata(), Clock::now() - age);
    }

    static void SetDeadline(const ObjectMetadata& metadata,
                            Clock::time_point deadline) {
        SpinLocker locker(&metadata.lock);
        metadata.lease_->SetDeadline(deadline);
    }

    // Publishes `key` in `tenant`'s namespace with its lease ending at
    // `deadline`, so keys published apart can share one rank.
    void PublishAt(const std::string& tenant, const std::string& key,
                   Clock::time_point deadline) {
        Publish(tenant, key, seconds(0));
        auto hold =
            namespaces_.GetOrCreate(TenantId(tenant)).objects.Write(key);
        ASSERT_TRUE(hold.has_value());
        SetDeadline(hold->metadata(), deadline);
    }

    // Adds `count` 128 B memory replicas to `key` in `tenant`'s namespace.
    void AddMemoryReplicas(const std::string& tenant, const std::string& key,
                           int count) {
        auto hold =
            namespaces_.GetOrCreate(TenantId(tenant)).objects.Write(key);
        ASSERT_TRUE(hold.has_value());
        std::vector<Replica> list;
        for (int i = 0; i < count; ++i) {
            list.emplace_back(allocator_->allocate(128),
                              ReplicaStatus::COMPLETE);
        }
        hold->metadata().AddReplicas(std::move(list));
    }

    eviction::Report Run(const ObjectGoal& goal) {
        eviction::Evictor evictor(namespaces_, policy_, reclaimer_,
                                  {&observer_});
        return evictor.Run(Scope::All(), goal);
    }

    eviction::Report Run(const ByteGoal& goal) {
        eviction::Evictor evictor(namespaces_, policy_, reclaimer_,
                                  {&observer_});
        return evictor.Run(TenantId("a"), goal);
    }

    // Declared first, so it outlives the replicas it gave out.
    std::shared_ptr<OffsetBufferAllocator> allocator_ =
        std::make_shared<OffsetBufferAllocator>("segment", 0x100000000ULL,
                                                64 * 1024 * 1024, "segment");
    metadata::NamespaceTable namespaces_{[](const TenantId& tenant_id) {
        return std::make_unique<metadata::Namespace>(tenant_id);
    }};
    FakePolicy policy_;
    FakeReclaimer reclaimer_;
    RecordingObserver observer_;
};

// Each namespace gives up its own share first, however much older another
// namespace's objects are.
TEST_F(EvictorTest, EachNamespaceGivesUpItsOwnShare) {
    Publish("a", "a1", seconds(20));
    Publish("a", "a2", seconds(18));
    Publish("a", "a3", seconds(16));
    Publish("a", "a4", seconds(14));
    Publish("b", "b1", seconds(8));
    Publish("b", "b2", seconds(6));
    Publish("b", "b3", seconds(4));
    Publish("b", "b4", seconds(2));

    const auto report = Run(ObjectGoal{.target_ratio = 0.5, .min_ratio = 0.5});

    EXPECT_EQ(std::set<std::string>(reclaimer_.applied.begin(),
                                    reclaimer_.applied.end()),
              (std::set<std::string>{"a1", "a2", "b1", "b2"}));
    EXPECT_EQ(reclaimer_.torn_down, reclaimer_.applied);
    EXPECT_EQ(report.filed, 8u);
    EXPECT_EQ(report.evicted_objects, 4u);
    EXPECT_EQ(report.released_bytes, 4u * 128);
    EXPECT_EQ(report.Count(Outcome::kDropped), 4u);
}

// A share is rounded up, so a namespace too small for a whole object of it
// still gives one up.
TEST_F(EvictorTest, AShareIsRoundedUp) {
    Publish("a", "a1", seconds(20));
    Publish("a", "a2", seconds(18));
    Publish("a", "a3", seconds(16));
    Publish("a", "a4", seconds(14));
    Publish("b", "b1", seconds(2));

    const auto report =
        Run(ObjectGoal{.target_ratio = 0.25, .min_ratio = 0.25});

    EXPECT_EQ(std::set<std::string>(reclaimer_.applied.begin(),
                                    reclaimer_.applied.end()),
              (std::set<std::string>{"a1", "b1"}));
    EXPECT_EQ(report.evicted_objects, 2u);
}

// What a namespace cannot give up of its share is not made up by another.
TEST_F(EvictorTest, ANamespaceShortOfItsShareIsNotMadeUpByAnother) {
    Publish("a", "a1", seconds(20));
    Publish("a", "a2", seconds(18));
    Publish("a", "a3", seconds(16));
    Publish("a", "a4", seconds(14));
    Publish("c", "c1", seconds(9));
    Publish("c", "c2", seconds(5));
    policy_.soft_pinned = {"c1", "c2"};

    // Shares of 2 and 1, of which c gives up nothing.
    const auto report = Run(ObjectGoal{.target_ratio = 0.5, .min_ratio = 0.5});

    EXPECT_EQ(std::set<std::string>(reclaimer_.applied.begin(),
                                    reclaimer_.applied.end()),
              (std::set<std::string>{"a1", "a2"}));
    EXPECT_EQ(report.evicted_objects, 2u);
}

// A run cut short leaves the namespaces after the one it stopped in to the
// next run, which starts one namespace further on.
TEST_F(EvictorTest, ARunCutShortLeavesTheRestToTheNextRun) {
    Publish("a", "a1", seconds(10));
    Publish("a", "a2", seconds(8));
    Publish("b", "b1", seconds(10));
    Publish("b", "b2", seconds(8));
    eviction::Evictor evictor(namespaces_, policy_, reclaimer_, {});
    const ObjectGoal goal{.target_ratio = 0.5, .min_ratio = 0.5};

    // Shares of one each; the second reclaim fails, as an OpLog write would.
    reclaimer_.fail_at = 1;
    EXPECT_TRUE(evictor.Run(Scope::All(), goal).stopped);
    ASSERT_EQ(reclaimer_.applied.size(), 1u);
    const char first = reclaimer_.applied.front()[0];

    reclaimer_.fail_at = -1;
    reclaimer_.applied.clear();
    EXPECT_FALSE(evictor.Run(Scope::All(), goal).stopped);
    ASSERT_EQ(reclaimer_.applied.size(), 2u);
    EXPECT_NE(reclaimer_.applied.front()[0], first);
}

// A ratio above 1 asks for every eligible object; one that is negative or not
// a number asks for none.
TEST_F(EvictorTest, ARatioOutsideZeroToOneIsClamped) {
    Publish("a", "k1", seconds(10));
    Publish("a", "k2", seconds(8));

    for (const double ratio :
         {std::numeric_limits<double>::quiet_NaN(), -0.5}) {
        const auto report =
            Run(ObjectGoal{.target_ratio = ratio, .min_ratio = ratio});
        EXPECT_EQ(report.evicted_objects, 0u) << ratio;
    }
    EXPECT_TRUE(reclaimer_.applied.empty());

    const auto report = Run(ObjectGoal{.target_ratio = 1.5, .min_ratio = 1.5});
    EXPECT_EQ(report.evicted_objects, 2u);
}

TEST_F(EvictorTest, StopsOnceTheTargetShareIsMet) {
    Publish("a", "k1", seconds(10));
    Publish("a", "k2", seconds(8));
    Publish("a", "k3", seconds(6));
    Publish("a", "k4", seconds(4));

    const auto report = Run(ObjectGoal{.target_ratio = 0.5, .min_ratio = 0.5});

    EXPECT_EQ(reclaimer_.applied, (std::vector<std::string>{"k1", "k2"}));
    EXPECT_EQ(report.filed, 4u);
    EXPECT_EQ(report.evicted_objects, 2u);
}

TEST_F(EvictorTest, BytesGoalCountsReleasedMemory) {
    Publish("a", "k1", seconds(10), {}, /*replicas=*/2);
    Publish("a", "k2", seconds(8));

    const auto report = Run(ByteGoal{.target_bytes = 200});

    EXPECT_EQ(reclaimer_.applied, std::vector<std::string>{"k1"});
    EXPECT_EQ(report.released_bytes, 256u);
}

TEST_F(EvictorTest, DeferredObjectsGoOnlyWhileShortOfTheMinimum) {
    Publish("a", "pinned", seconds(10));
    Publish("a", "plain", seconds(8));
    policy_.soft_pinned = {"pinned"};

    // Not deferred: the soft pin holds.
    Run(ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});
    EXPECT_EQ(reclaimer_.applied, std::vector<std::string>{"plain"});
    ASSERT_FALSE(observer_.skipped.empty());
    EXPECT_EQ(observer_.skipped.front(),
              std::make_pair(std::string("pinned"), SkipReason::kSoftPinned));

    // Deferred: still short of the minimum once the walk is done, so the
    // pinned one goes too.
    Publish("a", "plain", seconds(8));
    reclaimer_.applied.clear();
    observer_.skipped.clear();
    policy_.defer_soft_pinned = true;
    Run(ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});
    EXPECT_EQ(reclaimer_.applied,
              (std::vector<std::string>{"plain", "pinned"}));
    EXPECT_EQ(observer_.skipped.front(),
              std::make_pair(std::string("pinned"), SkipReason::kDeferred));
}

TEST_F(EvictorTest, DeferredObjectsStayOnceTheMinimumIsMet) {
    Publish("a", "pinned", seconds(10));
    Publish("a", "plain", seconds(8));
    policy_.soft_pinned = {"pinned"};
    policy_.defer_soft_pinned = true;

    Run(ObjectGoal{.target_ratio = 1.0, .min_ratio = 0.5});

    EXPECT_EQ(reclaimer_.applied, std::vector<std::string>{"plain"});
}

TEST_F(EvictorTest, AMinimumAboveTheTargetIsLowered) {
    Publish("a", "pinned", seconds(10));
    Publish("a", "plain", seconds(8));
    policy_.soft_pinned = {"pinned"};
    policy_.defer_soft_pinned = true;

    Run(ObjectGoal{.target_ratio = 0.5, .min_ratio = 1.0});

    EXPECT_EQ(reclaimer_.applied, std::vector<std::string>{"plain"});
}

// The fallback comes back to the publication the policy deferred; one that
// replaced it meanwhile was never judged.
TEST_F(EvictorTest, AReplacedDeferredObjectIsLeftAlone) {
    Publish("a", "pinned", seconds(10));
    Publish("a", "plain", seconds(8));
    policy_.soft_pinned = {"pinned"};
    policy_.defer_soft_pinned = true;
    reclaimer_.on_apply = [this](metadata::Namespace& ns,
                                 const std::string& key) {
        if (key != "plain") {
            return;
        }
        auto hold = ns.objects.Write("pinned");
        ASSERT_TRUE(hold.has_value());
        hold->TearDown();
        std::vector<Replica> list;
        list.emplace_back(allocator_->allocate(128), ReplicaStatus::COMPLETE);
        (void)hold->Publish(UUID{1, 2}, Clock::time_point{}, 128,
                            std::move(list), std::nullopt, false,
                            ObjectDataType::UNKNOWN, std::string(),
                            TenantId("a"), "pinned");
    };

    Run(ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});

    EXPECT_EQ(reclaimer_.applied, std::vector<std::string>{"plain"});
}

TEST_F(EvictorTest, AFailedReclaimStopsTheRun) {
    Publish("a", "k1", seconds(10));
    Publish("a", "k2", seconds(8));
    Publish("a", "k3", seconds(6));
    reclaimer_.fail_at = 1;

    const auto report = Run(ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});

    EXPECT_TRUE(report.stopped);
    EXPECT_EQ(reclaimer_.applied, std::vector<std::string>{"k1"});
    EXPECT_EQ(reclaimer_.applies, 2u);
}

TEST_F(EvictorTest, AGroupGoesTogetherByItsSharedLease) {
    // Members share one lease, so the last one published sets it for both.
    Publish("a", "g1", seconds(10), "g");
    Publish("a", "g2", seconds(10), "g");
    Publish("a", "x", seconds(5));
    Publish("a", "y", seconds(3));

    // One object's worth of target, but the group goes whole.
    const auto report =
        Run(ObjectGoal{.target_ratio = 0.25, .min_ratio = 0.25});

    EXPECT_EQ(std::set<std::string>(reclaimer_.applied.begin(),
                                    reclaimer_.applied.end()),
              (std::set<std::string>{"g1", "g2"}));
    EXPECT_EQ(std::set<std::string>(reclaimer_.torn_down.begin(),
                                    reclaimer_.torn_down.end()),
              (std::set<std::string>{"g1", "g2"}));
    EXPECT_EQ(report.evicted_objects, 2u);
}

// Each member is judged again under its own lock: one replaced with an object
// outside the group, while the walk was on another member, is left alone.
TEST_F(EvictorTest, AMemberReplacedBeforeItsTurnIsLeftAlone) {
    Publish("a", "g1", seconds(10), "g");
    Publish("a", "g2", seconds(10), "g");
    std::string replaced;
    reclaimer_.on_apply = [this, &replaced](metadata::Namespace& ns,
                                            const std::string& key) {
        if (!replaced.empty()) {
            return;
        }
        replaced = key == "g1" ? "g2" : "g1";
        auto hold = ns.objects.Write(replaced);
        ASSERT_TRUE(hold.has_value());
        hold->TearDown();
        std::vector<Replica> list;
        list.emplace_back(allocator_->allocate(128), ReplicaStatus::COMPLETE);
        (void)hold->Publish(UUID{1, 2}, Clock::time_point{}, 128,
                            std::move(list), std::nullopt, false,
                            ObjectDataType::UNKNOWN, std::string(),
                            TenantId("a"), replaced);
        SetDeadline(hold->metadata(), Clock::now() + std::chrono::hours(1));
    };

    Run(ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});

    ASSERT_FALSE(replaced.empty());
    ASSERT_EQ(reclaimer_.applied.size(), 1u);
    EXPECT_NE(reclaimer_.applied.front(), replaced);
    EXPECT_EQ(reclaimer_.torn_down, reclaimer_.applied);
    auto hold = namespaces_.GetOrCreate(TenantId("a")).objects.Read(replaced);
    ASSERT_TRUE(hold.has_value());
    EXPECT_TRUE(hold->metadata().IsValid());
}

// The walk meets a group once, so a member the policy defers is kept for the
// fallback once, however many members the group has.
TEST_F(EvictorTest, AGroupMemberIsDeferredOnce) {
    Publish("a", "g1", seconds(10), "g");
    Publish("a", "g2", seconds(10), "g");
    Publish("a", "g3", seconds(10), "g");
    policy_.soft_pinned = {"g2"};
    policy_.defer_soft_pinned = true;

    Run(ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});

    EXPECT_EQ(std::set<std::string>(reclaimer_.applied.begin(),
                                    reclaimer_.applied.end() - 1),
              (std::set<std::string>{"g1", "g3"}));
    EXPECT_EQ(reclaimer_.applied.back(), "g2");
    EXPECT_EQ(
        std::count(observer_.skipped.begin(), observer_.skipped.end(),
                   std::make_pair(std::string("g2"), SkipReason::kDeferred)),
        1);
}

// The fallback asks whether the minimum is met only between entries, so a
// group's deferred members go together even past the minimum.
TEST_F(EvictorTest, AGroupsDeferredMembersGoTogetherInTheFallback) {
    Publish("a", "g1", seconds(10), "g");
    Publish("a", "g2", seconds(10), "g");
    Publish("a", "g3", seconds(10), "g");
    Publish("a", "x", seconds(5));
    Publish("a", "y", seconds(3));
    policy_.soft_pinned = {"g1", "g2", "g3"};
    policy_.defer_soft_pinned = true;

    // The walk frees x and y, one short of the minimum of 3 (2.5 rounded up).
    const auto report = Run(ObjectGoal{.target_ratio = 1.0, .min_ratio = 0.5});

    ASSERT_EQ(reclaimer_.applied.size(), 5u);
    EXPECT_EQ(std::vector<std::string>(reclaimer_.applied.begin(),
                                       reclaimer_.applied.begin() + 2),
              (std::vector<std::string>{"x", "y"}));
    EXPECT_EQ(std::set<std::string>(reclaimer_.applied.begin() + 2,
                                    reclaimer_.applied.end()),
              (std::set<std::string>{"g1", "g2", "g3"}));
    EXPECT_EQ(report.evicted_objects, 5u);
}

// A read that extends a lease leaves the key filed where it was. The run meets
// it there, leaves it alone, and refiles it at its real deadline.
TEST_F(EvictorTest, AKeyWhoseLeaseWasExtendedIsRefiledNotEvicted) {
    Publish("a", "k", seconds(10));
    metadata::Namespace& ns = namespaces_.GetOrCreate(TenantId("a"));
    {
        auto hold = ns.objects.Read("k");
        ASSERT_TRUE(hold.has_value());
        SetDeadline(hold->metadata(), Clock::now() + std::chrono::hours(1));
    }

    Run(ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});

    EXPECT_TRUE(reclaimer_.applied.empty());
    ASSERT_EQ(observer_.skipped.size(), 1u);
    EXPECT_EQ(observer_.skipped.front(),
              std::make_pair(std::string("k"), SkipReason::kMoved));
    EXPECT_FALSE(ns.eviction_index().Scan(Clock::now())->Next().has_value());
}

TEST_F(EvictorTest, ObserversHearOfEachObjectAndTheRun) {
    Publish("a", "k1", seconds(10));
    Publish("b", "k2", seconds(8));

    Run(ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});

    EXPECT_EQ(std::set<std::string>(observer_.applied.begin(),
                                    observer_.applied.end()),
              (std::set<std::string>{"k1", "k2"}));
    EXPECT_EQ(observer_.runs, 1);
    EXPECT_EQ(observer_.filed, 2u);
}

TEST_F(EvictorTest, AScopedRunStaysInItsNamespace) {
    Publish("a", "a1", seconds(10));
    Publish("b", "b1", seconds(10));

    eviction::Evictor evictor(namespaces_, policy_, reclaimer_, {});
    const TenantId tenant("b");
    const auto report = evictor.Run(tenant, ByteGoal{.target_bytes = 1 << 20});

    EXPECT_EQ(reclaimer_.applied, std::vector<std::string>{"b1"});
    EXPECT_EQ(report.filed, 1u);

    // A tenant with no namespace owns nothing to evict.
    const TenantId absent("absent");
    EXPECT_EQ(evictor.Run(absent, ByteGoal{.target_bytes = 1}).filed, 0u);
}

// --- OffloadingReclaimer -----------------------------------------------------

// Accepts the first `capacity` offloads.
class FakeOffloadQueue final : public eviction::OffloadQueue {
   public:
    bool Push(const TenantId&, const std::string& key, Replica&,
              std::vector<UUID>* mirror_clients) override {
        if (pushed.size() >= capacity) {
            return false;
        }
        pushed.push_back(key);
        mirror_clients->push_back(UUID{7, 7});
        return true;
    }

    size_t capacity = SIZE_MAX;
    std::vector<std::string> pushed;
};

class OffloadingReclaimerTest : public EvictorTest {
   protected:
    tl::expected<eviction::Effect, ErrorCode> Apply(
        const std::string& key, eviction::OffloadingReclaimer::Options options,
        const eviction::Report& so_far = {}) {
        eviction::OffloadingReclaimer reclaimer(inner_, queue_, options);
        metadata::Namespace& ns = namespaces_.GetOrCreate(TenantId("a"));
        auto hold = ns.objects.Write(key);
        EXPECT_TRUE(hold.has_value());
        eviction::Candidate candidate{ns.id(), key, hold->metadata(),
                                      Clock::now(), false};
        auto plan = policy_.Decide(candidate);
        EXPECT_TRUE(plan.has_value());
        auto effect = reclaimer.Apply(ns, *hold, *plan, so_far);
        if (effect && hold->state().offloading_task.has_value()) {
            offloading_source_ = hold->state().offloading_task->source_id;
        }
        return effect;
    }

    FakeReclaimer inner_;
    FakeOffloadQueue queue_;
    std::optional<ReplicaID> offloading_source_;
};

TEST_F(OffloadingReclaimerTest, PinsOneReplicaAndReleasesTheRest) {
    Publish("a", "k", seconds(10), {}, /*replicas=*/2);

    auto effect = Apply("k", {});

    ASSERT_TRUE(effect.has_value());
    EXPECT_EQ(effect->outcome, Outcome::kOffloadQueued);
    EXPECT_EQ(effect->offload_bytes, 128u);
    EXPECT_EQ(effect->released_bytes, 128u);
    EXPECT_FALSE(effect->needs_teardown);
    EXPECT_EQ(queue_.pushed, std::vector<std::string>{"k"});
    EXPECT_TRUE(offloading_source_.has_value());
}

TEST_F(OffloadingReclaimerTest, AnObjectOnLocalDiskIsDemotedWithoutOffload) {
    Publish("a", "k", seconds(10));
    {
        auto hold = namespaces_.GetOrCreate(TenantId("a")).objects.Write("k");
        std::vector<Replica> disk;
        disk.emplace_back(UUID{3, 4}, 128, "endpoint", ReplicaStatus::COMPLETE);
        hold->metadata().AddReplicas(std::move(disk));
    }

    auto effect = Apply("k", {});

    ASSERT_TRUE(effect.has_value());
    EXPECT_EQ(effect->outcome, Outcome::kDemoted);
    EXPECT_TRUE(queue_.pushed.empty());
}

// With the OpLog on, a local-disk replica being evicted stays attached,
// marked removed, until its removal is durable: it keeps nothing by then.
TEST_F(OffloadingReclaimerTest,
       ALocalDiskReplicaBeingRemovedStillNeedsOffload) {
    Publish("a", "k", seconds(10));
    {
        auto hold = namespaces_.GetOrCreate(TenantId("a")).objects.Write("k");
        std::vector<Replica> disk;
        disk.emplace_back(UUID{3, 4}, 128, "endpoint", ReplicaStatus::COMPLETE);
        disk.back().mark_removed();
        hold->metadata().AddReplicas(std::move(disk));
    }

    auto effect = Apply("k", {});

    ASSERT_TRUE(effect.has_value());
    EXPECT_EQ(effect->outcome, Outcome::kOffloadQueued);
    EXPECT_EQ(queue_.pushed, std::vector<std::string>{"k"});
}

// The replica the offload in flight pinned keeps the data, so memory added
// since goes without a second offload, which nothing would unpin.
TEST_F(OffloadingReclaimerTest, AnOffloadInFlightIsNotQueuedAgain) {
    Publish("a", "k", seconds(10));
    ASSERT_EQ(Apply("k", {})->outcome, Outcome::kOffloadQueued);
    const auto source = offloading_source_;
    AddMemoryReplicas("a", "k", 1);

    auto effect = Apply("k", {});

    ASSERT_TRUE(effect.has_value());
    EXPECT_EQ(effect->outcome, Outcome::kDemoted);
    EXPECT_EQ(effect->released_bytes, 128u);
    EXPECT_EQ(queue_.pushed, std::vector<std::string>{"k"});
    EXPECT_EQ(offloading_source_, source);
}

TEST_F(OffloadingReclaimerTest, ARejectedOffloadKeepsTheObjectUnlessForced) {
    Publish("a", "kept", seconds(10));
    Publish("a", "forced", seconds(10));
    queue_.capacity = 0;

    auto kept = Apply("kept", {});
    ASSERT_TRUE(kept.has_value());
    EXPECT_EQ(kept->outcome, Outcome::kKept);
    EXPECT_TRUE(inner_.applied.empty());

    auto forced = Apply("forced", {.cap = 100, .force_evict = true});
    ASSERT_TRUE(forced.has_value());
    EXPECT_EQ(forced->outcome, Outcome::kDropped);
    EXPECT_EQ(forced->reason, eviction::DropReason::kOffloadRejected);
}

TEST_F(OffloadingReclaimerTest, PastTheCapAForcedRunDropsWithoutOffload) {
    Publish("a", "k", seconds(10));
    eviction::Report so_far;
    so_far.outcomes[static_cast<size_t>(Outcome::kOffloadQueued)] = 2;

    auto effect = Apply("k", {.cap = 2, .force_evict = true}, so_far);

    ASSERT_TRUE(effect.has_value());
    EXPECT_EQ(effect->outcome, Outcome::kDropped);
    EXPECT_EQ(effect->reason, eviction::DropReason::kOffloadCap);
    EXPECT_TRUE(queue_.pushed.empty());
}

// A COMPLETE memory replica whose allocator is already gone.
Replica DeadMemoryReplica() {
    auto dying = std::make_shared<OffsetBufferAllocator>(
        "dead_segment", 0x500000000ULL, 1 << 20, "dead_segment");
    auto buffer = dying->allocate(128);
    dying.reset();
    return Replica(std::move(buffer), ReplicaStatus::COMPLETE);
}

// Publishes `key` with a COMPLETE memory replica from `allocator`, filed at
// `filed_deadline`, then extends its lease through the read path. The index
// does not observe reads, so the key stays filed at the old rank while its
// real deadline moved ahead: eviction sees a real "moved" condition unless it
// removes the key first.
void PublishStaleIndexKey(
    metadata::Namespace& ns, const std::string& key,
    const std::shared_ptr<OffsetBufferAllocator>& allocator,
    Clock::time_point filed_deadline, Clock::time_point extended_deadline) {
    auto buffer = allocator->allocate(128);
    {
        auto guard = ns.objects.WriteKey(key);
        std::vector<Replica> replicas;
        replicas.emplace_back(std::move(buffer), ReplicaStatus::COMPLETE);
        (void)guard.Publish(UUID{1, 2}, Clock::time_point{}, 128,
                            std::move(replicas), std::nullopt, false,
                            ObjectDataType::UNKNOWN, std::string(), ns.id(),
                            key);
        SpinLocker locker(&guard.metadata().lock);
        guard.metadata().lease_->SetDeadline(filed_deadline);
    }
    auto read = ns.objects.Read(key);
    ASSERT_TRUE(read.has_value());
    SpinLocker locker(&read->metadata().lock);
    read->metadata().lease_->SetDeadline(extended_deadline);
}

// Flips the window on the first object the run skips, as an offboarding sweep
// starting after the run did.
class FlipWindowOnSkip final : public eviction::EvictionObserver {
   public:
    explicit FlipWindowOnSkip(std::atomic<bool>& window) : window_(window) {}
    void OnSkipped(const eviction::ObjectView&, SkipReason) override {
        window_.store(true, std::memory_order_release);
    }

   private:
    std::atomic<bool>& window_;
};

// --- Report observers --------------------------------------------------------

TEST_F(EvictorTest, NothingToReleaseSuppressesPermanentlyUnavailableCandidate) {
    metadata::Namespace& ns = namespaces_.GetOrCreate(TenantId("a"));
    {
        auto guard = ns.objects.WriteKey("dead");
        std::vector<Replica> replicas;
        replicas.push_back(DeadMemoryReplica());
        // The reference count is why the policy releases nothing, but the
        // allocator lifetime is what makes the candidate permanently useless.
        replicas.back().inc_refcnt();
        (void)guard.Publish(UUID{1, 2}, Clock::time_point{}, 128,
                            std::move(replicas), std::nullopt, false,
                            ObjectDataType::UNKNOWN, std::string(),
                            TenantId("a"), "dead");
        SetDeadline(guard.metadata(), Clock::now() - seconds(10));
    }

    const auto first = Run(ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});
    EXPECT_EQ(first.filed, 1u);
    EXPECT_EQ(first.visited, 1u);
    EXPECT_EQ(first.NothingCount(NothingCause::kPermanentlyUnavailable), 1u);
    EXPECT_EQ(first.suppressed_permanently_unavailable, 1u);
    EXPECT_TRUE(observer_.applied.empty());

    // The second run no longer has the candidate to visit.
    const auto second = Run(ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});
    EXPECT_EQ(second.filed, 0u);
    EXPECT_EQ(second.visited, 0u);
    EXPECT_TRUE(observer_.applied.empty());
}

// While an offboarding sweep is in progress, a permanently unavailable
// candidate leaves the index on first contact: it is never re-ranked as moved
// and the policy never runs for it.
TEST_F(EvictorTest,
       OffboardingWindowSuppressesPermanentlyUnavailableOnFirstContact) {
    metadata::Namespace& ns = namespaces_.GetOrCreate(TenantId("a"));
    {
        auto guard = ns.objects.WriteKey("dead");
        std::vector<Replica> replicas;
        replicas.push_back(DeadMemoryReplica());
        // Nobody holds it, so the policy would otherwise release it.
        (void)guard.Publish(UUID{1, 2}, Clock::time_point{}, 128,
                            std::move(replicas), std::nullopt, false,
                            ObjectDataType::UNKNOWN, std::string(),
                            TenantId("a"), "dead");
        SetDeadline(guard.metadata(), Clock::now() - seconds(10));
    }

    eviction::Evictor evictor(namespaces_, policy_, reclaimer_, {&observer_},
                              eviction::Diagnostics{}, [] { return true; });
    const auto first = evictor.Run(
        Scope::All(), ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});
    EXPECT_EQ(first.evicted_objects, 0u);
    EXPECT_EQ(first.SkipCount(SkipReason::kPermanentlyUnavailable), 1u);
    EXPECT_EQ(first.suppressed_permanently_unavailable, 1u);
    EXPECT_TRUE(observer_.applied.empty());

    const auto second = evictor.Run(
        Scope::All(), ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});
    EXPECT_EQ(second.filed, 0u);
    EXPECT_EQ(second.visited, 0u);
}

// Without the window, a permanent candidate whose index rank lags its real
// lease is only discovered as "moved": it costs a re-file and a later revisit.
TEST_F(EvictorTest, StaleIndexPermanentCandidateIsMovedWithoutTheWindow) {
    metadata::Namespace& ns = namespaces_.GetOrCreate(TenantId("a"));
    const auto now = Clock::now();
    auto dying = std::make_shared<OffsetBufferAllocator>(
        "dead_segment", 0x500000000ULL, 1 << 20, "dead_segment");
    PublishStaleIndexKey(ns, "dead", dying, now - seconds(10), now);
    dying.reset();

    eviction::Evictor evictor(namespaces_, policy_, reclaimer_, {&observer_});
    const auto report = evictor.Run(
        Scope::All(), ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});

    EXPECT_EQ(report.filed, 1u);
    EXPECT_EQ(report.visited, 1u);
    EXPECT_EQ(report.SkipCount(SkipReason::kMoved), 1u);
    EXPECT_EQ(report.SkipCount(SkipReason::kPermanentlyUnavailable), 0u);
    EXPECT_EQ(report.suppressed_permanently_unavailable, 0u);
    EXPECT_EQ(report.evicted_objects, 0u);
}

// The window takes the same key out on first contact, before the moved check:
// no re-file and no later revisit, and its metadata stays for offboarding.
TEST_F(EvictorTest, StaleIndexPermanentCandidateIsSuppressedFirstWithTheWindow) {
    metadata::Namespace& ns = namespaces_.GetOrCreate(TenantId("a"));
    const auto now = Clock::now();
    auto dying = std::make_shared<OffsetBufferAllocator>(
        "dead_segment", 0x500000000ULL, 1 << 20, "dead_segment");
    PublishStaleIndexKey(ns, "dead", dying, now - seconds(10), now);
    dying.reset();

    eviction::Evictor evictor(namespaces_, policy_, reclaimer_, {&observer_},
                              eviction::Diagnostics{},
                              [] { return true; });
    const auto first = evictor.Run(
        Scope::All(), ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});
    EXPECT_EQ(first.filed, 1u);
    EXPECT_EQ(first.visited, 1u);
    EXPECT_EQ(first.SkipCount(SkipReason::kMoved), 0u);
    EXPECT_EQ(first.SkipCount(SkipReason::kPermanentlyUnavailable), 1u);
    EXPECT_EQ(first.suppressed_permanently_unavailable, 1u);
    EXPECT_EQ(first.evicted_objects, 0u);
    // The suggestion changed only the derived index.
    EXPECT_TRUE(ns.objects.Read("dead").has_value());

    const auto second = evictor.Run(
        Scope::All(), ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});
    EXPECT_EQ(second.filed, 0u);
    EXPECT_EQ(second.visited, 0u);
}

// A sweep that starts after the run did still applies to the candidates the
// run has not reached yet: the window is asked per held key, not per run.
TEST_F(EvictorTest, WindowOpeningDuringARunSuppressesLaterCandidates) {
    metadata::Namespace& ns = namespaces_.GetOrCreate(TenantId("a"));
    // Filed earlier than the stale key below, and skipped by the policy, which
    // flips the window before the run reaches "dead".
    Publish("a", "healthy", seconds(10));
    policy_.soft_pinned.insert("healthy");

    const auto now = Clock::now();
    auto dying = std::make_shared<OffsetBufferAllocator>(
        "dead_segment", 0x500000000ULL, 1 << 20, "dead_segment");
    PublishStaleIndexKey(ns, "dead", dying, now - seconds(2), now);
    dying.reset();

    std::atomic<bool> window{false};
    FlipWindowOnSkip observer(window);
    eviction::Evictor evictor(namespaces_, policy_, reclaimer_, {&observer},
                              eviction::Diagnostics{}, [&window] {
                                  return window.load(std::memory_order_acquire);
                              });
    const auto report = evictor.Run(
        Scope::All(), ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});

    EXPECT_EQ(report.SkipCount(SkipReason::kSoftPinned), 1u);
    EXPECT_EQ(report.SkipCount(SkipReason::kPermanentlyUnavailable), 1u);
    EXPECT_EQ(report.SkipCount(SkipReason::kMoved), 0u);
    EXPECT_TRUE(ns.objects.Read("dead").has_value());
}

// The window only takes permanently unavailable memory sets. Live replicas,
// mixed sets, recoverable states and live group members keep the normal lease
// and group path.
TEST_F(EvictorTest, WindowDoesNotSuppressRecoverableOrLiveCandidates) {
    metadata::Namespace& ns = namespaces_.GetOrCreate(TenantId("a"));
    const auto now = Clock::now();
    Publish("a", "healthy", seconds(10));
    Publish("a", "lease_live", -seconds(10));
    Publish("a", "group_live", seconds(10), "g");

    // One permanently unavailable replica beside a live one: still holds a
    // live COMPLETE memory replica, so it is not suppressed.
    {
        auto guard = ns.objects.WriteKey("mixed");
        std::vector<Replica> replicas;
        replicas.push_back(DeadMemoryReplica());
        replicas.emplace_back(allocator_->allocate(128),
                              ReplicaStatus::COMPLETE);
        (void)guard.Publish(UUID{1, 2}, Clock::time_point{}, 128,
                            std::move(replicas), std::nullopt, false,
                            ObjectDataType::UNKNOWN, std::string(),
                            TenantId("a"), "mixed");
        SetDeadline(guard.metadata(), now - seconds(10));
    }
    // A live replica held by a reference count: recoverable, never suppressed.
    {
        auto guard = ns.objects.WriteKey("busy");
        Replica replica(allocator_->allocate(128), ReplicaStatus::COMPLETE);
        replica.inc_refcnt();
        std::vector<Replica> replicas;
        replicas.push_back(std::move(replica));
        (void)guard.Publish(UUID{1, 2}, Clock::time_point{}, 128,
                            std::move(replicas), std::nullopt, false,
                            ObjectDataType::UNKNOWN, std::string(),
                            TenantId("a"), "busy");
        SetDeadline(guard.metadata(), now - seconds(10));
    }

    eviction::Evictor evictor(namespaces_, policy_, reclaimer_, {&observer_},
                              eviction::Diagnostics{},
                              [] { return true; });
    const auto report = evictor.Run(
        Scope::All(), ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});

    EXPECT_EQ(report.SkipCount(SkipReason::kPermanentlyUnavailable), 0u);
    EXPECT_EQ(report.suppressed_permanently_unavailable, 0u);
    EXPECT_GE(report.SkipCount(SkipReason::kLeaseActive), 1u);
    EXPECT_GE(report.SkipCount(SkipReason::kNothingToRelease), 1u);
    EXPECT_GE(report.evicted_objects, 1u);
    EXPECT_TRUE(ns.objects.Read("lease_live").has_value());
    EXPECT_TRUE(ns.objects.Read("busy").has_value());
}

TEST_F(EvictorTest, MetricsLeaveAPoolRunToItsCaller) {
    Publish("a", "k1", seconds(10));
    Publish("b", "k2", seconds(8));
    auto& metrics = MasterMetricManager::instance();
    const int64_t attempts = metrics.get_mem_eviction_attempts();

    eviction::MetricsObserver observer;
    eviction::Evictor evictor(namespaces_, policy_, reclaimer_, {&observer});
    const auto report = evictor.Run(
        Scope::All(), ObjectGoal{.target_ratio = 1.0, .min_ratio = 1.0});

    EXPECT_EQ(report.evicted_objects, 2u);
    EXPECT_EQ(metrics.get_mem_eviction_attempts(), attempts);
}

}  // namespace
}  // namespace mooncake

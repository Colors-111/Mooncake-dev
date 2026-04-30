#include "master_service.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "types.h"

namespace mooncake::test {

// Helper: blake3-style 64-char hex digest strings for use as prefix hashes.
static const std::string kPrefixHashA =
    "13b825898e41332c2beeae39b161d50994fd2ea4af86f4b95fb5dd0e66893882";
static const std::string kPrefixHashB =
    "27c936a0b7f844d1a3e5c0d2b9f7e6a4d1c8b3a7f0e5d4c2b1a9f8e7d6c5b4a3";
static const std::string kPrefixHashC =
    "4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b";
static const std::string kPrefixHashD =
    "5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f";

class RadixTreeTest : public ::testing::Test {
   protected:
    void SetUp() override {
        google::InitGoogleLogging("RadixTreeTest");
        FLAGS_logtostderr = true;
    }

    void TearDown() override { google::ShutdownGoogleLogging(); }

    static constexpr size_t kDefaultSegmentBase = 0x300000000;
    static constexpr size_t kDefaultSegmentSize = 1024 * 1024 * 16;

    // Create a MasterService with radix tree enabled.
    std::unique_ptr<MasterService> MakeServiceWithRadixTree() {
        auto config = MasterServiceConfig::builder()
                          .set_enable_radix_tree(true)
                          .build();
        return std::make_unique<MasterService>(config);
    }

    // Create a MasterService with radix tree disabled (default).
    std::unique_ptr<MasterService> MakeServiceWithoutRadixTree() {
        auto config = MasterServiceConfig::builder()
                          .set_enable_radix_tree(false)
                          .build();
        return std::make_unique<MasterService>(config);
    }

    // Mount a simple segment for object storage.
    struct MountedSegmentContext {
        UUID segment_id;
        UUID client_id;
    };

    MountedSegmentContext PrepareSimpleSegment(
        MasterService& service, std::string name = "test_segment",
        size_t base = kDefaultSegmentBase,
        size_t size = kDefaultSegmentSize) const {
        Segment segment;
        segment.id = generate_uuid();
        segment.name = std::move(name);
        segment.base = base;
        segment.size = size;
        segment.te_endpoint = segment.name;
        UUID client_id = generate_uuid();
        auto mount_result = service.MountSegment(segment, client_id);
        EXPECT_TRUE(mount_result.has_value());
        return {.segment_id = segment.id, .client_id = client_id};
    }

    // Put an object with optional parent_block_hash for radix tree
    // registration.
    std::string PutObject(MasterService& service,
                          const UUID& client_id,
                          const std::string& key,
                          const std::string& parent_block_hash = "",
                          size_t slice_length = 1024) const {
        ReplicateConfig config;
        config.replica_num = 1;
        config.parent_block_hash = parent_block_hash;

        auto put_start =
            service.PutStart(client_id, key, slice_length, config);
        EXPECT_TRUE(put_start.has_value())
            << "PutStart failed for key: " << key;
        auto put_end =
            service.PutEnd(client_id, key, ReplicaType::MEMORY);
        EXPECT_TRUE(put_end.has_value())
            << "PutEnd failed for key: " << key;
        return key;
    }

    // Helper to check that a vector contains a specific element.
    template <typename T>
    static bool VectorContains(const std::vector<T>& vec, const T& val) {
        return std::find(vec.begin(), vec.end(), val) != vec.end();
    }

    // Helper: wait for lease to expire then remove.
    void WaitAndRemove(MasterService& service,
                       const std::string& key,
                       uint64_t kv_lease_ttl_ms = 50) const {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kv_lease_ttl_ms));
        auto result = service.Remove(key);
        EXPECT_TRUE(result.has_value())
            << "Remove failed for key: " << key;
    }
};

// ========================================================================
// 1. Basic CRUD
// ========================================================================

TEST_F(RadixTreeTest, RegisterSingleNode) {
    auto service = MakeServiceWithRadixTree();
    [[maybe_unused]] auto ctx = PrepareSimpleSegment(*service);

    auto reg_result = service->RegisterRadixTreeNode(
        kPrefixHashA, /*parent_prefix_hash=*/"", {kPrefixHashA + "_0_k"});
    ASSERT_TRUE(reg_result.has_value());

    auto query_result = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(query_result.has_value());
    EXPECT_EQ(1u, query_result->keys.size());
    EXPECT_TRUE(VectorContains(query_result->keys, kPrefixHashA + "_0_k"));
    EXPECT_TRUE(query_result->parent_prefix_hash.empty());
    EXPECT_TRUE(query_result->children_prefix_hashes.empty());
}

TEST_F(RadixTreeTest, RegisterMultipleKeysSamePrefix) {
    auto service = MakeServiceWithRadixTree();
    [[maybe_unused]] auto ctx = PrepareSimpleSegment(*service);

    std::vector<std::string> keys = {kPrefixHashA + "_0_k",
                                      kPrefixHashA + "_0_v",
                                      kPrefixHashA + "_1_k"};
    auto reg_result = service->RegisterRadixTreeNode(
        kPrefixHashA, /*parent_prefix_hash=*/"", keys);
    ASSERT_TRUE(reg_result.has_value());

    auto query_result = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(query_result.has_value());
    EXPECT_EQ(3u, query_result->keys.size());
    for (const auto& key : keys) {
        EXPECT_TRUE(VectorContains(query_result->keys, key));
    }
}

TEST_F(RadixTreeTest, RegisterWithParent) {
    auto service = MakeServiceWithRadixTree();
    [[maybe_unused]] auto ctx = PrepareSimpleSegment(*service);

    // Register A as root
    auto reg_a = service->RegisterRadixTreeNode(
        kPrefixHashA, /*parent_prefix_hash=*/"", {kPrefixHashA + "_0_k"});
    ASSERT_TRUE(reg_a.has_value());

    // Register B with parent A
    auto reg_b = service->RegisterRadixTreeNode(
        kPrefixHashB, kPrefixHashA, {kPrefixHashB + "_0_k"});
    ASSERT_TRUE(reg_b.has_value());

    // Verify A has B as a child
    auto query_a = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(query_a.has_value());
    EXPECT_TRUE(VectorContains(query_a->children_prefix_hashes, kPrefixHashB));

    // Verify B has A as parent
    auto query_b = service->GetKeysByPrefix(kPrefixHashB);
    ASSERT_TRUE(query_b.has_value());
    EXPECT_EQ(query_b->parent_prefix_hash, kPrefixHashA);
}

TEST_F(RadixTreeTest, UnregisterKeyViaRemove) {
    auto service = MakeServiceWithRadixTree();
    auto ctx = PrepareSimpleSegment(*service);
    const UUID client_id = generate_uuid();

    std::string key = kPrefixHashA + "_0_k";
    PutObject(*service, ctx.client_id, key, /*parent_block_hash=*/"");

    // Verify key is in radix tree
    auto query = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(query.has_value());

    // Remove the object (wait for lease to expire)
    WaitAndRemove(*service, key);

    // Verify radix tree is cleaned up
    auto query_after = service->GetKeysByPrefix(kPrefixHashA);
    EXPECT_FALSE(query_after.has_value());
    EXPECT_EQ(ErrorCode::OBJECT_NOT_FOUND, query_after.error());
}

TEST_F(RadixTreeTest, UnregisterLastKeyCascadeDelete) {
    auto service = MakeServiceWithRadixTree();
    auto ctx = PrepareSimpleSegment(*service);
    UUID client_id = generate_uuid();

    // Build A->B: put A (root) and B with parent A
    PutObject(*service, client_id, kPrefixHashA + "_0_k", "");
    PutObject(*service, client_id, kPrefixHashB + "_0_k", kPrefixHashA);

    // Verify A has B as child
    auto query_a = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(query_a.has_value());
    EXPECT_TRUE(VectorContains(query_a->children_prefix_hashes, kPrefixHashB));

    // Remove B's only key -> B becomes empty leaf, cascade deletes B,
    // then A becomes empty (only key, no children), cascade deletes A.
    WaitAndRemove(*service, kPrefixHashB + "_0_k");

    auto query_b_after = service->GetKeysByPrefix(kPrefixHashB);
    EXPECT_FALSE(query_b_after.has_value());

    auto query_a_after = service->GetKeysByPrefix(kPrefixHashA);
    EXPECT_FALSE(query_a_after.has_value());
}

TEST_F(RadixTreeTest, UnregisterNonLastKeyNoCascade) {
    auto service = MakeServiceWithRadixTree();
    auto ctx = PrepareSimpleSegment(*service);
    UUID client_id = generate_uuid();

    // Put two objects under the same prefix hash
    std::string key1 = kPrefixHashA + "_0_k";
    std::string key2 = kPrefixHashA + "_0_v";
    PutObject(*service, client_id, key1);
    PutObject(*service, client_id, key2);

    // Verify both keys are in radix tree
    auto query = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(query.has_value());
    EXPECT_EQ(2u, query->keys.size());

    // Remove one key
    WaitAndRemove(*service, key1);

    // Node should still exist with the remaining key
    auto query_after = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(query_after.has_value());
    EXPECT_EQ(1u, query_after->keys.size());
    EXPECT_TRUE(VectorContains(query_after->keys, key2));
}

TEST_F(RadixTreeTest, GetKeysByPrefixNotFound) {
    auto service = MakeServiceWithRadixTree();
    [[maybe_unused]] auto ctx = PrepareSimpleSegment(*service);

    auto query = service->GetKeysByPrefix(kPrefixHashA);
    EXPECT_FALSE(query.has_value());
    EXPECT_EQ(ErrorCode::OBJECT_NOT_FOUND, query.error());
}

// ========================================================================
// 2. Cross-shard
// ========================================================================

TEST_F(RadixTreeTest, ParentChildDifferentShards) {
    auto service = MakeServiceWithRadixTree();
    [[maybe_unused]] auto ctx = PrepareSimpleSegment(*service);

    // Register A as root, B as child of A (different shards expected)
    auto reg_a = service->RegisterRadixTreeNode(
        kPrefixHashA, "", {kPrefixHashA + "_0_k"});
    ASSERT_TRUE(reg_a.has_value());
    auto reg_b = service->RegisterRadixTreeNode(
        kPrefixHashB, kPrefixHashA, {kPrefixHashB + "_0_k"});
    ASSERT_TRUE(reg_b.has_value());

    // Verify parent-child relationship
    auto query_a = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(query_a.has_value());
    EXPECT_TRUE(VectorContains(query_a->children_prefix_hashes, kPrefixHashB));

    auto query_b = service->GetKeysByPrefix(kPrefixHashB);
    ASSERT_TRUE(query_b.has_value());
    EXPECT_EQ(kPrefixHashA, query_b->parent_prefix_hash);
}

TEST_F(RadixTreeTest, CascadeCleanupCrossShard) {
    auto service = MakeServiceWithRadixTree();
    auto ctx = PrepareSimpleSegment(*service);
    UUID client_id = generate_uuid();

    // Build A (root) -> B (leaf), both with single keys
    PutObject(*service, client_id, kPrefixHashA + "_0_k", "");
    PutObject(*service, client_id, kPrefixHashB + "_0_k", kPrefixHashA);

    // Verify B is a child of A
    auto query_a = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(query_a.has_value());
    EXPECT_TRUE(VectorContains(query_a->children_prefix_hashes, kPrefixHashB));

    // Remove B's key -> cascade cleanup: B removed, then A removed
    WaitAndRemove(*service, kPrefixHashB + "_0_k");

    auto query_b_after = service->GetKeysByPrefix(kPrefixHashB);
    EXPECT_FALSE(query_b_after.has_value());

    auto query_a_after = service->GetKeysByPrefix(kPrefixHashA);
    EXPECT_FALSE(query_a_after.has_value());
}

// ========================================================================
// 3. Cascade Depth
// ========================================================================

TEST_F(RadixTreeTest, CascadeDepth3) {
    auto service = MakeServiceWithRadixTree();
    auto ctx = PrepareSimpleSegment(*service);
    UUID client_id = generate_uuid();

    // Build A -> B -> C, each with exactly one registered key
    PutObject(*service, client_id, kPrefixHashA + "_0_k", "");
    PutObject(*service, client_id, kPrefixHashB + "_0_k", kPrefixHashA);
    PutObject(*service, client_id, kPrefixHashC + "_0_k", kPrefixHashB);

    // Verify structure
    auto qa = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(qa.has_value());
    EXPECT_TRUE(VectorContains(qa->children_prefix_hashes, kPrefixHashB));

    auto qb = service->GetKeysByPrefix(kPrefixHashB);
    ASSERT_TRUE(qb.has_value());
    EXPECT_TRUE(VectorContains(qb->children_prefix_hashes, kPrefixHashC));

    // Remove C's key -> cascade: C removed, B (1 key, 0 children) removed,
    // A (1 key, 0 children) removed
    WaitAndRemove(*service, kPrefixHashC + "_0_k");

    auto qc_after = service->GetKeysByPrefix(kPrefixHashC);
    EXPECT_FALSE(qc_after.has_value());

    auto qb_after = service->GetKeysByPrefix(kPrefixHashB);
    EXPECT_FALSE(qb_after.has_value());

    auto qa_after = service->GetKeysByPrefix(kPrefixHashA);
    EXPECT_FALSE(qa_after.has_value());
}

TEST_F(RadixTreeTest, CascadeStopsAtNonEmpty) {
    auto service = MakeServiceWithRadixTree();
    auto ctx = PrepareSimpleSegment(*service);
    UUID client_id = generate_uuid();

    // Build A -> B -> C, but B has two keys so it survives cascade
    PutObject(*service, client_id, kPrefixHashA + "_0_k", "");
    PutObject(*service, client_id, kPrefixHashB + "_0_k", kPrefixHashA);
    PutObject(*service, client_id, kPrefixHashB + "_0_v", kPrefixHashA);
    PutObject(*service, client_id, kPrefixHashC + "_0_k", kPrefixHashB);

    // Remove C's key. C becomes empty and is removed. But B still
    // has 2 keys, so cascade stops.
    WaitAndRemove(*service, kPrefixHashC + "_0_k");

    // C should be gone
    auto qc = service->GetKeysByPrefix(kPrefixHashC);
    EXPECT_FALSE(qc.has_value());

    // B should still exist (still has 2 keys)
    auto qb = service->GetKeysByPrefix(kPrefixHashB);
    ASSERT_TRUE(qb.has_value());
    EXPECT_EQ(2u, qb->keys.size());

    // A should still exist (B is still its child)
    auto qa = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(qa.has_value());
    EXPECT_TRUE(VectorContains(qa->children_prefix_hashes, kPrefixHashB));
}

TEST_F(RadixTreeTest, CascadeStopsAtNodeWithOtherChildren) {
    auto service = MakeServiceWithRadixTree();
    auto ctx = PrepareSimpleSegment(*service);
    UUID client_id = generate_uuid();

    // Build A -> B and A -> D (two children), each with one key
    PutObject(*service, client_id, kPrefixHashA + "_0_k", "");
    PutObject(*service, client_id, kPrefixHashB + "_0_k", kPrefixHashA);
    PutObject(*service, client_id, kPrefixHashD + "_0_k", kPrefixHashA);

    // Verify A has both B and D as children
    auto qa = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(qa.has_value());
    EXPECT_TRUE(VectorContains(qa->children_prefix_hashes, kPrefixHashB));
    EXPECT_TRUE(VectorContains(qa->children_prefix_hashes, kPrefixHashD));

    // Remove B's key. B becomes empty and is removed. But A still
    // has D as a child and its own key, so cascade stops.
    WaitAndRemove(*service, kPrefixHashB + "_0_k");

    // B should be gone
    auto qb = service->GetKeysByPrefix(kPrefixHashB);
    EXPECT_FALSE(qb.has_value());

    // D should still exist
    auto qd = service->GetKeysByPrefix(kPrefixHashD);
    ASSERT_TRUE(qd.has_value());

    // A should still exist (still has D as child and its own key)
    auto qa_after = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(qa_after.has_value());
    EXPECT_TRUE(
        VectorContains(qa_after->children_prefix_hashes, kPrefixHashD));
    EXPECT_FALSE(
        VectorContains(qa_after->children_prefix_hashes, kPrefixHashB));
}

// ========================================================================
// 4. Prefix Extraction
// ========================================================================

TEST_F(RadixTreeTest, ExtractPrefixHashMLA) {
    // MLA key: 64-char hash with no underscore -> full key is the prefix
    auto service = MakeServiceWithRadixTree();
    auto ctx = PrepareSimpleSegment(*service);
    UUID client_id = generate_uuid();

    std::string mla_key = kPrefixHashA;  // 64-char hash, no underscore

    PutObject(*service, client_id, mla_key);

    // Query with the full key as prefix -- should find it
    auto query = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(query.has_value());
    EXPECT_TRUE(VectorContains(query->keys, mla_key));
}

TEST_F(RadixTreeTest, ExtractPrefixHashMHA) {
    // MHA key: "<64-char-hash>_0_k" -> prefix is first 64 chars
    auto service = MakeServiceWithRadixTree();
    auto ctx = PrepareSimpleSegment(*service);
    UUID client_id = generate_uuid();

    std::string mha_key = kPrefixHashA + "_0_k";

    PutObject(*service, client_id, mha_key);

    // Query with the 64-char prefix -- should find the MHA key
    auto query = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(query.has_value());
    EXPECT_TRUE(VectorContains(query->keys, mha_key));
}

TEST_F(RadixTreeTest, ExtractPrefixHashNoUnderscore) {
    // Short key with no underscore -> full key is the prefix
    auto service = MakeServiceWithRadixTree();
    auto ctx = PrepareSimpleSegment(*service);
    UUID client_id = generate_uuid();

    std::string short_key = "shortkey";

    PutObject(*service, client_id, short_key);

    // Query with the full key as prefix -- should find it
    auto query = service->GetKeysByPrefix("shortkey");
    ASSERT_TRUE(query.has_value());
    EXPECT_TRUE(VectorContains(query->keys, short_key));
}

// ========================================================================
// 5. Shell Node
// ========================================================================

TEST_F(RadixTreeTest, ChildRegistersBeforeParent) {
    auto service = MakeServiceWithRadixTree();
    [[maybe_unused]] auto ctx = PrepareSimpleSegment(*service);

    // Register B with parent A before A is registered.
    // This should create a shell node for A (no keys, just child pointer).
    auto reg_b = service->RegisterRadixTreeNode(
        kPrefixHashB, kPrefixHashA, {kPrefixHashB + "_0_k"});
    ASSERT_TRUE(reg_b.has_value());

    // A should exist as a shell node (has B as child, but no keys)
    auto query_a = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(query_a.has_value());
    EXPECT_TRUE(query_a->keys.empty());
    EXPECT_TRUE(VectorContains(query_a->children_prefix_hashes, kPrefixHashB));

    // B should have A as parent
    auto query_b = service->GetKeysByPrefix(kPrefixHashB);
    ASSERT_TRUE(query_b.has_value());
    EXPECT_EQ(kPrefixHashA, query_b->parent_prefix_hash);
}

TEST_F(RadixTreeTest, ParentRegistersAfterShell) {
    auto service = MakeServiceWithRadixTree();
    [[maybe_unused]] auto ctx = PrepareSimpleSegment(*service);

    // First: register B with parent A, creating shell A
    auto reg_b = service->RegisterRadixTreeNode(
        kPrefixHashB, kPrefixHashA, {kPrefixHashB + "_0_k"});
    ASSERT_TRUE(reg_b.has_value());

    // A is a shell: exists but has no keys
    auto query_a = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(query_a.has_value());
    EXPECT_TRUE(query_a->keys.empty());

    // Now register a key for A -- it's no longer a shell
    auto reg_a = service->RegisterRadixTreeNode(
        kPrefixHashA, "", {kPrefixHashA + "_0_k"});
    ASSERT_TRUE(reg_a.has_value());

    // A should now have a key and still have B as child
    auto query_a2 = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(query_a2.has_value());
    EXPECT_TRUE(VectorContains(query_a2->keys, kPrefixHashA + "_0_k"));
    EXPECT_TRUE(VectorContains(query_a2->children_prefix_hashes, kPrefixHashB));
}

// ========================================================================
// 6. Deferred Pattern
// ========================================================================

TEST_F(RadixTreeTest, DeferredRegisterViaPutStart) {
    auto service = MakeServiceWithRadixTree();
    auto ctx = PrepareSimpleSegment(*service);
    UUID client_id = generate_uuid();

    // PutStart with parent_block_hash should register in radix tree
    std::string key = kPrefixHashB + "_0_k";
    {
        ReplicateConfig config;
        config.replica_num = 1;
        config.parent_block_hash = kPrefixHashA;
        auto put_start = service->PutStart(client_id, key, 1024, config);
        ASSERT_TRUE(put_start.has_value());
        auto put_end =
            service->PutEnd(client_id, key, ReplicaType::MEMORY);
        ASSERT_TRUE(put_end.has_value());
    }

    // Verify B is in radix tree with A as parent
    auto query_b = service->GetKeysByPrefix(kPrefixHashB);
    ASSERT_TRUE(query_b.has_value());
    EXPECT_TRUE(VectorContains(query_b->keys, key));
    EXPECT_EQ(kPrefixHashA, query_b->parent_prefix_hash);

    // Verify A exists (shell or with child)
    auto query_a = service->GetKeysByPrefix(kPrefixHashA);
    ASSERT_TRUE(query_a.has_value());
    EXPECT_TRUE(VectorContains(query_a->children_prefix_hashes, kPrefixHashB));
}

TEST_F(RadixTreeTest, DeferredUnregisterViaRemove) {
    auto service = MakeServiceWithRadixTree();
    auto ctx = PrepareSimpleSegment(*service);
    UUID client_id = generate_uuid();

    // Put an object with parent_block_hash
    std::string key = kPrefixHashB + "_0_k";
    {
        ReplicateConfig config;
        config.replica_num = 1;
        config.parent_block_hash = kPrefixHashA;
        auto put_start = service->PutStart(client_id, key, 1024, config);
        ASSERT_TRUE(put_start.has_value());
        auto put_end =
            service->PutEnd(client_id, key, ReplicaType::MEMORY);
        ASSERT_TRUE(put_end.has_value());
    }

    // Verify it's registered
    auto query_b = service->GetKeysByPrefix(kPrefixHashB);
    ASSERT_TRUE(query_b.has_value());

    // Remove the object (wait for lease to expire first)
    WaitAndRemove(*service, key);

    // B's node should be gone (was the only key), and A's shell
    // should also be gone (cascade cleanup: A had no keys and
    // lost its only child).
    auto query_b_after = service->GetKeysByPrefix(kPrefixHashB);
    EXPECT_FALSE(query_b_after.has_value());

    auto query_a_after = service->GetKeysByPrefix(kPrefixHashA);
    EXPECT_FALSE(query_a_after.has_value());
}

// ========================================================================
// 7. Radix Tree Disabled
// ========================================================================

TEST_F(RadixTreeTest, DisabledByDefault) {
    auto service = MakeServiceWithoutRadixTree();
    [[maybe_unused]] auto ctx = PrepareSimpleSegment(*service);

    // RegisterRadixTreeNode should return INVALID_PARAMS when disabled
    auto reg_result = service->RegisterRadixTreeNode(
        kPrefixHashA, "", {kPrefixHashA + "_0_k"});
    EXPECT_FALSE(reg_result.has_value());
    EXPECT_EQ(ErrorCode::INVALID_PARAMS, reg_result.error());

    // GetKeysByPrefix should also return INVALID_PARAMS when disabled
    auto query_result = service->GetKeysByPrefix(kPrefixHashA);
    EXPECT_FALSE(query_result.has_value());
    EXPECT_EQ(ErrorCode::INVALID_PARAMS, query_result.error());
}

}  // namespace mooncake::test

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#pragma once

// EvictionIndex: the order in which eviction visits one namespace's objects.
// The namespace owns one and tells it about every key the route hands out: a
// read as its read guard is released, a write as its write guard is. Eviction
// reads it through a cursor, oldest-ranked entry first.
//
// An entry is what eviction takes together: one key, or the members of a
// group, whose shared lease makes them due together. The index files a group
// as one entry, so a walk meets each group once.
//
// An index may rank keys however it likes (by lease deadline, by access
// frequency, by size) as long as it keeps one rule: the rank a key is filed at
// is never later than the rank it really has. A read may raise a key's real
// rank without telling the index, which then catches up when the cursor's
// StillAt is asked about the key under its write lock: that release files the
// key again at its real rank. So an index only has to watch reads when a read
// can lower a rank, and says so at construction; the route skips the read hook
// for every other index.
//
// Lock order: whatever lock an index takes is a leaf under a key lock, and
// nothing takes a key lock while holding it.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

#include "metadata/object_route.h"

namespace mooncake {
namespace metadata {

class EvictionIndex {
   public:
    using Clock = std::chrono::system_clock;
    // Lower ranks are evicted first.
    using Rank = int64_t;

    // What the index filed at one rank: one key, or a group's members. The
    // keys view the cursor that handed the entry out, until it is asked for
    // the next entry or destroyed.
    struct Entry {
        std::span<const std::string_view> keys;
        Rank rank;
        // Which incarnation of a group the keys were filed in, as the index
        // numbers them; 0 for a key on its own.
        uint64_t group = 0;
    };

    // One walk over the filed entries, lowest rank first. What it hands out
    // is a snapshot: a key may be refiled or gone by the time it is acted on.
    class Cursor {
       public:
        virtual ~Cursor() = default;
        // The next filed entry; nullopt once the walk is done.
        virtual std::optional<Entry> Next() = 0;
        // Whether the key `guard` holds, one of `entry`'s, which this cursor
        // handed out, still stands where the entry filed it: at its rank, in
        // the same incarnation of its group. Asked under the key's write lock,
        // so a key that moved on is refiled where it belongs as that lock is
        // let go.
        [[nodiscard]] virtual bool StillAt(const route::WriteGuard& guard,
                                           const Entry& entry) const = 0;
        // How many keys were filed when the walk began.
        [[nodiscard]] virtual size_t Total() const = 0;
        // How reading this walk's buckets went, for diagnostics. A walk that
        // does not read in buckets leaves it zero.
        struct ReadTiming {
            uint64_t bucket_reads = 0;
            uint64_t read_ns = 0;
            uint64_t max_read_ns = 0;
        };
        [[nodiscard]] virtual ReadTiming Timing() const { return {}; }
    };

    explicit EvictionIndex(bool observes_reads)
        : observes_reads_(observes_reads) {}
    EvictionIndex(const EvictionIndex&) = delete;
    EvictionIndex& operator=(const EvictionIndex&) = delete;
    virtual ~EvictionIndex() = default;

    // Called as a read guard on the key is released, with the key's lock held
    // shared, possibly by several readers at once; only when the index
    // observes reads. It must not touch the slot's owner state, which the
    // shared lock does not guard.
    virtual void OnRead(const route::ReadGuard& guard) = 0;
    // Called as a write guard on the key is released, with the key's lock
    // held exclusively: files, refiles or drops the key to match the object
    // it holds now.
    virtual void OnWrite(const route::WriteGuard& guard) = 0;
    // A walk over the keys that may be due by `now`.
    [[nodiscard]] virtual std::unique_ptr<Cursor> Scan(
        Clock::time_point now) const = 0;

    // Told under a key's write lock that the policy found nothing to release
    // for its object. The index may take the key out of its candidates for
    // good, but only on a permanent condition: every COMPLETE memory replica
    // of the object has lost its allocator or segment lifetime. A replica that
    // is merely unreadable, awaiting remount or held by a reference count may
    // recover, so it never qualifies. Returns whether the key was suppressed;
    // an index that keeps no such state returns false and does nothing.
    //
    // The suppression is derived index state only: it touches no replica, no
    // metadata and no eviction result. A later write that finds a live
    // COMPLETE memory replica clears it (see OnWrite).
    virtual bool SuppressPermanentlyUnavailable(
        const route::WriteGuard& /*guard*/) {
        return false;
    }

    // Whether OnRead must be called.
    [[nodiscard]] bool observes_reads() const { return observes_reads_; }

   private:
    const bool observes_reads_;
};

}  // namespace metadata
}  // namespace mooncake

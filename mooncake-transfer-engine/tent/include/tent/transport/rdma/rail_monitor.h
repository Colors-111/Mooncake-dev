// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENT_RAIL_MONITOR_H
#define TENT_RAIL_MONITOR_H

#include "tent/common/config.h"
#include "tent/common/status.h"
#include "tent/runtime/topology.h"

namespace mooncake {
namespace tent {
// Not thread-safe. Each RDMA worker owns its own RailMonitor via
// WorkerContext::rails, so markFailed / markRecovered / available / load
// all execute on a single worker thread. Do not share across threads
// without adding external synchronization.
class RailMonitor {
    const static size_t kMaxNuma = 16;

    // Upper bound on the exponential-backoff cooldown; prevents the
    // per-rail pause from growing without bound under repeated failure.
    static constexpr std::chrono::seconds kMaxCooldown{300};

   public:
    // Config keys. Exposed as constants so callers (and docs) reference
    // a single source of truth instead of duplicating the string path.
    static constexpr const char *kCfgErrorThreshold =
        "transports/rdma/rail_error_threshold";
    static constexpr const char *kCfgErrorWindowSecs =
        "transports/rdma/rail_error_window_secs";
    static constexpr const char *kCfgCooldownSecs =
        "transports/rdma/rail_cooldown_secs";
    static constexpr const char *kCfgProbeIntervalSecs =
        "transports/rdma/rail_probe_interval_secs";

   public:
    RailMonitor() = default;

    ~RailMonitor() = default;

    RailMonitor(const RailMonitor &) = delete;
    RailMonitor &operator=(const RailMonitor &) = delete;

   public:
    // The shared topology references keep segment snapshots alive while rail
    // failure handling uses their derived mappings.
    // rail_topo_json: optional JSON string describing rail topology.
    // conf: optional Config pointer; when non-null, overrides the default
    //   error_threshold / error_window_secs / cooldown_secs values via
    //   kCfgErrorThreshold / kCfgErrorWindowSecs / kCfgCooldownSecs.
    // A later call with the same NIC/memory layout only refreshes the pins
    // (segment COW snapshots); it does not rebuild rail_states_ or reprint
    // the config banner.
    Status load(std::shared_ptr<const Topology> local,
                std::shared_ptr<const Topology> remote,
                const std::string &rail_topo_json = "",
                const Config *conf = nullptr);

    bool ready() { return ready_; }

    // Pure predicate: true iff the rail is Closed/healthy and directly usable.
    // Does NOT mutate state and does NOT call updateBestMapping, so it is safe
    // to call from updateBestMapping (no recursion) and from any query path.
    bool isAvailable(int local_nic, int remote_nic) const;

    // Mutating admit: the ONLY path that arms a probe/trial. Returns true when
    // a transfer may use this rail -- Closed (no-op), or an exploratory probe /
    // Half-Open trial. Sets probe_in_flight for paused rails so a second probe
    // cannot race the first while it is on the wire. Callers that select a rail
    // but fail to post must call cancelProbe() to clear the flag.
    bool admit(int local_nic, int remote_nic);

    void markFailed(int local_nic, int remote_nic);

    void markRecovered(int local_nic, int remote_nic);

    // Clear an armed probe/trial when a slice selected the rail via admit() but
    // never reached the wire (e.g. endpoint creation failed, HW rejected the
    // post). No-op when no probe is in flight, so callers may invoke it
    // unconditionally on the re-queue path.
    void cancelProbe(int local_nic, int remote_nic);

    int findBestRemoteDevice(int local_nic, int remote_numa);

    const Topology *local() const { return local_.get(); }

    const Topology *remote() const { return remote_.get(); }

   private:
    Status loadFromJson(const std::string &rail_topo_json);

    Status loadDefault();

    void updateBestMapping();

   private:
    bool ready_{false};
    std::shared_ptr<const Topology> local_;
    std::shared_ptr<const Topology> remote_;

    struct PairHash {
        std::size_t operator()(const std::pair<int, int> &p) const noexcept {
            return std::hash<int>()(p.first) ^
                   (std::hash<int>()(p.second) << 1);
        }
    };

    struct RailState {
        int error_count = 0;
        std::chrono::seconds cooldown{0};
        std::chrono::steady_clock::time_point last_error{};
        std::chrono::steady_clock::time_point resume_time{};
        // Last time admit() armed a probe/trial. Throttles the probe rate
        // during Open and defers the first probe by one probe_interval_ after a
        // pause.
        std::chrono::steady_clock::time_point last_probe_time{};
        // Half-Open: the cooldown expired but recovery is not yet proven.
        // admit() arms exactly one trial; a trial success closes the rail
        // (markRecovered), a trial failure escalates the cooldown and re-arms
        // (markFailed). This replaces the old "expiry fully reopens" path,
        // which slammed a still-dead peer with every slice and re-triggered the
        // storm at 30s/60s/120s.
        bool half_open = false;
        // A probe/trial slice is selected/posted and its completion has not yet
        // resolved the rail. admit() refuses to arm a second while this is set,
        // so at most one probe/trial is on the wire per rail (single-threaded
        // worker ownership makes a bool sufficient). Cleared by markFailed /
        // markRecovered on completion, and by cancelProbe() on a pre-wire
        // failure that never reaches the wire.
        bool probe_in_flight = false;

        // Derived: a rail is paused iff a resume_time has been armed. The
        // Half-Open sub-state also has resume_time armed so paused() stays
        // true.
        bool paused() const {
            return resume_time != std::chrono::steady_clock::time_point{};
        }
    };

    std::unordered_map<std::pair<int, int>, RailState, PairHash> rail_states_;
    std::unordered_map<int, int> direct_rails_;  // keep static after loaded
    std::unordered_map<int, int> best_mapping_[kMaxNuma];

    int error_threshold_ = 3;
    std::chrono::seconds error_window_{10};
    std::chrono::seconds cooldown_{30};
    std::chrono::seconds probe_interval_{1};
};

}  // namespace tent
}  // namespace mooncake

#endif  // TENT_RAIL_MONITOR_H

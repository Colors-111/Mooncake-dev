package main

import (
	"context"
	"encoding/json"
	"log"
	"net"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	etcdpb "go.etcd.io/etcd/api/v3/etcdserverpb"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
)

type countingLogWriter struct{ writes atomic.Uint64 }

func TestLeaseDiagnosticsRecordsPayloadsAndBoundsHistory(t *testing.T) {
	d := &leaseTransportDiagnostics{}
	for i := 0; i < 1000; i++ {
		d.HandleRPC(context.Background(), &stats.Begin{})
		d.HandleRPC(context.Background(), &stats.OutPayload{})
		d.HandleRPC(context.Background(), &stats.InPayload{Payload: &etcdpb.LeaseKeepAliveResponse{TTL: 15}})
		d.HandleRPC(context.Background(), &stats.End{Error: status.Error(codes.Unavailable, "broken stream")})
	}
	detail := d.summary()
	for _, field := range []string{"rpc_begins=1000", "requests_sent=1000", "responses_received=1000", "last_response_ttl_sec=15", "transport_events="} {
		if !strings.Contains(detail, field) {
			t.Errorf("missing %q in %s", field, detail)
		}
	}
	if len(detail) > 6000 || strings.Count(detail, "broken stream") > 13 {
		t.Fatalf("transport history is not bounded: %d bytes", len(detail))
	}
}

func TestLeaseTCPDiagnosticsReadsSocketWithoutKeepingItAlive(t *testing.T) {
	if !leaseTCPDiagnosticsSupported() {
		t.Skip("TCP_INFO diagnostics require Linux")
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	client, err := net.Dial("tcp", listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	peer, err := listener.Accept()
	if err != nil {
		t.Fatal(err)
	}
	defer peer.Close()
	probe := leaseTCPProbe{fd: -1}
	snapshot := probe.sample(client.LocalAddr().String(), client.RemoteAddr().String())
	if snapshot.Error != "" || snapshot.Cookie == 0 || snapshot.State != 1 || snapshot.RTOUS == 0 {
		t.Fatalf("missing live TCP evidence: %+v", snapshot)
	}
	wrongCookie := readLeaseTCP(probe.fd, snapshot.Cookie+1)
	if wrongCookie.Error == "" {
		t.Fatal("accepted a reused/mismatched socket cookie")
	}
	if err := client.Close(); err != nil {
		t.Fatal(err)
	}
	if err := peer.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	var b [1]byte
	_, err = peer.Read(b[:])
	if e, ok := err.(net.Error); ok && e.Timeout() {
		t.Fatal("diagnostics kept a reference to a closed socket")
	}
}

func TestLeaseFlightRecorderBoundsSamplesAndNeverWaitsForSampler(t *testing.T) {
	r := &leaseFlightRecorder{}
	for i := 0; i < 1000; i++ {
		r.record(leaseFlightSample{At: time.Unix(int64(i), 0)})
	}
	var samples []leaseFlightSample
	if err := json.Unmarshal([]byte(r.history()), &samples); err != nil {
		t.Fatal(err)
	}
	if len(samples) != 8 || !samples[0].At.Equal(time.Unix(992, 0)) {
		t.Fatalf("unexpected ring: %+v", samples)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	done := make(chan string, 1)
	go func() { done <- r.history() }()
	select {
	case got := <-done:
		if !strings.Contains(got, "busy") {
			t.Fatalf("missing busy marker: %s", got)
		}
	case <-time.After(time.Second):
		t.Fatal("failure report waited for sampler")
	}
}

func TestLeaseRecoveryOnlyReportsAResponseAfterAnOutage(t *testing.T) {
	t0 := time.Unix(1000, 0)
	r := leaseRecoveryTracker{}
	if r.update(t0, 0, 0, t0, 15, "READY") {
		t.Fatal("healthy response reported as recovery")
	}
	// A 15-second lease normally responds about every five seconds.
	if r.update(t0.Add(6*time.Second), 0, 0, t0, 15, "READY") || !r.since.IsZero() {
		t.Fatal("normal 15-second TTL cadence was treated as an outage")
	}
	if r.update(t0.Add(8*time.Second), 1, 1, t0, 15, "CONNECTING") {
		t.Fatal("disconnection reported as recovery")
	}
	if !r.update(t0.Add(9*time.Second), 1, 1, t0.Add(9*time.Second), 15, "READY") {
		t.Fatal("successful response after outage was not reported")
	}
	if r.update(t0.Add(10*time.Second), 1, 1, t0.Add(9*time.Second), 15, "READY") {
		t.Fatal("same recovery reported twice")
	}
	// A failure and successful reconnect can both occur between sample ticks.
	if !r.update(t0.Add(11*time.Second), 2, 2, t0.Add(11*time.Second), 15, "READY") {
		t.Fatal("short outage between ticks was lost")
	}
}

func TestLeaseFlightRecorderSurvivesAnUnavailableSampler(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	d := &leaseTransportDiagnostics{}
	blocked := make(chan struct{})
	release := make(chan struct{})
	defer func() { cancel(); close(release) }()
	r := startLeaseFlightRecorderWithSampler(ctx, 42, d, func() string { return "CONNECTING" }, func() leaseFlightSample {
		close(blocked)
		<-release
		return leaseFlightSample{}
	})
	select {
	case <-blocked:
	case <-time.After(time.Second):
		t.Fatal("sampler did not start")
	}
	responses := make(chan *clientv3.LeaseKeepAliveResponse)
	close(responses)
	code, detail := observeKeepAlive(ctx, ctx, 42, responses, func() string { return "CONNECTING flight_history=" + r.history() })
	if code != -1 || !strings.Contains(detail, "keep alive channel closed") {
		t.Fatal(detail)
	}
}

func TestLeaseTCPDiagnosticsReportsAnInvalidFDAsUnavailable(t *testing.T) {
	// Invalid FDs must be reported as unavailable, never as healthy/zero data.
	snapshot := readLeaseTCP(-1, 0)
	if snapshot.Error == "" {
		t.Fatal("invalid socket reported as available")
	}
}

func TestLeaseCPUStatPathRespectsTheMountedCgroupSubtree(t *testing.T) {
	for _, tc := range []struct{ member, mount, want string }{
		{"0::/a/b\n", "1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw", "/sys/fs/cgroup/a/b/cpu.stat"},
		{"0::/a/b\n", "1 0 0:1 /a/b /sys/fs/cgroup rw - cgroup2 cgroup rw", "/sys/fs/cgroup/cpu.stat"},
		{"2:cpu,cpuacct:/a/b\n", "1 0 0:1 /a /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct", "/sys/fs/cgroup/cpu/b/cpu.stat"},
		{"0::/a/b\n", "1 0 0:1 /other /sys/fs/cgroup rw - cgroup2 cgroup rw", ""},
		{"0::/../../other\n", "1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw", ""},
	} {
		if got := leaseCPUStatPath(tc.member, tc.mount); got != tc.want {
			t.Errorf("got %q want %q", got, tc.want)
		}
	}
}

func TestLeaseRecoveryDoesNotMistakeAnOldResponseForRecovery(t *testing.T) {
	t0 := time.Unix(1000, 0)
	r := leaseRecoveryTracker{}
	r.update(t0, 0, 0, t0, 15, "READY")
	r.disruption = t0.Add(2 * time.Second)
	if r.update(t0.Add(3*time.Second), 1, 1, t0.Add(time.Second), 15, "READY") {
		t.Fatal("response before the failure was treated as recovery")
	}
	if !r.update(t0.Add(4*time.Second), 1, 1, t0.Add(4*time.Second), 15, "READY") {
		t.Fatal("response after the failure was lost")
	}
}

func TestLeaseFlightHistoryHasAByteBudget(t *testing.T) {
	r := &leaseFlightRecorder{}
	for i := 0; i < 8; i++ {
		r.record(leaseFlightSample{At: time.Unix(int64(i), 0), Resources: map[string]string{"cgroup_cpu_source": strings.Repeat("x", 2000)}})
	}
	var samples []leaseFlightSample
	data := r.history()
	if len(data) > 6000 || json.Unmarshal([]byte(data), &samples) != nil || len(samples) == 0 || len(samples) >= 8 || !samples[len(samples)-1].At.Equal(time.Unix(7, 0)) {
		t.Fatalf("history did not retain the newest complete samples within its budget: %d bytes", len(data))
	}
}

func TestLeaseRecoveryRequiresAPositiveLeaseTTL(t *testing.T) {
	t0 := time.Unix(1000, 0)
	r := leaseRecoveryTracker{}
	r.update(t0, 0, 0, t0, 15, "READY")
	r.update(t0.Add(time.Second), 1, 0, t0, 15, "CONNECTING")
	if r.update(t0.Add(2*time.Second), 1, 0, t0.Add(2*time.Second), 0, "READY") {
		t.Fatal("expired lease reported as recovered")
	}
}

func TestHealthyLeaseFlightRecorderDoesNotLog(t *testing.T) {
	output := &countingLogWriter{}
	original := log.Writer()
	log.SetOutput(output)
	defer log.SetOutput(original)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	d := &leaseTransportDiagnostics{}
	d.HandleRPC(ctx, &stats.InPayload{Payload: &etcdpb.LeaseKeepAliveResponse{TTL: 15}})
	r := startLeaseFlightRecorderWithSampler(ctx, 42, d, func() string { return "READY" }, func() leaseFlightSample { return leaseFlightSample{At: time.Now(), State: "READY"} })
	time.Sleep(1100 * time.Millisecond)
	cancel()
	if r.history() == "[]" || output.writes.Load() != 0 {
		t.Fatalf("healthy sampler produced %d log writes", output.writes.Load())
	}
}

func (w *countingLogWriter) Write(p []byte) (int, error) {
	w.writes.Add(1)
	return len(p), nil
}

func TestLeaseDiagnosticsNeverLogFromCallbacksOrAckLoop(t *testing.T) {
	output := &countingLogWriter{}
	original := log.Writer()
	log.SetOutput(output)
	defer log.SetOutput(original)
	diagnostics := &leaseTransportDiagnostics{}
	for i := 0; i < 1000; i++ {
		diagnostics.HandleRPC(context.Background(), &stats.End{
			Error: status.Error(codes.Unavailable, "repeated connection failure"),
		})
		diagnostics.HandleConn(context.Background(), &stats.ConnEnd{})
	}
	responses := make(chan *clientv3.LeaseKeepAliveResponse, 1000)
	for i := 0; i < cap(responses); i++ {
		responses <- &clientv3.LeaseKeepAliveResponse{ID: 42, TTL: 5}
	}
	close(responses)
	_, detail := observeKeepAlive(context.Background(), context.Background(), 42, responses,
		func() string { return "READY" })
	if output.writes.Load() != 0 {
		t.Fatalf("diagnostics performed %d synchronous writes during a retry/ACK storm", output.writes.Load())
	}
	if !strings.Contains(detail, "ack_count=1000") {
		t.Fatalf("ACK summary was lost: %s", detail)
	}
	if !strings.Contains(diagnostics.summary(), "rpc_failures=1000") {
		t.Fatalf("transport failure summary was lost: %s", diagnostics.summary())
	}
}

func TestLeaseCallbacksDoNotWaitForDiagnosticSnapshot(t *testing.T) {
	diagnostics := &leaseTransportDiagnostics{}
	diagnostics.stateMu.Lock()
	defer diagnostics.stateMu.Unlock()
	done := make(chan struct{})
	go func() {
		ctx := diagnostics.TagConn(context.Background(), &stats.ConnTagInfo{LocalAddr: &net.TCPAddr{IP: net.ParseIP("127.0.0.1"), Port: 10000}, RemoteAddr: &net.TCPAddr{IP: net.ParseIP("127.0.0.1"), Port: 2379}})
		diagnostics.HandleRPC(context.Background(), &stats.End{Error: status.Error(codes.Unavailable, "failure")})
		diagnostics.HandleConn(ctx, &stats.ConnEnd{})
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("transport callback blocked on diagnostic state")
	}
	if !strings.Contains(diagnostics.summary(), "transport_detail_busy=1") {
		t.Fatal("summary waited for a busy diagnostic lock")
	}
	if a := diagnostics.addresses.Load(); a == nil || a.remote != "127.0.0.1:2379" {
		t.Fatal("diagnostic contention lost the only connection address")
	}
}

func TestObserveKeepAliveReportsLastAcknowledgementOnClose(t *testing.T) {
	responses := make(chan *clientv3.LeaseKeepAliveResponse, 2)
	responses <- &clientv3.LeaseKeepAliveResponse{ID: 42, TTL: 5}
	responses <- &clientv3.LeaseKeepAliveResponse{ID: 42, TTL: 15}
	close(responses)
	code, detail := observeKeepAlive(context.Background(), context.Background(),
		42, responses, func() string { return "TRANSIENT_FAILURE" })
	if code != -1 {
		t.Fatalf("closed channel code = %d, want -1", code)
	}
	for _, field := range []string{"keep alive channel closed", "ack_count=2",
		"last_ttl_sec=15", "transport_state=TRANSIENT_FAILURE", "context_error=<nil>"} {
		if !strings.Contains(detail, field) {
			t.Errorf("diagnostic %q does not contain %q", detail, field)
		}
	}
}

func TestObserveKeepAlivePreservesCancellationResult(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	code, detail := observeKeepAlive(ctx, context.Background(), 42,
		make(chan *clientv3.LeaseKeepAliveResponse), func() string { return "READY" })
	if code != -2 || !strings.Contains(detail, "context_error=context canceled") {
		t.Fatalf("cancellation result = (%d, %q), want -2 and cancelled context", code, detail)
	}
}

func TestObserveKeepAliveReportsMissingFirstAcknowledgement(t *testing.T) {
	responses := make(chan *clientv3.LeaseKeepAliveResponse)
	close(responses)
	code, detail := observeKeepAlive(context.Background(), context.Background(),
		42, responses, func() string { return "CONNECTING" })
	if code != -1 || !strings.Contains(detail, "ack_count=0") ||
		!strings.Contains(detail, "last_ack_age_ms=-1") {
		t.Fatalf("no-ACK result = (%d, %q), want explicit missing ACK", code, detail)
	}
}

type brokenLeaseStream struct {
	etcdpb.UnimplementedLeaseServer
	acknowledged atomic.Bool
}

func (s *brokenLeaseStream) LeaseKeepAlive(stream etcdpb.Lease_LeaseKeepAliveServer) error {
	request, err := stream.Recv()
	if err != nil {
		return err
	}
	if !s.acknowledged.Swap(true) {
		if err := stream.Send(&etcdpb.LeaseKeepAliveResponse{ID: request.ID, TTL: 1}); err != nil {
			return err
		}
	}
	return status.Error(codes.Unavailable, "injected lease stream failure")
}

// A real gRPC stream error must survive the etcd client's retry loop in the
// diagnostics, even though the public KeepAlive API only returns a closed
// channel after its ACK deadline. No external etcd process is needed.
func TestLeaseTransportDiagnosticsReportsStreamFailure(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := grpc.NewServer()
	etcdpb.RegisterLeaseServer(server, &brokenLeaseStream{})
	go server.Serve(listener)
	defer server.Stop()
	diagnostics := &leaseTransportDiagnostics{}
	cli, err := clientv3.New(clientv3.Config{
		Endpoints: []string{listener.Addr().String()}, DialTimeout: time.Second,
		DialOptions: []grpc.DialOption{grpc.WithStatsHandler(diagnostics)},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer cli.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	responses, err := cli.KeepAlive(ctx, 42)
	if err != nil {
		t.Fatal(err)
	}
	recorder := startLeaseFlightRecorder(ctx, 42, diagnostics, func() string { return cli.ActiveConnection().GetState().String() })
	code, detail := observeKeepAlive(ctx, cli.Ctx(), 42, responses, func() string {
		return cli.ActiveConnection().GetState().String() + " " + diagnostics.summary() + " flight_history=" + recorder.history()
	})
	if code != -1 || !strings.Contains(detail, "ack_count=1") {
		t.Fatalf("stream result = (%d, %q), want closed stream after one ACK", code, detail)
	}
	for _, field := range []string{"last_rpc_code=Unavailable", "injected lease stream failure", "conn_begins=1", "remote=127.0.0.1:", "flight_history=[{", "requests_sent=", "responses_received=1"} {
		if !strings.Contains(detail, field) {
			t.Errorf("missing transport diagnostic %q in %s", field, detail)
		}
	}
}

type recoveringLeaseStream struct {
	etcdpb.UnimplementedLeaseServer
	calls atomic.Uint64
}

func (s *recoveringLeaseStream) LeaseKeepAlive(stream etcdpb.Lease_LeaseKeepAliveServer) error {
	first := s.calls.Add(1) == 1
	for {
		req, err := stream.Recv()
		if err != nil {
			return err
		}
		if err = stream.Send(&etcdpb.LeaseKeepAliveResponse{ID: req.ID, TTL: 15}); err != nil {
			return err
		}
		if first {
			return status.Error(codes.Unavailable, "brief injected outage")
		}
	}
}

func TestLeaseFlightRecorderReportsRecoveryBeforeTheTTLExpires(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := grpc.NewServer()
	etcdpb.RegisterLeaseServer(server, &recoveringLeaseStream{})
	go server.Serve(listener)
	defer server.Stop()
	d := &leaseTransportDiagnostics{}
	cli, err := clientv3.New(clientv3.Config{Endpoints: []string{listener.Addr().String()}, DialTimeout: time.Second, DialOptions: []grpc.DialOption{grpc.WithStatsHandler(d)}})
	if err != nil {
		t.Fatal(err)
	}
	defer cli.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	responses, err := cli.KeepAlive(ctx, 42)
	if err != nil {
		t.Fatal(err)
	}
	r := startLeaseFlightRecorder(ctx, 42, d, func() string { return cli.ActiveConnection().GetState().String() })
	var acks int
	tick := time.NewTicker(10 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case response, ok := <-responses:
			if !ok || response == nil {
				t.Fatal("lease expired during a short recoverable outage")
			}
			acks++
		case <-tick.C:
			if r.recoveries.Load() > 0 {
				if acks < 2 || d.rpcFailures.Load() == 0 {
					t.Fatal("recovery was not backed by a failure and new ACK")
				}
				return
			}
		case <-ctx.Done():
			t.Fatal("successful reconnection was not reported")
		}
	}
}

func TestNewMaintenanceSessionCancelsStartup(t *testing.T) {
	original := startMaintenanceSession
	defer func() { startMaintenanceSession = original }()

	started := make(chan struct{})
	startMaintenanceSession = func(ctx context.Context, _ *clientv3.Client,
		_ int) (*concurrency.Session, error) {
		close(started)
		<-ctx.Done()
		return nil, ctx.Err()
	}

	result := make(chan error, 1)
	go func() {
		_, err := newMaintenanceSession(nil, 30, 10*time.Millisecond)
		result <- err
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("maintenance session startup was not attempted")
	}
	select {
	case err := <-result:
		if err == nil {
			t.Fatal("expected startup cancellation error")
		}
	case <-time.After(time.Second):
		t.Fatal("maintenance session startup did not cancel")
	}
}

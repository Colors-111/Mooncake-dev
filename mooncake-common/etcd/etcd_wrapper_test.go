package main

import (
	"context"
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
		diagnostics.HandleRPC(context.Background(), &stats.End{Error: status.Error(codes.Unavailable, "failure")})
		diagnostics.HandleConn(context.Background(), &stats.ConnEnd{})
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
	code, detail := observeKeepAlive(ctx, cli.Ctx(), 42, responses, func() string {
		return cli.ActiveConnection().GetState().String() + " " + diagnostics.summary()
	})
	if code != -1 || !strings.Contains(detail, "ack_count=1") {
		t.Fatalf("stream result = (%d, %q), want closed stream after one ACK", code, detail)
	}
	for _, field := range []string{"last_rpc_code=Unavailable", "injected lease stream failure", "conn_begins=1", "remote=127.0.0.1:"} {
		if !strings.Contains(detail, field) {
			t.Errorf("missing transport diagnostic %q in %s", field, detail)
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

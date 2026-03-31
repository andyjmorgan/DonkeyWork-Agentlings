package loop

import (
	"context"
	"strings"
	"testing"

	"github.com/andyjmorgan/agentlings-go/internal/config"
	"github.com/andyjmorgan/agentlings-go/internal/llm"
	"github.com/andyjmorgan/agentlings-go/internal/store"
	"github.com/andyjmorgan/agentlings-go/internal/tools"
)

func newTestLoop(t *testing.T, toolNames []string) *MessageLoop {
	t.Helper()
	dataDir := t.TempDir()
	cfg := &config.Config{
		Definition: config.AgentDefinition{
			Name:        "test-agent",
			Description: "test",
		},
	}
	s := store.NewJournalStore(dataDir)
	mock := llm.NewMockLLMClient(toolNames)
	reg := tools.NewToolRegistry()
	if len(toolNames) > 0 {
		reg.RegisterTools(toolNames)
	}
	return New(cfg, s, mock, reg)
}

func TestNewContext(t *testing.T) {
	ml := newTestLoop(t, nil)
	result, err := ml.ProcessMessage(context.Background(), "hello", "", "a2a")
	if err != nil {
		t.Fatalf("ProcessMessage: %v", err)
	}
	if result.ContextID == "" {
		t.Error("expected non-empty context ID")
	}
	if len(result.Content) == 0 {
		t.Error("expected non-empty content")
	}
	text, _ := result.Content[0]["text"].(string)
	if !strings.Contains(text, "hello") {
		t.Errorf("expected echo, got %q", text)
	}
}

func TestContinuation(t *testing.T) {
	ml := newTestLoop(t, nil)

	r1, err := ml.ProcessMessage(context.Background(), "first", "", "a2a")
	if err != nil {
		t.Fatalf("first: %v", err)
	}

	r2, err := ml.ProcessMessage(context.Background(), "second", r1.ContextID, "a2a")
	if err != nil {
		t.Fatalf("second: %v", err)
	}

	if r2.ContextID != r1.ContextID {
		t.Errorf("context IDs differ: %q vs %q", r1.ContextID, r2.ContextID)
	}
}

func TestExternalContextID(t *testing.T) {
	ml := newTestLoop(t, nil)
	result, err := ml.ProcessMessage(context.Background(), "hello", "external-id", "mcp")
	if err != nil {
		t.Fatalf("ProcessMessage: %v", err)
	}
	if result.ContextID != "external-id" {
		t.Errorf("context ID = %q, want external-id", result.ContextID)
	}
}

func TestToolExecution(t *testing.T) {
	ml := newTestLoop(t, []string{"bash"})

	result, err := ml.ProcessMessage(context.Background(), "please run bash", "", "a2a")
	if err != nil {
		t.Fatalf("ProcessMessage: %v", err)
	}

	text, _ := result.Content[0]["text"].(string)
	if !strings.Contains(text, "Tool result received") {
		t.Errorf("expected tool result acknowledgment, got %q", text)
	}
}

func TestJournalRecording(t *testing.T) {
	dataDir := t.TempDir()
	cfg := &config.Config{
		Definition: config.AgentDefinition{Name: "test", Description: "test"},
	}
	s := store.NewJournalStore(dataDir)
	mock := llm.NewMockLLMClient(nil)
	reg := tools.NewToolRegistry()
	ml := New(cfg, s, mock, reg)

	result, err := ml.ProcessMessage(context.Background(), "hello", "", "a2a")
	if err != nil {
		t.Fatalf("ProcessMessage: %v", err)
	}

	messages, err := s.Replay(result.ContextID)
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}

	if len(messages) < 2 {
		t.Fatalf("expected at least 2 messages in journal, got %d", len(messages))
	}
	if messages[0]["role"] != "user" {
		t.Errorf("first message role = %v, want user", messages[0]["role"])
	}
	if messages[1]["role"] != "assistant" {
		t.Errorf("second message role = %v, want assistant", messages[1]["role"])
	}
}

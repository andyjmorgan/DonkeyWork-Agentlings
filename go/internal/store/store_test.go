package store

import (
	"testing"

	"github.com/andyjmorgan/agentlings-go/internal/models"
)

func TestCreateAndExists(t *testing.T) {
	s := NewJournalStore(t.TempDir())

	if s.Exists("ctx-1") {
		t.Error("context should not exist before creation")
	}

	if err := s.Create("ctx-1"); err != nil {
		t.Fatalf("Create: %v", err)
	}

	if !s.Exists("ctx-1") {
		t.Error("context should exist after creation")
	}
}

func TestAppendAndReplay(t *testing.T) {
	s := NewJournalStore(t.TempDir())
	s.Create("ctx-1")

	user := models.NewMessageEntry("ctx-1", "user", "a2a", []map[string]any{
		{"type": "text", "text": "hello"},
	})
	asst := models.NewMessageEntry("ctx-1", "assistant", "a2a", []map[string]any{
		{"type": "text", "text": "hi there"},
	})

	if err := s.Append("ctx-1", user); err != nil {
		t.Fatalf("Append user: %v", err)
	}
	if err := s.Append("ctx-1", asst); err != nil {
		t.Fatalf("Append assistant: %v", err)
	}

	msgs, err := s.Replay("ctx-1")
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	if len(msgs) != 2 {
		t.Fatalf("Replay returned %d messages, want 2", len(msgs))
	}
	if msgs[0]["role"] != "user" {
		t.Errorf("msgs[0].role = %v, want user", msgs[0]["role"])
	}
	if msgs[1]["role"] != "assistant" {
		t.Errorf("msgs[1].role = %v, want assistant", msgs[1]["role"])
	}
}

func TestReplayEmpty(t *testing.T) {
	s := NewJournalStore(t.TempDir())
	s.Create("ctx-1")

	msgs, err := s.Replay("ctx-1")
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	if msgs != nil {
		t.Errorf("expected nil for empty replay, got %v", msgs)
	}
}

func TestReplayWithCompaction(t *testing.T) {
	s := NewJournalStore(t.TempDir())
	s.Create("ctx-1")

	s.Append("ctx-1", models.NewMessageEntry("ctx-1", "user", "a2a", []map[string]any{
		{"type": "text", "text": "old message"},
	}))
	s.Append("ctx-1", models.NewMessageEntry("ctx-1", "assistant", "a2a", []map[string]any{
		{"type": "text", "text": "old reply"},
	}))

	s.Append("ctx-1", models.NewCompactionEntry("ctx-1", "summary of earlier conversation"))

	s.Append("ctx-1", models.NewMessageEntry("ctx-1", "user", "a2a", []map[string]any{
		{"type": "text", "text": "new message"},
	}))

	msgs, err := s.Replay("ctx-1")
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}

	if len(msgs) != 2 {
		t.Fatalf("Replay returned %d messages, want 2", len(msgs))
	}

	if msgs[0]["role"] != "assistant" {
		t.Errorf("msgs[0].role = %v, want assistant (compaction summary)", msgs[0]["role"])
	}
	if msgs[0]["content"] != "summary of earlier conversation" {
		t.Errorf("msgs[0].content = %v, want compaction summary", msgs[0]["content"])
	}
	if msgs[1]["role"] != "user" {
		t.Errorf("msgs[1].role = %v, want user", msgs[1]["role"])
	}
}

func TestAppendNonExistentContext(t *testing.T) {
	s := NewJournalStore(t.TempDir())
	err := s.Append("nonexistent", models.NewMessageEntry("nonexistent", "user", "a2a", nil))
	if err == nil {
		t.Error("expected error for nonexistent context")
	}
}

func TestReplayNonExistentContext(t *testing.T) {
	s := NewJournalStore(t.TempDir())
	_, err := s.Replay("nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent context")
	}
}

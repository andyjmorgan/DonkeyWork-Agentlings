package models

import (
	"encoding/json"
	"testing"
)

func TestNewMessageEntry(t *testing.T) {
	content := []map[string]any{{"type": "text", "text": "hello"}}
	e := NewMessageEntry("ctx-1", "user", "a2a", content)

	if e.T != "msg" {
		t.Errorf("T = %q, want %q", e.T, "msg")
	}
	if e.Ctx != "ctx-1" {
		t.Errorf("Ctx = %q, want %q", e.Ctx, "ctx-1")
	}
	if e.Role != "user" {
		t.Errorf("Role = %q, want %q", e.Role, "user")
	}
	if e.Via != "a2a" {
		t.Errorf("Via = %q, want %q", e.Via, "a2a")
	}
	if e.TS == "" {
		t.Error("TS should not be empty")
	}
}

func TestNewCompactionEntry(t *testing.T) {
	e := NewCompactionEntry("ctx-2", "summary text")

	if e.T != "compact" {
		t.Errorf("T = %q, want %q", e.T, "compact")
	}
	if e.Content != "summary text" {
		t.Errorf("Content = %q, want %q", e.Content, "summary text")
	}
}

func TestMessageEntryRoundTrip(t *testing.T) {
	original := NewMessageEntry("ctx-1", "assistant", "mcp", []map[string]any{{"type": "text", "text": "hi"}})

	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	parsed, err := ParseJournalEntry(data)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	msg, ok := parsed.(MessageEntry)
	if !ok {
		t.Fatalf("expected MessageEntry, got %T", parsed)
	}
	if msg.Role != "assistant" {
		t.Errorf("Role = %q, want %q", msg.Role, "assistant")
	}
}

func TestCompactionEntryRoundTrip(t *testing.T) {
	original := NewCompactionEntry("ctx-2", "compacted")

	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	parsed, err := ParseJournalEntry(data)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	compact, ok := parsed.(CompactionEntry)
	if !ok {
		t.Fatalf("expected CompactionEntry, got %T", parsed)
	}
	if compact.Content != "compacted" {
		t.Errorf("Content = %q, want %q", compact.Content, "compacted")
	}
}

func TestParseJournalEntryUnknownType(t *testing.T) {
	data := []byte(`{"t":"unknown"}`)
	_, err := ParseJournalEntry(data)
	if err == nil {
		t.Error("expected error for unknown type")
	}
}

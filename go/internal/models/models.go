package models

import (
	"encoding/json"
	"fmt"
	"time"
)

// MessageEntry represents a conversational message stored in the JSONL journal.
// T is the type discriminator, always "msg" for message entries. TS holds a
// UTC timestamp in RFC 3339 nanosecond format recording when the entry was
// created. Ctx is the conversation context ID that groups related journal
// entries together. Role identifies the message author, typically "user" or
// "assistant". Content carries the message body as Anthropic-style content
// blocks (each map contains at least a "type" key, e.g. "text" or "tool_use").
// Via records the protocol that originated the message, such as "a2a" or "mcp".
type MessageEntry struct {
	T       string           `json:"t"`
	TS      string           `json:"ts"`
	Ctx     string           `json:"ctx"`
	Role    string           `json:"role"`
	Content []map[string]any `json:"content"`
	Via     string           `json:"via"`
}

// NewMessageEntry creates a MessageEntry with the current UTC timestamp and
// the "msg" type tag. The ctx argument is the conversation context ID, role is
// the message author ("user" or "assistant"), via is the originating protocol
// ("a2a" or "mcp"), and content is a slice of Anthropic-style content blocks.
// Content may be nil when a message carries no body, such as a placeholder entry.
func NewMessageEntry(ctx, role, via string, content []map[string]any) MessageEntry {
	return MessageEntry{
		T:       "msg",
		TS:      time.Now().UTC().Format(time.RFC3339Nano),
		Ctx:     ctx,
		Role:    role,
		Content: content,
		Via:     via,
	}
}

// CompactionEntry represents a journal compaction marker that serves as a replay
// cursor. When the journal is replayed, entries before the last compaction
// marker are skipped, and the compaction's Content is used as a summary of the
// earlier conversation. T is the type discriminator, always "compact". TS holds
// a UTC timestamp in RFC 3339 nanosecond format. Ctx is the conversation
// context ID. Content is the LLM-generated summary text that replaces all
// preceding messages during replay.
type CompactionEntry struct {
	T       string `json:"t"`
	TS      string `json:"ts"`
	Ctx     string `json:"ctx"`
	Content string `json:"content"`
}

// NewCompactionEntry creates a CompactionEntry with the current UTC timestamp
// and the "compact" type tag. The ctx argument is the conversation context ID,
// and content is the summary text that will replace all earlier journal entries
// for that context during replay.
func NewCompactionEntry(ctx, content string) CompactionEntry {
	return CompactionEntry{
		T:       "compact",
		TS:      time.Now().UTC().Format(time.RFC3339Nano),
		Ctx:     ctx,
		Content: content,
	}
}

// JournalEntry holds the type discriminator and raw JSON for a journal line
// before full parsing. T contains the discriminator value (e.g. "msg" or
// "compact") extracted during an initial probe unmarshal, and raw retains the
// original JSON bytes for subsequent decoding into the concrete entry type.
type JournalEntry struct {
	T string `json:"t"`
	raw json.RawMessage
}

// ParseJournalEntry unmarshals a raw JSON line from the JSONL journal into the
// appropriate concrete type. It probes the "t" field in data to determine the
// entry kind: "msg" yields a MessageEntry, "compact" yields a CompactionEntry.
// The returned value is one of those two types; callers should type-switch on
// the result. An error is returned if data is not valid JSON, or if the type
// discriminator is missing or unrecognized.
func ParseJournalEntry(data []byte) (any, error) {
	var probe struct {
		T string `json:"t"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, fmt.Errorf("parse journal entry: %w", err)
	}
	switch probe.T {
	case "msg":
		var e MessageEntry
		if err := json.Unmarshal(data, &e); err != nil {
			return nil, fmt.Errorf("parse message entry: %w", err)
		}
		return e, nil
	case "compact":
		var e CompactionEntry
		if err := json.Unmarshal(data, &e); err != nil {
			return nil, fmt.Errorf("parse compaction entry: %w", err)
		}
		return e, nil
	default:
		return nil, fmt.Errorf("unknown journal entry type: %q", probe.T)
	}
}

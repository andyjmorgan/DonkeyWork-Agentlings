package models

import (
	"encoding/json"
	"fmt"
	"time"
)

type MessageEntry struct {
	T       string           `json:"t"`
	TS      string           `json:"ts"`
	Ctx     string           `json:"ctx"`
	Role    string           `json:"role"`
	Content []map[string]any `json:"content"`
	Via     string           `json:"via"`
}

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

type CompactionEntry struct {
	T       string `json:"t"`
	TS      string `json:"ts"`
	Ctx     string `json:"ctx"`
	Content string `json:"content"`
}

func NewCompactionEntry(ctx, content string) CompactionEntry {
	return CompactionEntry{
		T:       "compact",
		TS:      time.Now().UTC().Format(time.RFC3339Nano),
		Ctx:     ctx,
		Content: content,
	}
}

type JournalEntry struct {
	T string `json:"t"`
	raw json.RawMessage
}

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

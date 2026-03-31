package store

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"syscall"

	"github.com/andyjmorgan/agentlings-go/internal/models"
)

var ErrContextNotFound = errors.New("context not found")

type JournalStore struct {
	dataDir string
}

func NewJournalStore(dataDir string) *JournalStore {
	os.MkdirAll(dataDir, 0o755)
	return &JournalStore{dataDir: dataDir}
}

func (s *JournalStore) path(ctxID string) string {
	return filepath.Join(s.dataDir, ctxID+".jsonl")
}

func (s *JournalStore) Create(ctxID string) error {
	p := s.path(ctxID)
	f, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("create context %s: %w", ctxID, err)
	}
	f.Close()
	slog.Info("created context", "ctx", ctxID)
	return nil
}

func (s *JournalStore) Exists(ctxID string) bool {
	_, err := os.Stat(s.path(ctxID))
	return err == nil
}

func (s *JournalStore) Append(ctxID string, entry any) error {
	p := s.path(ctxID)
	if _, err := os.Stat(p); err != nil {
		return fmt.Errorf("%w: %s", ErrContextNotFound, ctxID)
	}

	data, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("marshal entry: %w", err)
	}
	data = append(data, '\n')

	f, err := os.OpenFile(p, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("open %s: %w", p, err)
	}
	defer f.Close()

	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX); err != nil {
		return fmt.Errorf("lock %s: %w", p, err)
	}
	defer syscall.Flock(int(f.Fd()), syscall.LOCK_UN)

	if _, err := f.Write(data); err != nil {
		return fmt.Errorf("write %s: %w", p, err)
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("sync %s: %w", p, err)
	}

	t := "unknown"
	switch entry.(type) {
	case models.MessageEntry:
		t = "msg"
	case models.CompactionEntry:
		t = "compact"
	}
	slog.Debug("appended entry", "type", t, "ctx", ctxID)
	return nil
}

func (s *JournalStore) Replay(ctxID string) ([]map[string]any, error) {
	p := s.path(ctxID)
	if _, err := os.Stat(p); err != nil {
		return nil, fmt.Errorf("%w: %s", ErrContextNotFound, ctxID)
	}

	f, err := os.Open(p)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", p, err)
	}
	defer f.Close()

	var parsed []map[string]any
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		var entry map[string]any
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			return nil, fmt.Errorf("parse journal line: %w", err)
		}
		parsed = append(parsed, entry)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan %s: %w", p, err)
	}

	if len(parsed) == 0 {
		return nil, nil
	}

	cursor := 0
	for i := len(parsed) - 1; i >= 0; i-- {
		if t, _ := parsed[i]["t"].(string); t == "compact" {
			cursor = i
			break
		}
	}

	var messages []map[string]any
	for _, entry := range parsed[cursor:] {
		t, _ := entry["t"].(string)
		switch t {
		case "compact":
			messages = append(messages, map[string]any{
				"role":    "assistant",
				"content": entry["content"],
			})
		case "msg":
			messages = append(messages, map[string]any{
				"role":    entry["role"],
				"content": entry["content"],
			})
		}
	}

	return messages, nil
}

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

// ErrContextNotFound is returned when a journal file does not exist for the given context ID.
var ErrContextNotFound = errors.New("context not found")

// JournalStore manages append-only JSONL journals keyed by context ID.
// Each context ID maps to a single .jsonl file under the configured data
// directory. Writes are serialised with flock to allow concurrent appenders.
type JournalStore struct {
	dataDir string
}

// NewJournalStore returns a JournalStore that persists journals under dataDir,
// creating the directory (and any missing parents) if it does not already exist.
// The returned store is ready for use immediately.
func NewJournalStore(dataDir string) *JournalStore {
	os.MkdirAll(dataDir, 0o755)
	return &JournalStore{dataDir: dataDir}
}

func (s *JournalStore) path(ctxID string) string {
	return filepath.Join(s.dataDir, ctxID+".jsonl")
}

// Create initialises an empty journal file for ctxID. The context ID is used
// verbatim as the filename stem (with a .jsonl extension), so callers should
// ensure it is filesystem-safe. Returns a non-nil error if the file cannot be
// created.
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

// Exists reports whether a journal file already exists on disk for ctxID.
// It returns false for any stat error, including permission failures.
func (s *JournalStore) Exists(ctxID string) bool {
	_, err := os.Stat(s.path(ctxID))
	return err == nil
}

// Append serialises entry as a single JSON line and appends it to the journal
// for ctxID. The entry should be a MessageEntry or CompactionEntry, but any
// JSON-marshalable value is accepted. The write is protected by an exclusive
// flock and followed by an fsync, so concurrent appenders are safe and data is
// durable on return. If the journal file for ctxID does not exist, Append
// returns an error wrapping ErrContextNotFound.
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

// Replay reads the full journal for ctxID and returns the reconstructed message
// history starting from the most recent compaction marker. If no compaction
// marker exists, the entire journal is replayed from the beginning. Each
// returned map contains "role" and "content" keys suitable for passing to the
// LLM messages API. An empty journal yields a nil slice with no error. If the
// journal file for ctxID does not exist, Replay returns an error wrapping
// ErrContextNotFound. Individual lines up to 10 MB are supported.
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

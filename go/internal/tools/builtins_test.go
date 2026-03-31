package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBashTool(t *testing.T) {
	result := bashTool(context.Background(), map[string]any{"command": "echo hello"})
	if result.Output != "hello" {
		t.Errorf("Output = %q, want %q", result.Output, "hello")
	}
	if result.IsError {
		t.Error("should not be error")
	}
}

func TestBashToolFailure(t *testing.T) {
	result := bashTool(context.Background(), map[string]any{"command": "exit 1"})
	if !result.IsError {
		t.Error("should be error")
	}
}

func TestBashToolTimeout(t *testing.T) {
	result := bashTool(context.Background(), map[string]any{
		"command": "sleep 10",
		"timeout": float64(1),
	})
	if !result.IsError {
		t.Error("should be error")
	}
	if !strings.Contains(result.Output, "timed out") {
		t.Errorf("expected timeout message, got %q", result.Output)
	}
}

func TestReadFileTool(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "test.txt")
	os.WriteFile(path, []byte("line1\nline2\nline3\n"), 0o644)

	result := readFileTool(context.Background(), map[string]any{"path": path})
	if result.IsError {
		t.Fatalf("error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "1\tline1") {
		t.Errorf("expected numbered lines, got %q", result.Output)
	}
}

func TestReadFileToolOffset(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "test.txt")
	os.WriteFile(path, []byte("a\nb\nc\nd\n"), 0o644)

	result := readFileTool(context.Background(), map[string]any{
		"path":   path,
		"offset": float64(2),
		"limit":  float64(1),
	})
	if result.IsError {
		t.Fatalf("error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "3\tc") {
		t.Errorf("expected line 3, got %q", result.Output)
	}
	if !strings.Contains(result.Output, "1 more lines") {
		t.Errorf("expected truncation notice, got %q", result.Output)
	}
}

func TestReadFileToolNotFound(t *testing.T) {
	result := readFileTool(context.Background(), map[string]any{"path": "/nonexistent/file.txt"})
	if !result.IsError {
		t.Error("expected error for missing file")
	}
}

func TestWriteFileTool(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "out.txt")

	result := writeFileTool(context.Background(), map[string]any{"path": path, "content": "hello"})
	if result.IsError {
		t.Fatalf("error: %s", result.Output)
	}

	data, _ := os.ReadFile(path)
	if string(data) != "hello" {
		t.Errorf("file content = %q, want %q", string(data), "hello")
	}
}

func TestEditFileTool(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "edit.txt")
	os.WriteFile(path, []byte("hello world"), 0o644)

	result := editFileTool(context.Background(), map[string]any{
		"path":     path,
		"old_text": "world",
		"new_text": "go",
	})
	if result.IsError {
		t.Fatalf("error: %s", result.Output)
	}

	data, _ := os.ReadFile(path)
	if string(data) != "hello go" {
		t.Errorf("file content = %q, want %q", string(data), "hello go")
	}
}

func TestEditFileToolNotFound(t *testing.T) {
	result := editFileTool(context.Background(), map[string]any{
		"path":     "/nonexistent",
		"old_text": "a",
		"new_text": "b",
	})
	if !result.IsError {
		t.Error("expected error")
	}
}

func TestEditFileToolMultipleMatches(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "dup.txt")
	os.WriteFile(path, []byte("aaa"), 0o644)

	result := editFileTool(context.Background(), map[string]any{
		"path":     path,
		"old_text": "a",
		"new_text": "b",
	})
	if !result.IsError {
		t.Error("expected error for multiple matches without replace_all")
	}
}

func TestEditFileToolReplaceAll(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "all.txt")
	os.WriteFile(path, []byte("aaa"), 0o644)

	result := editFileTool(context.Background(), map[string]any{
		"path":        path,
		"old_text":    "a",
		"new_text":    "b",
		"replace_all": true,
	})
	if result.IsError {
		t.Fatalf("error: %s", result.Output)
	}

	data, _ := os.ReadFile(path)
	if string(data) != "bbb" {
		t.Errorf("file content = %q, want %q", string(data), "bbb")
	}
}

func TestListDirectoryTool(t *testing.T) {
	tmp := t.TempDir()
	os.Mkdir(filepath.Join(tmp, "subdir"), 0o755)
	os.WriteFile(filepath.Join(tmp, "file.txt"), []byte(""), 0o644)

	result := listDirectoryTool(context.Background(), map[string]any{"path": tmp})
	if result.IsError {
		t.Fatalf("error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "[DIR] subdir") {
		t.Errorf("expected [DIR] subdir in output: %q", result.Output)
	}
	if !strings.Contains(result.Output, "[FILE] file.txt") {
		t.Errorf("expected [FILE] file.txt in output: %q", result.Output)
	}
}

func TestSearchFilesTool(t *testing.T) {
	tmp := t.TempDir()
	os.WriteFile(filepath.Join(tmp, "a.go"), []byte(""), 0o644)
	os.WriteFile(filepath.Join(tmp, "b.txt"), []byte(""), 0o644)

	result := searchFilesTool(context.Background(), map[string]any{
		"path":    tmp,
		"pattern": "*.go",
	})
	if result.IsError {
		t.Fatalf("error: %s", result.Output)
	}
	if !strings.Contains(result.Output, "a.go") {
		t.Errorf("expected a.go in output: %q", result.Output)
	}
	if strings.Contains(result.Output, "b.txt") {
		t.Errorf("should not contain b.txt: %q", result.Output)
	}
}

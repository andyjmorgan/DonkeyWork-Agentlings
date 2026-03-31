package tools

import (
	"context"
	"testing"
)

func TestRegisterAndExecute(t *testing.T) {
	r := NewToolRegistry()
	r.Register("echo", "echo input", map[string]any{"type": "object"}, func(_ context.Context, input map[string]any) ToolResult {
		msg, _ := input["message"].(string)
		return ToolResult{Output: msg}
	})

	result := r.Execute(context.Background(), "echo", map[string]any{"message": "hello"})
	if result.Output != "hello" {
		t.Errorf("Output = %q, want %q", result.Output, "hello")
	}
	if result.IsError {
		t.Error("should not be error")
	}
}

func TestExecuteUnknownTool(t *testing.T) {
	r := NewToolRegistry()
	result := r.Execute(context.Background(), "nonexistent", nil)
	if !result.IsError {
		t.Error("expected error for unknown tool")
	}
}

func TestListSchemas(t *testing.T) {
	r := NewToolRegistry()
	r.Register("tool1", "desc1", map[string]any{"type": "object"}, nil)
	r.Register("tool2", "desc2", map[string]any{"type": "object"}, nil)

	schemas := r.ListSchemas()
	if len(schemas) != 2 {
		t.Errorf("expected 2 schemas, got %d", len(schemas))
	}
}

func TestToolNames(t *testing.T) {
	r := NewToolRegistry()
	r.Register("a", "a", nil, nil)
	r.Register("b", "b", nil, nil)

	names := r.ToolNames()
	if len(names) != 2 {
		t.Errorf("expected 2 names, got %d", len(names))
	}
}

func TestRegisterToolsByGroup(t *testing.T) {
	r := NewToolRegistry()
	r.RegisterTools([]string{"filesystem"})

	expected := []string{"read_file", "write_file", "edit_file", "list_directory", "search_files"}
	names := r.ToolNames()
	if len(names) != len(expected) {
		t.Errorf("expected %d tools, got %d", len(expected), len(names))
	}
}

func TestRegisterToolsUnknown(t *testing.T) {
	r := NewToolRegistry()
	r.RegisterTools([]string{"nonexistent_tool"})
	if len(r.ToolNames()) != 0 {
		t.Error("expected 0 tools for unknown name")
	}
}

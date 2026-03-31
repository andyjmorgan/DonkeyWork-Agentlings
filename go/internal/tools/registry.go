package tools

import (
	"context"
	"log/slog"
)

var ToolGroups = map[string][]string{
	"bash":       {"bash"},
	"filesystem": {"read_file", "write_file", "edit_file", "list_directory", "search_files"},
}

type ToolResult struct {
	Output  string
	IsError bool
}

type ToolFunc func(ctx context.Context, input map[string]any) ToolResult

type toolEntry struct {
	Name        string
	Description string
	InputSchema map[string]any
	ExecuteFn   ToolFunc
}

type ToolRegistry struct {
	tools map[string]toolEntry
}

func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{tools: make(map[string]toolEntry)}
}

func (r *ToolRegistry) Register(name, description string, inputSchema map[string]any, fn ToolFunc) {
	r.tools[name] = toolEntry{
		Name:        name,
		Description: description,
		InputSchema: inputSchema,
		ExecuteFn:   fn,
	}
	slog.Info("registered tool", "name", name)
}

func (r *ToolRegistry) ListSchemas() []map[string]any {
	schemas := make([]map[string]any, 0, len(r.tools))
	for _, t := range r.tools {
		schemas = append(schemas, map[string]any{
			"name":         t.Name,
			"description":  t.Description,
			"input_schema": t.InputSchema,
		})
	}
	return schemas
}

func (r *ToolRegistry) ToolNames() []string {
	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	return names
}

func (r *ToolRegistry) Execute(ctx context.Context, name string, input map[string]any) ToolResult {
	entry, ok := r.tools[name]
	if !ok {
		return ToolResult{Output: "Unknown tool: " + name, IsError: true}
	}
	slog.Debug("executing tool", "name", name)
	return entry.ExecuteFn(ctx, input)
}

func (r *ToolRegistry) RegisterTools(enabled []string) {
	resolved := make(map[string]struct{})
	for _, name := range enabled {
		if group, ok := ToolGroups[name]; ok {
			for _, t := range group {
				resolved[t] = struct{}{}
			}
		} else {
			resolved[name] = struct{}{}
		}
	}

	for toolName := range resolved {
		defn, ok := BuiltinRegistry[toolName]
		if !ok {
			slog.Warn("unknown tool", "name", toolName)
			continue
		}
		r.Register(defn.Name, defn.Description, defn.InputSchema, defn.ExecuteFn)
	}
}

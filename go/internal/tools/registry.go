package tools

import (
	"context"
	"log/slog"
)

// ToolGroups maps group names to the individual tool names they contain.
var ToolGroups = map[string][]string{
	"bash":       {"bash"},
	"filesystem": {"read_file", "write_file", "edit_file", "list_directory", "search_files"},
}

// ToolResult holds the output of a tool execution. Output contains the
// human-readable text produced by the tool, and IsError indicates whether the
// execution failed. When IsError is true, Output carries a diagnostic message
// suitable for returning to the LLM as a tool-result error.
type ToolResult struct {
	Output  string
	IsError bool
}

// ToolFunc is the signature for a callable tool. It receives a context for
// cancellation and a map of input parameters whose keys and value types are
// defined by the tool's JSON Schema. It returns a ToolResult describing the
// outcome of the execution.
type ToolFunc func(ctx context.Context, input map[string]any) ToolResult

type toolEntry struct {
	Name        string
	Description string
	InputSchema map[string]any
	ExecuteFn   ToolFunc
}

// ToolRegistry stores registered tools and dispatches execution by name. It is
// not safe for concurrent use; all registration should happen during startup
// before any calls to Execute.
type ToolRegistry struct {
	tools map[string]toolEntry
}

// NewToolRegistry returns an initialised, empty ToolRegistry ready for tool
// registration via Register or RegisterTools.
func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{tools: make(map[string]toolEntry)}
}

// Register adds a tool to the registry under the given name with the provided
// description and JSON Schema inputSchema. The fn callback is invoked when the
// tool is executed. Registering a name that already exists overwrites the
// previous entry.
func (r *ToolRegistry) Register(name, description string, inputSchema map[string]any, fn ToolFunc) {
	r.tools[name] = toolEntry{
		Name:        name,
		Description: description,
		InputSchema: inputSchema,
		ExecuteFn:   fn,
	}
	slog.Info("registered tool", "name", name)
}

// ListSchemas returns a slice of maps, one per registered tool, each containing
// "name", "description", and "input_schema" keys. The returned slice is
// suitable for inclusion in an Anthropic Messages API tools array. The order of
// entries is non-deterministic.
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

// ToolNames returns the names of all registered tools in non-deterministic
// order. The returned slice is newly allocated and safe to mutate.
func (r *ToolRegistry) ToolNames() []string {
	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	return names
}

// Execute runs the tool identified by name, passing ctx and input to its
// ToolFunc. If no tool with that name is registered, it returns a ToolResult
// with IsError set and an "Unknown tool" message rather than panicking.
func (r *ToolRegistry) Execute(ctx context.Context, name string, input map[string]any) ToolResult {
	entry, ok := r.tools[name]
	if !ok {
		return ToolResult{Output: "Unknown tool: " + name, IsError: true}
	}
	slog.Debug("executing tool", "name", name)
	return entry.ExecuteFn(ctx, input)
}

// RegisterTools resolves each entry in enabled against ToolGroups and
// BuiltinRegistry. An entry that matches a group name (e.g. "filesystem") is
// expanded to its member tool names; otherwise it is treated as a literal tool
// name. Resolved names that do not appear in BuiltinRegistry are logged as
// warnings and silently skipped.
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

package tools

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const defaultTimeout = 30

// BuiltinDef defines a built-in tool's metadata and execution function. Name
// and Description appear in the Anthropic tool schema sent to the LLM,
// InputSchema is the JSON Schema object describing accepted parameters, and
// ExecuteFn is the callback invoked at runtime with the parsed input.
type BuiltinDef struct {
	Name        string
	Description string
	InputSchema map[string]any
	ExecuteFn   ToolFunc
}

func bashTool(_ context.Context, input map[string]any) ToolResult {
	command, _ := input["command"].(string)
	timeout := defaultTimeout
	if t, ok := input["timeout"].(float64); ok {
		timeout = int(t)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sh", "-c", command)
	out, err := cmd.CombinedOutput()
	output := strings.TrimSpace(string(out))

	if ctx.Err() == context.DeadlineExceeded {
		return ToolResult{
			Output:  fmt.Sprintf("Command timed out after %d seconds", timeout),
			IsError: true,
		}
	}

	isError := err != nil
	if output == "" {
		if isError {
			output = fmt.Sprintf("Command failed with exit code %d", cmd.ProcessState.ExitCode())
		} else {
			output = "(no output)"
		}
	}

	return ToolResult{Output: output, IsError: isError}
}

func readFileTool(_ context.Context, input map[string]any) ToolResult {
	path, _ := input["path"].(string)
	offset := 0
	limit := 2000
	if o, ok := input["offset"].(float64); ok {
		offset = int(o)
	}
	if l, ok := input["limit"].(float64); ok {
		limit = int(l)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return ToolResult{Output: err.Error(), IsError: true}
	}

	lines := strings.Split(string(data), "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}

	end := offset + limit
	if end > len(lines) {
		end = len(lines)
	}
	selected := lines[offset:end]

	var numbered []string
	for i, line := range selected {
		numbered = append(numbered, fmt.Sprintf("%d\t%s", i+offset+1, line))
	}
	output := strings.Join(numbered, "\n")
	if offset+limit < len(lines) {
		output += fmt.Sprintf("\n... (%d more lines)", len(lines)-offset-limit)
	}

	return ToolResult{Output: output}
}

func writeFileTool(_ context.Context, input map[string]any) ToolResult {
	path, _ := input["path"].(string)
	content, _ := input["content"].(string)

	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return ToolResult{Output: err.Error(), IsError: true}
	}
	return ToolResult{Output: fmt.Sprintf("Written %d bytes to %s", len(content), path)}
}

func editFileTool(_ context.Context, input map[string]any) ToolResult {
	path, _ := input["path"].(string)
	oldText, _ := input["old_text"].(string)
	newText, _ := input["new_text"].(string)
	replaceAll, _ := input["replace_all"].(bool)

	data, err := os.ReadFile(path)
	if err != nil {
		return ToolResult{Output: err.Error(), IsError: true}
	}

	content := string(data)
	count := strings.Count(content, oldText)

	if count == 0 {
		return ToolResult{Output: "old_text not found in file", IsError: true}
	}
	if count > 1 && !replaceAll {
		return ToolResult{
			Output:  fmt.Sprintf("old_text found %d times — set replace_all to replace all occurrences", count),
			IsError: true,
		}
	}

	var newContent string
	if replaceAll {
		newContent = strings.ReplaceAll(content, oldText, newText)
	} else {
		newContent = strings.Replace(content, oldText, newText, 1)
	}

	if err := os.WriteFile(path, []byte(newContent), 0o644); err != nil {
		return ToolResult{Output: err.Error(), IsError: true}
	}

	replacements := 1
	if replaceAll {
		replacements = count
	}
	return ToolResult{Output: fmt.Sprintf("Replaced %d occurrence(s) in %s", replacements, path)}
}

func listDirectoryTool(_ context.Context, input map[string]any) ToolResult {
	path := "."
	if p, ok := input["path"].(string); ok && p != "" {
		path = p
	}

	info, err := os.Stat(path)
	if err != nil {
		return ToolResult{Output: err.Error(), IsError: true}
	}
	if !info.IsDir() {
		return ToolResult{Output: fmt.Sprintf("Not a directory: %s", path), IsError: true}
	}

	entries, err := os.ReadDir(path)
	if err != nil {
		return ToolResult{Output: err.Error(), IsError: true}
	}

	sort.Slice(entries, func(i, j int) bool {
		iDir := entries[i].IsDir()
		jDir := entries[j].IsDir()
		if iDir != jDir {
			return iDir
		}
		return strings.ToLower(entries[i].Name()) < strings.ToLower(entries[j].Name())
	})

	var lines []string
	for _, e := range entries {
		prefix := "[FILE]"
		if e.IsDir() {
			prefix = "[DIR]"
		}
		lines = append(lines, fmt.Sprintf("%s %s", prefix, e.Name()))
	}

	if len(lines) == 0 {
		return ToolResult{Output: "(empty directory)"}
	}
	return ToolResult{Output: strings.Join(lines, "\n")}
}

func searchFilesTool(_ context.Context, input map[string]any) ToolResult {
	path, _ := input["path"].(string)
	pattern, _ := input["pattern"].(string)

	var matches []string
	err := filepath.Walk(path, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		matched, err := filepath.Match(pattern, filepath.Base(p))
		if err != nil {
			return nil
		}
		if matched {
			matches = append(matches, p)
		}
		return nil
	})
	if err != nil {
		return ToolResult{Output: err.Error(), IsError: true}
	}

	sort.Strings(matches)
	if len(matches) == 0 {
		return ToolResult{Output: "No matches found"}
	}
	return ToolResult{Output: strings.Join(matches, "\n")}
}

// BuiltinRegistry maps tool names to their definitions for all built-in tools.
var BuiltinRegistry = map[string]BuiltinDef{
	"bash": {
		Name:        "bash",
		Description: "Execute a shell command and return its output.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"command": map[string]any{
					"type":        "string",
					"description": "The shell command to execute",
				},
				"timeout": map[string]any{
					"type":        "integer",
					"description": "Timeout in seconds (default 30)",
				},
			},
			"required": []string{"command"},
		},
		ExecuteFn: bashTool,
	},
	"read_file": {
		Name:        "read_file",
		Description: "Read the contents of a file with line numbers. Supports offset and limit for large files.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "Path to the file to read",
				},
				"offset": map[string]any{
					"type":        "integer",
					"description": "Line number to start reading from (0-based, default 0)",
				},
				"limit": map[string]any{
					"type":        "integer",
					"description": "Maximum number of lines to read (default 2000)",
				},
			},
			"required": []string{"path"},
		},
		ExecuteFn: readFileTool,
	},
	"write_file": {
		Name:        "write_file",
		Description: "Write content to a file, creating it if it doesn't exist or overwriting if it does.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "Path to the file to write",
				},
				"content": map[string]any{
					"type":        "string",
					"description": "Content to write to the file",
				},
			},
			"required": []string{"path", "content"},
		},
		ExecuteFn: writeFileTool,
	},
	"edit_file": {
		Name:        "edit_file",
		Description: "Replace text in a file. Finds old_text and replaces with new_text. Fails if old_text is not found or matches multiple times (unless replace_all is true).",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "Path to the file to edit",
				},
				"old_text": map[string]any{
					"type":        "string",
					"description": "The exact text to find and replace",
				},
				"new_text": map[string]any{
					"type":        "string",
					"description": "The text to replace it with",
				},
				"replace_all": map[string]any{
					"type":        "boolean",
					"description": "Replace all occurrences (default false)",
				},
			},
			"required": []string{"path", "old_text", "new_text"},
		},
		ExecuteFn: editFileTool,
	},
	"list_directory": {
		Name:        "list_directory",
		Description: "List the contents of a directory, showing [DIR] and [FILE] prefixes.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "Path to the directory to list (default: current directory)",
				},
			},
		},
		ExecuteFn: listDirectoryTool,
	},
	"search_files": {
		Name:        "search_files",
		Description: "Recursively search for files matching a glob pattern.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "Directory to search in",
				},
				"pattern": map[string]any{
					"type":        "string",
					"description": "Glob pattern to match (e.g. '*.py', '**/*.json')",
				},
			},
			"required": []string{"path", "pattern"},
		},
		ExecuteFn: searchFilesTool,
	},
}

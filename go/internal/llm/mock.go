package llm

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/uuid"
)

// MockLLMClient implements LLMClient with deterministic responses for testing.
// It produces canned replies and simulates tool_use responses when message text
// mentions a registered tool name. CallCount tracks the total number of
// Complete invocations, which is useful for asserting call sequences in tests.
type MockLLMClient struct {
	toolNames []string
	CallCount int
}

// NewMockLLMClient creates a mock LLM client. The toolNames slice determines
// which tool names the mock will recognize; when the last user message contains
// one of these names, Complete returns a tool_use content block for that tool
// instead of a plain text reply.
func NewMockLLMClient(toolNames []string) *MockLLMClient {
	return &MockLLMClient{toolNames: toolNames}
}

// Complete returns a deterministic response derived from the last entry in
// messages. The ctx, system, and tools parameters are accepted for interface
// compatibility but ignored. If messages is empty, a generic placeholder
// response is returned. If the last message contains a tool_result block, the
// response acknowledges the tool output. Otherwise, if the message text matches
// a registered tool name, a tool_use block with synthetic input is returned
// (stop reason "tool_use"); in all other cases a plain text echo is returned
// (stop reason "end_turn"). The method never returns an error.
func (m *MockLLMClient) Complete(_ context.Context, system, messages, tools []map[string]any) (*LLMResponse, error) {
	m.CallCount++

	if len(messages) == 0 {
		return &LLMResponse{
			Content:    []map[string]any{{"type": "text", "text": "Mock response to empty input"}},
			StopReason: "end_turn",
		}, nil
	}

	lastMessage := messages[len(messages)-1]
	lastText := extractText(lastMessage)

	if hasToolResultContent(lastMessage) {
		return &LLMResponse{
			Content:    []map[string]any{{"type": "text", "text": fmt.Sprintf("Tool result received: %s", lastText)}},
			StopReason: "end_turn",
		}, nil
	}

	for _, toolName := range m.toolNames {
		if strings.Contains(lastText, toolName) {
			return &LLMResponse{
				Content: []map[string]any{{
					"type":  "tool_use",
					"id":    fmt.Sprintf("toolu_%s", uuid.New().String()[:12]),
					"name":  toolName,
					"input": buildMockToolInput(toolName, lastText),
				}},
				StopReason: "tool_use",
			}, nil
		}
	}

	return &LLMResponse{
		Content:    []map[string]any{{"type": "text", "text": fmt.Sprintf("Mock response to: %s", lastText)}},
		StopReason: "end_turn",
	}, nil
}

func extractText(message map[string]any) string {
	content := message["content"]
	switch v := content.(type) {
	case string:
		return v
	case []any:
		var parts []string
		for _, block := range v {
			if m, ok := block.(map[string]any); ok {
				if t, _ := m["type"].(string); t == "text" {
					if text, _ := m["text"].(string); text != "" {
						parts = append(parts, text)
					}
				} else if t == "tool_result" {
					if c, _ := m["content"].(string); c != "" {
						parts = append(parts, c)
					}
				}
			}
		}
		return strings.Join(parts, " ")
	case []map[string]any:
		var parts []string
		for _, m := range v {
			if t, _ := m["type"].(string); t == "text" {
				if text, _ := m["text"].(string); text != "" {
					parts = append(parts, text)
				}
			} else if t == "tool_result" {
				if c, _ := m["content"].(string); c != "" {
					parts = append(parts, c)
				}
			}
		}
		return strings.Join(parts, " ")
	default:
		return fmt.Sprintf("%v", content)
	}
}

func hasToolResultContent(message map[string]any) bool {
	if role, _ := message["role"].(string); role == "tool" {
		return true
	}
	content := message["content"]
	switch v := content.(type) {
	case []any:
		for _, block := range v {
			if m, ok := block.(map[string]any); ok {
				if t, _ := m["type"].(string); t == "tool_result" {
					return true
				}
			}
		}
	case []map[string]any:
		for _, m := range v {
			if t, _ := m["type"].(string); t == "tool_result" {
				return true
			}
		}
	}
	return false
}

func buildMockToolInput(toolName, text string) map[string]any {
	switch toolName {
	case "bash":
		return map[string]any{"command": "echo 'mock command'"}
	case "read_file":
		return map[string]any{"path": "/tmp/mock_file.txt"}
	case "write_file":
		return map[string]any{"path": "/tmp/mock_file.txt", "content": "mock content"}
	case "edit_file":
		return map[string]any{"path": "/tmp/mock_file.txt", "old_text": "old", "new_text": "new"}
	case "list_directory":
		return map[string]any{"path": "/tmp"}
	case "search_files":
		return map[string]any{"path": "/tmp", "pattern": "*.txt"}
	default:
		return map[string]any{"input": text}
	}
}

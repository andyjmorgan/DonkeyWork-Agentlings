package llm

import (
	"context"
	"fmt"
	"log/slog"
)

// LLMResponse holds the parsed result of a single LLM completion call.
// Content contains the model's response as a slice of content blocks, where
// each block is a generic map (e.g. text blocks, tool_use blocks).
// StopReason indicates why the model stopped generating, such as "end_turn"
// when the model finished naturally or "tool_use" when it wants to invoke a tool.
type LLMResponse struct {
	Content    []map[string]any
	StopReason string
}

// LLMClient is the interface that all LLM backends must implement.
type LLMClient interface {
	// Complete sends the given conversation to the LLM and returns the
	// assistant's response. The system slice provides system-prompt text blocks,
	// messages carries the conversation history as role-tagged message maps, and
	// tools describes the available tool definitions (may be nil or empty when no
	// tools are registered). Returns an LLMResponse containing the model's
	// content blocks and stop reason, or an error if the call fails.
	Complete(ctx context.Context, system, messages, tools []map[string]any) (*LLMResponse, error)
}

// NewClient constructs an LLMClient for the named backend, which must be either
// "anthropic" or "mock". The apiKey and model configure the Anthropic backend
// (ignored for mock), and maxTokens caps the response length. The toolNames
// slice is passed to the mock backend so it knows which tool calls to simulate;
// it is ignored by the Anthropic backend. Returns an error if the backend name
// is unrecognized.
func NewClient(backend, apiKey, model string, maxTokens int, toolNames []string) (LLMClient, error) {
	switch backend {
	case "mock":
		slog.Info("using mock LLM backend")
		return NewMockLLMClient(toolNames), nil
	case "anthropic":
		slog.Info("using Anthropic LLM backend", "model", model)
		return NewAnthropicLLMClient(apiKey, model, maxTokens)
	default:
		return nil, fmt.Errorf("unknown LLM backend: %q", backend)
	}
}

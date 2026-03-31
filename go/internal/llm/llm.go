package llm

import (
	"context"
	"fmt"
	"log/slog"
)

type LLMResponse struct {
	Content    []map[string]any
	StopReason string
}

type LLMClient interface {
	Complete(ctx context.Context, system, messages, tools []map[string]any) (*LLMResponse, error)
}

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

package loop

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/andyjmorgan/agentlings-go/internal/config"
	"github.com/andyjmorgan/agentlings-go/internal/llm"
	"github.com/andyjmorgan/agentlings-go/internal/models"
	"github.com/andyjmorgan/agentlings-go/internal/prompt"
	"github.com/andyjmorgan/agentlings-go/internal/store"
	"github.com/andyjmorgan/agentlings-go/internal/tools"
	"github.com/google/uuid"
)

// LoopResult holds the final assistant response and the context ID for a completed
// message loop turn. ContextID is the conversation handle (server-generated or
// externally supplied) that can be reused for follow-up turns. Content contains
// the LLM's terminal response as raw Anthropic content blocks.
type LoopResult struct {
	ContextID string
	Content   []map[string]any
}

// MessageLoop orchestrates the agent turn loop: journal replay, LLM calls, and
// tool execution. It holds references to the journal store for conversation
// persistence, an LLM client for completions, a tool registry for executing
// tool calls, and the pre-built system prompt content blocks.
type MessageLoop struct {
	config *config.Config
	store  *store.JournalStore
	llm    llm.LLMClient
	tools  *tools.ToolRegistry
	system []map[string]any
}

// New creates a MessageLoop wired to the given configuration, journal store,
// LLM client, and tool registry. The cfg provides agent identity used to build
// the system prompt at construction time. The store s handles JSONL journal
// persistence, l provides LLM completions, and t supplies the available tool
// schemas and execution logic. The returned MessageLoop is safe to use
// concurrently from multiple goroutines since it delegates state to the
// underlying store.
func New(cfg *config.Config, s *store.JournalStore, l llm.LLMClient, t *tools.ToolRegistry) *MessageLoop {
	return &MessageLoop{
		config: cfg,
		store:  s,
		llm:    l,
		tools:  t,
		system: prompt.BuildSystemPrompt(cfg),
	}
}

// ProcessMessage runs a full agent turn for the given user message. The ctx
// controls cancellation and timeouts for LLM calls. The text parameter is the
// user's input message. If contextID is empty, a new UUID-based context is
// created and its journal initialised; if non-empty but not yet known to the
// store, the journal is created using the caller-supplied ID. The via string
// records the originating protocol (e.g. "a2a" or "mcp") in journal entries.
// It returns a LoopResult containing the final assistant content blocks and the
// context ID, or an error if any journal or LLM operation fails.
func (ml *MessageLoop) ProcessMessage(ctx context.Context, text, contextID, via string) (*LoopResult, error) {
	if contextID == "" {
		contextID = uuid.New().String()
		if err := ml.store.Create(contextID); err != nil {
			return nil, fmt.Errorf("create context: %w", err)
		}
		slog.Info("new context", "ctx", contextID)
	} else if !ml.store.Exists(contextID) {
		if err := ml.store.Create(contextID); err != nil {
			return nil, fmt.Errorf("create context: %w", err)
		}
		slog.Info("new context (external id)", "ctx", contextID)
	}

	userEntry := models.NewMessageEntry(contextID, "user", via, []map[string]any{
		{"type": "text", "text": text},
	})
	if err := ml.store.Append(contextID, userEntry); err != nil {
		return nil, fmt.Errorf("append user message: %w", err)
	}

	messages, err := ml.store.Replay(contextID)
	if err != nil {
		return nil, fmt.Errorf("replay: %w", err)
	}

	toolSchemas := ml.tools.ListSchemas()

	for {
		response, err := ml.llm.Complete(ctx, ml.system, messages, toolSchemas)
		if err != nil {
			return nil, fmt.Errorf("llm complete: %w", err)
		}

		hasToolUse := false
		for _, block := range response.Content {
			if t, _ := block["type"].(string); t == "tool_use" {
				hasToolUse = true
				break
			}
		}

		if hasToolUse {
			assistantEntry := models.NewMessageEntry(contextID, "assistant", via, response.Content)
			if err := ml.store.Append(contextID, assistantEntry); err != nil {
				return nil, fmt.Errorf("append assistant tool_use: %w", err)
			}

			var toolResults []map[string]any
			for _, block := range response.Content {
				if t, _ := block["type"].(string); t == "tool_use" {
					name, _ := block["name"].(string)
					input, _ := block["input"].(map[string]any)
					id, _ := block["id"].(string)

					result := ml.tools.Execute(ctx, name, input)
					toolResults = append(toolResults, map[string]any{
						"type":        "tool_result",
						"tool_use_id": id,
						"content":     result.Output,
						"is_error":    result.IsError,
					})
				}
			}

			toolEntry := models.NewMessageEntry(contextID, "user", via, toolResults)
			if err := ml.store.Append(contextID, toolEntry); err != nil {
				return nil, fmt.Errorf("append tool results: %w", err)
			}

			messages, err = ml.store.Replay(contextID)
			if err != nil {
				return nil, fmt.Errorf("replay after tools: %w", err)
			}
			continue
		}

		for _, block := range response.Content {
			if t, _ := block["type"].(string); t == "compaction" {
				content, _ := block["content"].(string)
				compEntry := models.NewCompactionEntry(contextID, content)
				if err := ml.store.Append(contextID, compEntry); err != nil {
					return nil, fmt.Errorf("append compaction: %w", err)
				}
				slog.Info("compaction marker stored", "ctx", contextID)
			}
		}

		finalEntry := models.NewMessageEntry(contextID, "assistant", via, response.Content)
		if err := ml.store.Append(contextID, finalEntry); err != nil {
			return nil, fmt.Errorf("append final response: %w", err)
		}

		return &LoopResult{
			ContextID: contextID,
			Content:   response.Content,
		}, nil
	}
}

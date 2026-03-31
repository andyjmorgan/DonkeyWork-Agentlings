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

type LoopResult struct {
	ContextID string
	Content   []map[string]any
}

type MessageLoop struct {
	config *config.Config
	store  *store.JournalStore
	llm    llm.LLMClient
	tools  *tools.ToolRegistry
	system []map[string]any
}

func New(cfg *config.Config, s *store.JournalStore, l llm.LLMClient, t *tools.ToolRegistry) *MessageLoop {
	return &MessageLoop{
		config: cfg,
		store:  s,
		llm:    l,
		tools:  t,
		system: prompt.BuildSystemPrompt(cfg),
	}
}

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

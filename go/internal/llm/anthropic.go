package llm

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

type AnthropicLLMClient struct {
	client    anthropic.Client
	model     string
	maxTokens int
}

func NewAnthropicLLMClient(apiKey, model string, maxTokens int) (*AnthropicLLMClient, error) {
	client := anthropic.NewClient(option.WithAPIKey(apiKey))
	return &AnthropicLLMClient{
		client:    client,
		model:     model,
		maxTokens: maxTokens,
	}, nil
}

func (a *AnthropicLLMClient) Complete(ctx context.Context, system, messages, tools []map[string]any) (*LLMResponse, error) {
	systemBlocks, err := toSystemBlocks(system)
	if err != nil {
		return nil, fmt.Errorf("build system blocks: %w", err)
	}

	messageParams, err := toMessageParams(messages)
	if err != nil {
		return nil, fmt.Errorf("build message params: %w", err)
	}

	params := anthropic.MessageNewParams{
		Model:     a.model,
		MaxTokens: int64(a.maxTokens),
		System:    systemBlocks,
		Messages:  messageParams,
	}

	if len(tools) > 0 {
		toolParams, err := toToolParams(tools)
		if err != nil {
			return nil, fmt.Errorf("build tool params: %w", err)
		}
		params.Tools = toolParams
	}

	response, err := a.client.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("anthropic messages.create: %w", err)
	}

	content := make([]map[string]any, 0, len(response.Content))
	for _, block := range response.Content {
		blockJSON, err := json.Marshal(block)
		if err != nil {
			return nil, fmt.Errorf("marshal content block: %w", err)
		}
		var blockMap map[string]any
		if err := json.Unmarshal(blockJSON, &blockMap); err != nil {
			return nil, fmt.Errorf("unmarshal content block: %w", err)
		}
		content = append(content, blockMap)
	}

	return &LLMResponse{
		Content:    content,
		StopReason: string(response.StopReason),
	}, nil
}

func toSystemBlocks(system []map[string]any) ([]anthropic.TextBlockParam, error) {
	blocks := make([]anthropic.TextBlockParam, 0, len(system))
	for _, s := range system {
		text, _ := s["text"].(string)
		block := anthropic.TextBlockParam{
			Text: text,
			Type: "text",
		}
		if cc, ok := s["cache_control"].(map[string]any); ok {
			if t, _ := cc["type"].(string); t == "ephemeral" {
				block.CacheControl = anthropic.NewCacheControlEphemeralParam()
			}
		}
		blocks = append(blocks, block)
	}
	return blocks, nil
}

func toMessageParams(messages []map[string]any) ([]anthropic.MessageParam, error) {
	params := make([]anthropic.MessageParam, 0, len(messages))
	for _, msg := range messages {
		data, err := json.Marshal(msg)
		if err != nil {
			return nil, fmt.Errorf("marshal message: %w", err)
		}
		var param anthropic.MessageParam
		if err := json.Unmarshal(data, &param); err != nil {
			return nil, fmt.Errorf("unmarshal message param: %w", err)
		}
		params = append(params, param)
	}
	return params, nil
}

func toToolParams(tools []map[string]any) ([]anthropic.ToolUnionParam, error) {
	params := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		name, _ := tool["name"].(string)
		description, _ := tool["description"].(string)
		inputSchema, _ := tool["input_schema"].(map[string]any)

		schema := anthropic.ToolInputSchemaParam{
			Properties: inputSchema["properties"],
		}
		if req, ok := inputSchema["required"].([]string); ok {
			schema.Required = req
		} else if reqAny, ok := inputSchema["required"].([]any); ok {
			for _, r := range reqAny {
				if s, ok := r.(string); ok {
					schema.Required = append(schema.Required, s)
				}
			}
		}

		params = append(params, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        name,
				Description: anthropic.String(description),
				InputSchema: schema,
			},
		})
	}
	return params, nil
}

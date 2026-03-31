package protocol

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/andyjmorgan/agentlings-go/internal/loop"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type MCPToolInput struct {
	Message   string `json:"message" jsonschema:"Natural language request to the agent"`
	ContextID string `json:"contextId,omitempty" jsonschema:"Context ID from a previous response. Include to continue an existing conversation."`
}

type MCPToolOutput struct {
	ContextID string `json:"contextId"`
	Message   string `json:"message"`
}

func CreateMCPServer(ml *loop.MessageLoop, agentCard *a2a.AgentCard) *mcp.Server {
	server := mcp.NewServer(&mcp.Implementation{
		Name:    agentCard.Name,
		Version: agentCard.Version,
	}, nil)

	toolDescription := fmt.Sprintf("%s. Returns a contextId for multi-turn conversations — pass it back to continue.", agentCard.Description)

	mcp.AddTool(server, &mcp.Tool{
		Name:        agentCard.Name,
		Description: toolDescription,
	}, func(ctx context.Context, req *mcp.CallToolRequest, input MCPToolInput) (*mcp.CallToolResult, MCPToolOutput, error) {
		result, err := ml.ProcessMessage(ctx, input.Message, input.ContextID, "mcp")
		if err != nil {
			return &mcp.CallToolResult{
				Content: []mcp.Content{&mcp.TextContent{Text: fmt.Sprintf("Error: %v", err)}},
				IsError: true,
			}, MCPToolOutput{}, nil
		}

		responseText := extractTextFromContent(result.Content)
		output := MCPToolOutput{
			ContextID: result.ContextID,
			Message:   responseText,
		}

		outputJSON, _ := json.Marshal(output)
		return &mcp.CallToolResult{
			Content: []mcp.Content{&mcp.TextContent{Text: string(outputJSON)}},
		}, output, nil
	})

	return server
}

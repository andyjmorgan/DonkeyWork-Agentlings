package protocol

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/andyjmorgan/agentlings-go/internal/loop"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// MCPToolInput represents the parameters accepted by the agent's MCP tool.
// Message is the natural-language request forwarded to the agent. ContextID is
// optional; when provided it resumes an existing conversation, and when empty
// the message loop creates a new context.
type MCPToolInput struct {
	Message   string `json:"message" jsonschema:"Natural language request to the agent"`
	ContextID string `json:"contextId,omitempty" jsonschema:"Context ID from a previous response. Include to continue an existing conversation."`
}

// MCPToolOutput represents the structured response returned by the agent's MCP tool.
// ContextID is the server-assigned conversation identifier that callers should
// pass back in subsequent requests to maintain continuity. Message contains the
// agent's natural-language response text.
type MCPToolOutput struct {
	ContextID string `json:"contextId"`
	Message   string `json:"message"`
}

// CreateMCPServer creates an MCP server with a single tool derived from the
// agent card. The ml handles all message processing, and agentCard supplies the
// tool's name, description, and the server's advertised version. The resulting
// server exposes exactly one tool whose name matches the agent card's Name; the
// tool description is the card's Description with a note about contextId for
// multi-turn conversations. On success the tool handler returns a JSON-encoded
// MCPToolOutput inside the CallToolResult content. On failure it returns a
// CallToolResult with IsError set and the error text as content, without
// propagating the error to the MCP framework.
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

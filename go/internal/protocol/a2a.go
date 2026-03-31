package protocol

import (
	"context"
	"iter"
	"log/slog"
	"strings"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/andyjmorgan/agentlings-go/internal/loop"
)

type AgentlingExecutor struct {
	loop *loop.MessageLoop
}

func NewAgentlingExecutor(l *loop.MessageLoop) *AgentlingExecutor {
	return &AgentlingExecutor{loop: l}
}

func (e *AgentlingExecutor) Execute(ctx context.Context, execCtx *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		userText := extractUserInput(execCtx.Message)
		contextID := execCtx.ContextID

		slog.Debug("a2a execute", "contextID", contextID, "text", truncate(userText, 100))

		result, err := e.loop.ProcessMessage(ctx, userText, contextID, "a2a")
		if err != nil {
			slog.Error("error processing A2A message", "error", err)
			errMsg := a2a.NewMessageForTask(a2a.MessageRoleAgent, execCtx,
				a2a.NewTextPart("Internal error processing request."))
			yield(errMsg, nil)
			return
		}

		responseText := extractTextFromContent(result.Content)
		slog.Debug("a2a response", "contextID", result.ContextID, "text", truncate(responseText, 100))

		msg := a2a.NewMessageForTask(a2a.MessageRoleAgent, execCtx,
			a2a.NewTextPart(responseText))
		yield(msg, nil)
	}
}

func (e *AgentlingExecutor) Cancel(ctx context.Context, execCtx *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		msg := a2a.NewMessageForTask(a2a.MessageRoleAgent, execCtx,
			a2a.NewTextPart("Cancellation is not supported."))
		yield(msg, nil)
	}
}

func extractUserInput(msg *a2a.Message) string {
	if msg == nil {
		return ""
	}
	var parts []string
	for _, part := range msg.Parts {
		if text, ok := part.Content.(a2a.Text); ok {
			parts = append(parts, string(text))
		}
	}
	return strings.Join(parts, "\n")
}

func extractTextFromContent(content []map[string]any) string {
	var parts []string
	for _, block := range content {
		if t, _ := block["type"].(string); t == "text" {
			if text, _ := block["text"].(string); text != "" {
				parts = append(parts, text)
			}
		}
	}
	return strings.Join(parts, "\n")
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

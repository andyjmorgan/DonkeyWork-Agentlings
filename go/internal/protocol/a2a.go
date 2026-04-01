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

// AgentlingExecutor bridges the A2A protocol to the shared message loop.
// It implements the a2asrv executor interface so the A2A server can delegate
// incoming JSON-RPC requests to the agentling's core processing pipeline.
type AgentlingExecutor struct {
	loop *loop.MessageLoop
}

// NewAgentlingExecutor returns an executor that dispatches A2A requests through
// the provided message loop. The loop must be fully initialised before any
// Execute or Cancel calls are made.
func NewAgentlingExecutor(l *loop.MessageLoop) *AgentlingExecutor {
	return &AgentlingExecutor{loop: l}
}

// Execute processes an incoming A2A message through the message loop and yields
// the agent's response. It extracts user-facing text parts from the message
// inside execCtx, forwards them to the message loop identified by the
// execCtx.ContextID, and produces an iterator that yields exactly one
// a2a.Event containing the agent's reply. If the message loop returns an error,
// a generic error message is yielded instead so the caller never sees a nil
// event. The ctx controls cancellation of the underlying LLM call.
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

// Cancel yields a single event telling the caller that cancellation is not
// supported by this agent. The returned iterator always yields exactly one
// a2a.Event with no error. Both ctx and execCtx are accepted to satisfy the
// executor interface but ctx is unused since no work is performed.
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

package llm

import (
	"context"
	"strings"
	"testing"
)

func TestMockEchoResponse(t *testing.T) {
	m := NewMockLLMClient(nil)
	resp, err := m.Complete(context.Background(), nil,
		[]map[string]any{{"role": "user", "content": []map[string]any{{"type": "text", "text": "hello"}}}},
		nil,
	)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("StopReason = %q, want end_turn", resp.StopReason)
	}
	text, _ := resp.Content[0]["text"].(string)
	if !strings.Contains(text, "hello") {
		t.Errorf("expected echo of input, got %q", text)
	}
}

func TestMockToolUseResponse(t *testing.T) {
	m := NewMockLLMClient([]string{"bash"})
	resp, err := m.Complete(context.Background(), nil,
		[]map[string]any{{"role": "user", "content": []map[string]any{{"type": "text", "text": "run bash command"}}}},
		nil,
	)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("StopReason = %q, want tool_use", resp.StopReason)
	}
	if resp.Content[0]["type"] != "tool_use" {
		t.Errorf("content type = %v, want tool_use", resp.Content[0]["type"])
	}
	if resp.Content[0]["name"] != "bash" {
		t.Errorf("tool name = %v, want bash", resp.Content[0]["name"])
	}
}

func TestMockToolResultResponse(t *testing.T) {
	m := NewMockLLMClient([]string{"bash"})
	resp, err := m.Complete(context.Background(), nil,
		[]map[string]any{{"role": "tool", "content": "command output"}},
		nil,
	)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("StopReason = %q, want end_turn", resp.StopReason)
	}
	text, _ := resp.Content[0]["text"].(string)
	if !strings.Contains(text, "Tool result received") {
		t.Errorf("expected tool result acknowledgment, got %q", text)
	}
}

func TestMockCallCount(t *testing.T) {
	m := NewMockLLMClient(nil)
	if m.CallCount != 0 {
		t.Errorf("initial CallCount = %d, want 0", m.CallCount)
	}

	m.Complete(context.Background(), nil, []map[string]any{{"role": "user", "content": "hi"}}, nil)
	m.Complete(context.Background(), nil, []map[string]any{{"role": "user", "content": "hi"}}, nil)

	if m.CallCount != 2 {
		t.Errorf("CallCount = %d, want 2", m.CallCount)
	}
}

func TestExtractTextString(t *testing.T) {
	msg := map[string]any{"content": "plain string"}
	if got := extractText(msg); got != "plain string" {
		t.Errorf("got %q, want %q", got, "plain string")
	}
}

func TestExtractTextBlocks(t *testing.T) {
	msg := map[string]any{
		"content": []any{
			map[string]any{"type": "text", "text": "hello"},
			map[string]any{"type": "text", "text": "world"},
		},
	}
	got := extractText(msg)
	if got != "hello world" {
		t.Errorf("got %q, want %q", got, "hello world")
	}
}

func TestNewClientMock(t *testing.T) {
	client, err := NewClient("mock", "", "", 0, nil)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if _, ok := client.(*MockLLMClient); !ok {
		t.Errorf("expected *MockLLMClient, got %T", client)
	}
}

func TestNewClientUnknown(t *testing.T) {
	_, err := NewClient("unknown", "", "", 0, nil)
	if err == nil {
		t.Error("expected error for unknown backend")
	}
}

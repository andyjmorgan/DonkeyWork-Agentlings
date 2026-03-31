package prompt

import (
	"strings"
	"testing"

	"github.com/andyjmorgan/agentlings-go/internal/config"
)

func TestBuildSystemPromptDefault(t *testing.T) {
	cfg := &config.Config{
		Definition: config.AgentDefinition{
			Name:        "test-agent",
			Description: "A test agent",
		},
	}

	blocks := BuildSystemPrompt(cfg)
	if len(blocks) != 1 {
		t.Fatalf("expected 1 block, got %d", len(blocks))
	}

	text, _ := blocks[0]["text"].(string)
	if !strings.Contains(text, "test-agent") {
		t.Error("default prompt should contain agent name")
	}
	if !strings.Contains(text, "A test agent") {
		t.Error("default prompt should contain description")
	}

	cc, _ := blocks[0]["cache_control"].(map[string]any)
	if cc["type"] != "ephemeral" {
		t.Error("cache_control type should be ephemeral")
	}
}

func TestBuildSystemPromptCustom(t *testing.T) {
	cfg := &config.Config{
		Definition: config.AgentDefinition{
			Name:         "custom-agent",
			Description:  "Custom",
			SystemPrompt: "You are a custom agent.\nDo custom things.",
		},
	}

	blocks := BuildSystemPrompt(cfg)
	text, _ := blocks[0]["text"].(string)
	if text != "You are a custom agent.\nDo custom things." {
		t.Errorf("expected custom prompt, got %q", text)
	}
}

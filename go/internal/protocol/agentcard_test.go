package protocol

import (
	"testing"

	"github.com/andyjmorgan/agentlings-go/internal/config"
)

func TestGenerateAgentCardDefaults(t *testing.T) {
	cfg := &config.Config{
		AgentHost: "0.0.0.0",
		AgentPort: 8420,
		Definition: config.AgentDefinition{
			Name:        "test-agent",
			Description: "A test agent",
		},
	}

	card := GenerateAgentCard(cfg)

	if card.Name != "test-agent" {
		t.Errorf("Name = %q, want %q", card.Name, "test-agent")
	}
	if card.Description != "A test agent" {
		t.Errorf("Description = %q, want %q", card.Description, "A test agent")
	}
	if card.Version != "0.1.0" {
		t.Errorf("Version = %q, want %q", card.Version, "0.1.0")
	}
	if len(card.Skills) != 1 {
		t.Fatalf("Skills len = %d, want 1", len(card.Skills))
	}
	if card.Skills[0].ID != "test-agent" {
		t.Errorf("Skill ID = %q, want %q", card.Skills[0].ID, "test-agent")
	}
}

func TestGenerateAgentCardExternalURL(t *testing.T) {
	cfg := &config.Config{
		AgentHost:        "0.0.0.0",
		AgentPort:        8420,
		AgentExternalURL: "https://example.com",
		Definition: config.AgentDefinition{
			Name:        "test-agent",
			Description: "A test agent",
		},
	}

	card := GenerateAgentCard(cfg)

	if len(card.SupportedInterfaces) == 0 {
		t.Fatal("expected SupportedInterfaces")
	}
}

func TestGenerateAgentCardWithSkills(t *testing.T) {
	cfg := &config.Config{
		AgentHost: "0.0.0.0",
		AgentPort: 8420,
		Definition: config.AgentDefinition{
			Name:        "skilled-agent",
			Description: "An agent with skills",
			Skills: []config.SkillConfig{
				{ID: "coding", Name: "Coding", Description: "Write code", Tags: []string{"code"}},
				{ID: "review", Name: "Review", Description: "Review code", Tags: []string{"review"}},
			},
		},
	}

	card := GenerateAgentCard(cfg)

	if len(card.Skills) != 2 {
		t.Fatalf("Skills len = %d, want 2", len(card.Skills))
	}
	if card.Skills[0].ID != "coding" {
		t.Errorf("Skill[0].ID = %q, want %q", card.Skills[0].ID, "coding")
	}
	if card.Skills[1].ID != "review" {
		t.Errorf("Skill[1].ID = %q, want %q", card.Skills[1].ID, "review")
	}
}

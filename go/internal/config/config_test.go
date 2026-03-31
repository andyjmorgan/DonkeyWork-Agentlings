package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadDefaults(t *testing.T) {
	for _, k := range []string{
		"ANTHROPIC_API_KEY", "AGENT_API_KEY", "AGENT_MODEL", "AGENT_MAX_TOKENS",
		"AGENT_HOST", "AGENT_PORT", "AGENT_DATA_DIR", "AGENT_LOG_LEVEL",
		"AGENT_LLM_BACKEND", "AGENT_EXTERNAL_URL", "AGENT_CONFIG",
	} {
		os.Unsetenv(k)
	}

	tmp := t.TempDir()
	os.Setenv("AGENT_DATA_DIR", filepath.Join(tmp, "data"))
	defer os.Unsetenv("AGENT_DATA_DIR")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}

	if cfg.AgentModel != "claude-sonnet-4-6" {
		t.Errorf("AgentModel = %q, want %q", cfg.AgentModel, "claude-sonnet-4-6")
	}
	if cfg.AgentMaxTokens != 4096 {
		t.Errorf("AgentMaxTokens = %d, want %d", cfg.AgentMaxTokens, 4096)
	}
	if cfg.AgentPort != 8420 {
		t.Errorf("AgentPort = %d, want %d", cfg.AgentPort, 8420)
	}
	if cfg.Definition.Name != "agentling" {
		t.Errorf("Definition.Name = %q, want %q", cfg.Definition.Name, "agentling")
	}
}

func TestLoadEnvOverrides(t *testing.T) {
	tmp := t.TempDir()
	os.Setenv("AGENT_DATA_DIR", filepath.Join(tmp, "data"))
	os.Setenv("AGENT_MODEL", "claude-opus-4-6")
	os.Setenv("AGENT_PORT", "9000")
	defer func() {
		os.Unsetenv("AGENT_DATA_DIR")
		os.Unsetenv("AGENT_MODEL")
		os.Unsetenv("AGENT_PORT")
	}()

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}

	if cfg.AgentModel != "claude-opus-4-6" {
		t.Errorf("AgentModel = %q, want %q", cfg.AgentModel, "claude-opus-4-6")
	}
	if cfg.AgentPort != 9000 {
		t.Errorf("AgentPort = %d, want %d", cfg.AgentPort, 9000)
	}
}

func TestLoadYAMLDefinition(t *testing.T) {
	tmp := t.TempDir()
	yamlPath := filepath.Join(tmp, "agent.yaml")
	yamlContent := `name: test-agent
description: A test agent
tools:
  - bash
  - filesystem
skills:
  - id: coding
    name: Coding Assistant
    description: Helps write code
    tags: [code, dev]
system_prompt: |
  You are a test agent.
`
	if err := os.WriteFile(yamlPath, []byte(yamlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	os.Setenv("AGENT_CONFIG", yamlPath)
	os.Setenv("AGENT_DATA_DIR", filepath.Join(tmp, "data"))
	defer func() {
		os.Unsetenv("AGENT_CONFIG")
		os.Unsetenv("AGENT_DATA_DIR")
	}()

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}

	if cfg.Definition.Name != "test-agent" {
		t.Errorf("Name = %q, want %q", cfg.Definition.Name, "test-agent")
	}
	if cfg.Definition.Description != "A test agent" {
		t.Errorf("Description = %q, want %q", cfg.Definition.Description, "A test agent")
	}
	if len(cfg.Definition.Tools) != 2 {
		t.Errorf("Tools len = %d, want %d", len(cfg.Definition.Tools), 2)
	}
	if len(cfg.Definition.Skills) != 1 {
		t.Errorf("Skills len = %d, want %d", len(cfg.Definition.Skills), 1)
	}
	if cfg.Definition.Skills[0].ID != "coding" {
		t.Errorf("Skill ID = %q, want %q", cfg.Definition.Skills[0].ID, "coding")
	}
	if cfg.Definition.SystemPrompt == "" {
		t.Error("SystemPrompt should not be empty")
	}
}

func TestLoadCreatesDataDir(t *testing.T) {
	tmp := t.TempDir()
	dataDir := filepath.Join(tmp, "nested", "data")
	os.Setenv("AGENT_DATA_DIR", dataDir)
	defer os.Unsetenv("AGENT_DATA_DIR")

	_, err := Load()
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}

	if _, err := os.Stat(dataDir); os.IsNotExist(err) {
		t.Error("data dir was not created")
	}
}

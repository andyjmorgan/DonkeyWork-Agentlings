package config

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"

	"github.com/joho/godotenv"
	"gopkg.in/yaml.v3"
)

type SkillConfig struct {
	ID          string   `yaml:"id"`
	Name        string   `yaml:"name"`
	Description string   `yaml:"description"`
	Tags        []string `yaml:"tags"`
}

type AgentDefinition struct {
	Name         string        `yaml:"name"`
	Description  string        `yaml:"description"`
	Tools        []string      `yaml:"tools"`
	Skills       []SkillConfig `yaml:"skills"`
	SystemPrompt string        `yaml:"system_prompt"`
}

type Config struct {
	AnthropicAPIKey string
	AgentAPIKey     string
	AgentModel      string
	AgentMaxTokens  int
	AgentHost       string
	AgentPort       int
	AgentDataDir    string
	AgentLogLevel   string
	AgentLLMBackend string
	AgentExternalURL string
	AgentConfig     string

	Definition AgentDefinition
}

func Load() (*Config, error) {
	_ = godotenv.Load()

	cfg := &Config{
		AnthropicAPIKey:  envOrDefault("ANTHROPIC_API_KEY", ""),
		AgentAPIKey:      envOrDefault("AGENT_API_KEY", ""),
		AgentModel:       envOrDefault("AGENT_MODEL", "claude-sonnet-4-6"),
		AgentMaxTokens:   envIntOrDefault("AGENT_MAX_TOKENS", 4096),
		AgentHost:        envOrDefault("AGENT_HOST", "0.0.0.0"),
		AgentPort:        envIntOrDefault("AGENT_PORT", 8420),
		AgentDataDir:     envOrDefault("AGENT_DATA_DIR", "./data"),
		AgentLogLevel:    envOrDefault("AGENT_LOG_LEVEL", "INFO"),
		AgentLLMBackend:  envOrDefault("AGENT_LLM_BACKEND", "anthropic"),
		AgentExternalURL: envOrDefault("AGENT_EXTERNAL_URL", ""),
		AgentConfig:      envOrDefault("AGENT_CONFIG", ""),
		Definition: AgentDefinition{
			Name:        "agentling",
			Description: "A lightweight AI agent",
		},
	}

	if err := os.MkdirAll(cfg.AgentDataDir, 0o755); err != nil {
		return nil, fmt.Errorf("create data dir: %w", err)
	}

	if cfg.AgentConfig != "" {
		def, err := loadDefinition(cfg.AgentConfig)
		if err != nil {
			return nil, fmt.Errorf("load agent config: %w", err)
		}
		cfg.Definition = *def
	}

	return cfg, nil
}

func LoadFromEnv(env map[string]string) (*Config, error) {
	for k, v := range env {
		os.Setenv(k, v)
	}
	return Load()
}

func loadDefinition(path string) (*AgentDefinition, error) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("resolve path: %w", err)
	}

	data, err := os.ReadFile(absPath)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", absPath, err)
	}

	var def AgentDefinition
	if err := yaml.Unmarshal(data, &def); err != nil {
		return nil, fmt.Errorf("parse %s: %w", absPath, err)
	}

	slog.Info("loaded agent definition", "path", absPath)
	return &def, nil
}

func envOrDefault(key, defaultVal string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return defaultVal
}

func envIntOrDefault(key string, defaultVal int) int {
	v, ok := os.LookupEnv(key)
	if !ok {
		return defaultVal
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return defaultVal
	}
	return n
}

// Package config loads and validates agentling runtime configuration from environment variables and YAML definitions.
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

// SkillConfig defines a skill that the agent can advertise and execute.
// ID is a unique machine-readable identifier for the skill, Name is its
// human-readable display name, and Description explains what the skill does.
// Tags provide optional classification labels used for discovery and filtering.
type SkillConfig struct {
	ID          string   `yaml:"id"`
	Name        string   `yaml:"name"`
	Description string   `yaml:"description"`
	Tags        []string `yaml:"tags"`
}

// AgentDefinition holds the agent's identity, capabilities, and system prompt
// as declared in a YAML config file. Name and Description populate the Agent
// Card and MCP tool metadata. Tools lists the tool group names the agent is
// allowed to use at runtime. Skills enumerates the discrete capabilities the
// agent advertises to callers. SystemPrompt, when non-empty, overrides the
// default system prompt sent to the LLM.
type AgentDefinition struct {
	Name         string        `yaml:"name"`
	Description  string        `yaml:"description"`
	Tools        []string      `yaml:"tools"`
	Skills       []SkillConfig `yaml:"skills"`
	SystemPrompt string        `yaml:"system_prompt"`
}

// Config aggregates all runtime configuration sourced from environment
// variables and the optional YAML agent definition. AnthropicAPIKey and
// AgentAPIKey hold the Anthropic API credential and the inbound request
// authentication key respectively. AgentModel and AgentMaxTokens control the
// LLM model identifier and response token ceiling. AgentHost and AgentPort
// set the HTTP listen address (defaults 0.0.0.0:8420). AgentDataDir is the
// directory used for JSONL journal storage and is created automatically if it
// does not exist. AgentLogLevel sets the slog level string (e.g. "INFO",
// "DEBUG"). AgentLLMBackend selects the LLM backend — "anthropic" for the
// real API or "mock" for testing. AgentExternalURL, when set, is the
// publicly-reachable base URL advertised in the Agent Card. AgentConfig
// points to an optional YAML file whose contents are unmarshalled into
// Definition, overriding the default agent identity.
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

// Load reads configuration from environment variables (and an optional .env
// file in the working directory), applies sensible defaults, creates the data
// directory if needed, and optionally overlays a YAML agent definition when
// AGENT_CONFIG is set. It returns the fully populated Config or an error if
// the data directory cannot be created or the YAML file cannot be read/parsed.
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

// LoadFromEnv injects the given key-value pairs into the process environment
// and then delegates to Load. The env map keys are environment variable names
// (e.g. "AGENT_PORT") and the values are their desired settings. This is
// primarily useful in tests where the caller needs deterministic configuration
// without touching .env files. It returns the same Config and error semantics
// as Load.
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

package prompt

import (
	"fmt"

	"github.com/andyjmorgan/agentlings-go/internal/config"
)

// BuildSystemPrompt returns the system prompt as a slice of Anthropic content
// blocks suitable for passing directly to the Messages API. The cfg provides the
// agent definition: if cfg.Definition.SystemPrompt is non-empty it is used
// verbatim; otherwise a default prompt is generated from the agent's Name and
// Description. The returned block includes an ephemeral cache_control directive
// so the prompt benefits from prompt caching across turns.
func BuildSystemPrompt(cfg *config.Config) []map[string]any {
	var text string
	if cfg.Definition.SystemPrompt != "" {
		text = cfg.Definition.SystemPrompt
	} else {
		text = defaultPrompt(cfg)
	}

	return []map[string]any{
		{
			"type":          "text",
			"text":          text,
			"cache_control": map[string]any{"type": "ephemeral"},
		},
	}
}

func defaultPrompt(cfg *config.Config) string {
	return fmt.Sprintf(`You are %s, %s.

Use the tools available to you to accomplish tasks. When a tool returns output, incorporate it into your response.

Be concise. Use code blocks for command output. Respond in plain text unless the user asks for a specific format.`,
		cfg.Definition.Name, cfg.Definition.Description)
}

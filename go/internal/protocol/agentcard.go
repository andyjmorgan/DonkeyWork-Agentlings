package protocol

import (
	"fmt"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/andyjmorgan/agentlings-go/internal/config"
)

// GenerateAgentCard builds an A2A Agent Card from the application configuration.
// The cfg supplies the agent's identity (name, description), network binding
// (host, port), and optional skills list. When cfg.AgentExternalURL is set it
// is used as the card's endpoint base; otherwise the endpoint is derived from
// cfg.AgentHost and cfg.AgentPort. If cfg.Definition.Skills is non-empty each
// skill is mapped directly to an AgentSkill; otherwise a single default skill
// is synthesised from the agent's name and description. The returned card
// advertises text/plain I/O, no streaming, no push notifications, and requires
// an X-API-Key header for authentication.
func GenerateAgentCard(cfg *config.Config) *a2a.AgentCard {
	var url string
	if cfg.AgentExternalURL != "" {
		url = cfg.AgentExternalURL + "/a2a"
	} else {
		url = fmt.Sprintf("http://%s:%d/a2a", cfg.AgentHost, cfg.AgentPort)
	}

	var skills []a2a.AgentSkill
	if len(cfg.Definition.Skills) > 0 {
		for _, s := range cfg.Definition.Skills {
			skills = append(skills, a2a.AgentSkill{
				ID:          s.ID,
				Name:        s.Name,
				Description: s.Description,
				Tags:        s.Tags,
			})
		}
	} else {
		skills = []a2a.AgentSkill{
			{
				ID:          cfg.Definition.Name,
				Name:        cfg.Definition.Name,
				Description: cfg.Definition.Description,
			},
		}
	}

	card := &a2a.AgentCard{
		Name:        cfg.Definition.Name,
		Description: cfg.Definition.Description,
		Version:     "0.1.0",
		Skills:      skills,
		SupportedInterfaces: []*a2a.AgentInterface{
			a2a.NewAgentInterface(url, a2a.TransportProtocolJSONRPC),
		},
		DefaultInputModes:  []string{"text/plain"},
		DefaultOutputModes: []string{"text/plain"},
		Capabilities: a2a.AgentCapabilities{
			Streaming:         false,
			PushNotifications: false,
		},
		SecuritySchemes: a2a.NamedSecuritySchemes{
			"apiKey": a2a.APIKeySecurityScheme{
				Location: a2a.APIKeySecuritySchemeLocationHeader,
				Name:     "X-API-Key",
			},
		},
		SecurityRequirements: a2a.SecurityRequirementsOptions{
			{"apiKey": {}},
		},
	}

	return card
}

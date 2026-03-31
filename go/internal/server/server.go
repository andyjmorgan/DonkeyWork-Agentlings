package server

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/andyjmorgan/agentlings-go/internal/config"
	applog "github.com/andyjmorgan/agentlings-go/internal/logging"
	"github.com/andyjmorgan/agentlings-go/internal/llm"
	"github.com/andyjmorgan/agentlings-go/internal/loop"
	"github.com/andyjmorgan/agentlings-go/internal/protocol"
	"github.com/andyjmorgan/agentlings-go/internal/store"
	"github.com/andyjmorgan/agentlings-go/internal/tools"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func New(cfg *config.Config) (http.Handler, error) {
	applog.Setup(cfg.AgentLogLevel)

	journalStore := store.NewJournalStore(cfg.AgentDataDir)

	toolRegistry := tools.NewToolRegistry()
	toolRegistry.RegisterTools(cfg.Definition.Tools)

	if len(toolRegistry.ToolNames()) == 0 {
		slog.Warn("no tools enabled — add tools to agent.yaml or set AGENT_CONFIG (run 'agentling --list-tools' to see available options)")
	}

	llmClient, err := llm.NewClient(
		cfg.AgentLLMBackend,
		cfg.AnthropicAPIKey,
		cfg.AgentModel,
		cfg.AgentMaxTokens,
		toolRegistry.ToolNames(),
	)
	if err != nil {
		return nil, fmt.Errorf("create LLM client: %w", err)
	}

	messageLoop := loop.New(cfg, journalStore, llmClient, toolRegistry)
	agentCard := protocol.GenerateAgentCard(cfg)

	executor := protocol.NewAgentlingExecutor(messageLoop)
	a2aHandler := a2asrv.NewHandler(executor)
	jsonrpcHandler := a2asrv.NewJSONRPCHandler(a2aHandler)

	mcpServer := protocol.CreateMCPServer(messageLoop, agentCard)
	mcpHandler := mcp.NewStreamableHTTPHandler(
		func(r *http.Request) *mcp.Server { return mcpServer },
		nil,
	)

	agentCardJSON, err := json.Marshal(agentCard)
	if err != nil {
		return nil, fmt.Errorf("marshal agent card: %w", err)
	}

	mux := http.NewServeMux()

	mux.HandleFunc("/.well-known/agent-card.json", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(agentCardJSON)
	})
	mux.HandleFunc("/.well-known/agent.json", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(agentCardJSON)
	})

	mux.Handle("/a2a", jsonrpcHandler)
	mux.Handle("/mcp", mcpHandler)

	handler := apiKeyMiddleware(cfg.AgentAPIKey, mux)

	return handler, nil
}

func Run(cfg *config.Config) error {
	handler, err := New(cfg)
	if err != nil {
		return err
	}

	addr := fmt.Sprintf("%s:%d", cfg.AgentHost, cfg.AgentPort)
	slog.Info("agentling started", "name", cfg.Definition.Name, "addr", addr)
	return http.ListenAndServe(addr, handler)
}

var publicPaths = map[string]bool{
	"/.well-known/agent.json":      true,
	"/.well-known/agent-card.json": true,
}

func apiKeyMiddleware(apiKey string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if publicPaths[r.URL.Path] {
			next.ServeHTTP(w, r)
			return
		}

		key := r.Header.Get("X-API-Key")
		if !strings.EqualFold(key, apiKey) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			json.NewEncoder(w).Encode(map[string]string{"error": "Unauthorized"})
			return
		}

		next.ServeHTTP(w, r)
	})
}

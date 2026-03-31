package integration

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/andyjmorgan/agentlings-go/internal/config"
	"github.com/andyjmorgan/agentlings-go/internal/server"
)

var (
	testServer *httptest.Server
	apiKey     = "integration-test-key"
)

func TestMain(m *testing.M) {
	if url := os.Getenv("AGENT_URL"); url != "" {
		apiKey = os.Getenv("AGENT_API_KEY")
		os.Exit(m.Run())
	}

	tmp, err := os.MkdirTemp("", "agentling-test-*")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(tmp)

	yamlPath := filepath.Join(tmp, "agent.yaml")
	os.WriteFile(yamlPath, []byte("name: test-agent\ndescription: A test agent\ntools:\n  - bash\n  - filesystem\n"), 0o644)

	os.Setenv("AGENT_CONFIG", yamlPath)
	os.Setenv("AGENT_DATA_DIR", filepath.Join(tmp, "data"))
	os.Setenv("AGENT_API_KEY", apiKey)
	os.Setenv("AGENT_LLM_BACKEND", "mock")
	os.Setenv("AGENT_LOG_LEVEL", "WARNING")
	defer func() {
		os.Unsetenv("AGENT_CONFIG")
		os.Unsetenv("AGENT_DATA_DIR")
		os.Unsetenv("AGENT_API_KEY")
		os.Unsetenv("AGENT_LLM_BACKEND")
		os.Unsetenv("AGENT_LOG_LEVEL")
	}()

	cfg, err := config.Load()
	if err != nil {
		panic(fmt.Sprintf("load config: %v", err))
	}

	handler, err := server.New(cfg)
	if err != nil {
		panic(fmt.Sprintf("create server: %v", err))
	}

	testServer = httptest.NewServer(handler)
	defer testServer.Close()

	os.Exit(m.Run())
}

func serverURL() string {
	if url := os.Getenv("AGENT_URL"); url != "" {
		return url
	}
	return testServer.URL
}

// --- Agent Card Tests ---

func TestAgentCardReturns200NoAuth(t *testing.T) {
	resp, err := http.Get(serverURL() + "/.well-known/agent-card.json")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
}

func TestAgentCardValidJSON(t *testing.T) {
	resp, err := http.Get(serverURL() + "/.well-known/agent-card.json")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()

	var card map[string]any
	json.NewDecoder(resp.Body).Decode(&card)

	if card["name"] != "test-agent" {
		t.Errorf("name = %v, want test-agent", card["name"])
	}
	if card["description"] != "A test agent" {
		t.Errorf("description = %v, want A test agent", card["description"])
	}
}

func TestAgentCardLegacyPath(t *testing.T) {
	resp, err := http.Get(serverURL() + "/.well-known/agent.json")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
}

// --- A2A Tests ---

func a2aSend(t *testing.T, text string, contextID string) map[string]any {
	t.Helper()
	msg := map[string]any{
		"role":      "user",
		"parts":     []any{map[string]any{"text": text}},
		"messageId": "test-msg-1",
	}
	if contextID != "" {
		msg["contextId"] = contextID
	}

	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      "1",
		"method":  "SendMessage",
		"params":  map[string]any{"message": msg},
	})

	req, _ := http.NewRequest("POST", serverURL()+"/a2a", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", apiKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST /a2a: %v", err)
	}
	defer resp.Body.Close()

	var result map[string]any
	json.NewDecoder(resp.Body).Decode(&result)
	return result
}

func TestA2ASendReturnsResponse(t *testing.T) {
	result := a2aSend(t, "hello", "")
	if result["error"] != nil {
		t.Fatalf("unexpected error: %v", result["error"])
	}
	if result["result"] == nil {
		t.Fatal("expected result in response")
	}
}

func TestA2ANoAuthReturns401(t *testing.T) {
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      "1",
		"method":  "message/send",
		"params": map[string]any{
			"message": map[string]any{
				"role":  "user",
				"parts": []any{map[string]any{"text": "hello"}},
			},
		},
	})

	req, _ := http.NewRequest("POST", serverURL()+"/a2a", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 401 {
		t.Errorf("status = %d, want 401", resp.StatusCode)
	}
}

func TestA2AWrongKeyReturns401(t *testing.T) {
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      "1",
		"method":  "message/send",
		"params": map[string]any{
			"message": map[string]any{
				"role":  "user",
				"parts": []any{map[string]any{"text": "hello"}},
			},
		},
	})

	req, _ := http.NewRequest("POST", serverURL()+"/a2a", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", "wrong-key")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 401 {
		t.Errorf("status = %d, want 401", resp.StatusCode)
	}
}

// --- MCP Tests ---

type mcpSession struct {
	t         *testing.T
	sessionID string
}

func newMCPSession(t *testing.T) *mcpSession {
	t.Helper()
	s := &mcpSession{t: t}
	s.initialize()
	return s
}

func (s *mcpSession) request(method string, params any) map[string]any {
	s.t.Helper()
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  method,
		"params":  params,
	})

	req, _ := http.NewRequest("POST", serverURL()+"/mcp", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set("X-API-Key", apiKey)
	if s.sessionID != "" {
		req.Header.Set("Mcp-Session-Id", s.sessionID)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		s.t.Fatalf("POST /mcp: %v", err)
	}
	defer resp.Body.Close()

	if sid := resp.Header.Get("Mcp-Session-Id"); sid != "" {
		s.sessionID = sid
	}

	data, _ := io.ReadAll(resp.Body)

	ct := resp.Header.Get("Content-Type")
	if strings.HasPrefix(ct, "text/event-stream") {
		return parseSseResponse(s.t, data)
	}

	var result map[string]any
	json.Unmarshal(data, &result)
	return result
}

func (s *mcpSession) notify(method string) {
	s.t.Helper()
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"method":  method,
	})

	req, _ := http.NewRequest("POST", serverURL()+"/mcp", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set("X-API-Key", apiKey)
	if s.sessionID != "" {
		req.Header.Set("Mcp-Session-Id", s.sessionID)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		s.t.Fatalf("POST /mcp notification: %v", err)
	}
	resp.Body.Close()
}

func parseSseResponse(t *testing.T, data []byte) map[string]any {
	t.Helper()
	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			jsonData := strings.TrimPrefix(line, "data: ")
			var result map[string]any
			if err := json.Unmarshal([]byte(jsonData), &result); err == nil {
				return result
			}
		}
	}
	return map[string]any{}
}

func (s *mcpSession) initialize() {
	s.t.Helper()
	result := s.request("initialize", map[string]any{
		"protocolVersion": "2025-03-26",
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]any{"name": "test-client", "version": "1.0"},
	})
	if result["error"] != nil {
		s.t.Fatalf("initialize error: %v", result["error"])
	}

	s.notify("notifications/initialized")
}

func TestMCPInitialize(t *testing.T) {
	newMCPSession(t)
}

func TestMCPListTools(t *testing.T) {
	s := newMCPSession(t)
	result := s.request("tools/list", map[string]any{})

	if result["error"] != nil {
		t.Fatalf("tools/list error: %v", result["error"])
	}

	res, _ := result["result"].(map[string]any)
	if res == nil {
		t.Fatalf("expected result, got: %v", result)
	}
	toolsList, _ := res["tools"].([]any)
	if len(toolsList) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(toolsList))
	}

	tool, _ := toolsList[0].(map[string]any)
	if tool["name"] != "test-agent" {
		t.Errorf("tool name = %v, want test-agent", tool["name"])
	}
}

func TestMCPCallTool(t *testing.T) {
	s := newMCPSession(t)
	result := s.request("tools/call", map[string]any{
		"name":      "test-agent",
		"arguments": map[string]any{"message": "hello"},
	})

	if result["error"] != nil {
		t.Fatalf("tools/call error: %v", result["error"])
	}

	res, _ := result["result"].(map[string]any)
	if res == nil {
		t.Fatal("expected result")
	}
	content, _ := res["content"].([]any)
	if len(content) == 0 {
		t.Fatal("expected content")
	}
}

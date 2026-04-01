// Package main is the entry point for the agentling server.
package main

import (
	"fmt"
	"log"
	"os"
	"sort"

	"github.com/andyjmorgan/agentlings-go/internal/config"
	"github.com/andyjmorgan/agentlings-go/internal/server"
	"github.com/andyjmorgan/agentlings-go/internal/tools"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "--list-tools" {
		listTools()
		return
	}

	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}

	if err := server.Run(cfg); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

func listTools() {
	fmt.Println("Available tools:")
	fmt.Println()

	for group, members := range tools.ToolGroups {
		sort.Strings(members)
		fmt.Printf("  %s (group):\n", group)
		for _, name := range members {
			if defn, ok := tools.BuiltinRegistry[name]; ok {
				fmt.Printf("    - %s: %s\n", name, defn.Description)
			}
		}
		fmt.Println()
	}
}

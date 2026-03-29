"""CLI entry point for the agentling server."""

import sys


def main() -> None:
    """Start the agentling HTTP server, or show tool info with ``--list-tools``."""
    if "--list-tools" in sys.argv:
        _list_tools()
        return

    from agentlings.server import run

    run()


def _list_tools() -> None:
    from agentlings.tools.builtins import BUILTIN_REGISTRY
    from agentlings.tools.registry import TOOL_GROUPS

    print("Available tool groups:\n")
    for group, tools in TOOL_GROUPS.items():
        print(f"  {group}")
        for tool in tools:
            desc = BUILTIN_REGISTRY[tool]["description"]
            print(f"    {tool:20s} {desc}")
        print()

    standalone = set(BUILTIN_REGISTRY.keys())
    for tools in TOOL_GROUPS.values():
        standalone -= set(tools)
    if standalone:
        print("Standalone tools:\n")
        for name in sorted(standalone):
            desc = BUILTIN_REGISTRY[name]["description"]
            print(f"  {name:22s} {desc}")
        print()

    print("Enable via AGENT_TOOLS env var (comma-separated).")
    print("Examples:")
    print("  AGENT_TOOLS=bash,filesystem")
    print("  AGENT_TOOLS=bash,read_file")


if __name__ == "__main__":
    main()

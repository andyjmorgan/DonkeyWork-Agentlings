"""CLI entry point for the agentling server."""

import asyncio
import sys


def main() -> None:
    """Route CLI commands: server, tool listing, memory, or sleep."""
    args = sys.argv[1:]

    if "--list-tools" in args:
        _list_tools()
    elif args and args[0] == "memory":
        _memory_command(args[1:])
    elif args and args[0] == "sleep":
        _sleep_command(args[1:])
    else:
        from agentlings.server import run
        run()


def _list_tools() -> None:
    from agentlings.tools.builtins import BUILTIN_REGISTRY
    from agentlings.tools.memory import MEMORY_TOOL_DEFINITION
    from agentlings.tools.registry import TOOL_GROUPS

    print("Available tool groups:\n")
    for group, tools in TOOL_GROUPS.items():
        print(f"  {group}")
        all_tools = {**BUILTIN_REGISTRY, MEMORY_TOOL_DEFINITION["name"]: MEMORY_TOOL_DEFINITION}
        for tool in tools:
            desc = all_tools.get(tool, {}).get("description", "")
            print(f"    {tool:20s} {desc}")
        print()

    standalone = set(all_tools.keys())
    for tools in TOOL_GROUPS.values():
        standalone -= set(tools)
    if standalone:
        print("Standalone tools:\n")
        for name in sorted(standalone):
            desc = all_tools.get(name, {}).get("description", "")
            print(f"  {name:22s} {desc}")
        print()

    print("Enable via agent YAML 'tools' list or AGENT_TOOLS env var.")
    print("Examples:")
    print("  tools: [bash, filesystem, memory]")


def _memory_command(args: list[str]) -> None:
    """Handle 'agentling memory show'."""
    if not args or args[0] != "show":
        print("Usage: agentling memory show")
        sys.exit(1)

    from agentlings.config import AgentConfig
    from agentlings.core.memory_store import MemoryFileStore

    config = AgentConfig()
    store = MemoryFileStore(config.agent_data_dir)
    memory = store.load()

    if not memory.entries:
        print("Memory is empty.")
        return

    for entry in memory.entries:
        print(f"  {entry.key}: {entry.value}")
        print(f"    recorded: {entry.recorded.isoformat()}")


def _sleep_command(args: list[str]) -> None:
    """Handle 'agentling sleep --date YYYY-MM-DD'."""
    from datetime import datetime, timezone

    date = None
    for i, arg in enumerate(args):
        if arg == "--date" and i + 1 < len(args):
            try:
                date = datetime.strptime(args[i + 1], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                print(f"Invalid date format: {args[i + 1]} (expected YYYY-MM-DD)")
                sys.exit(1)

    from agentlings.config import AgentConfig
    from agentlings.core.llm import create_llm_client
    from agentlings.core.memory_store import MemoryFileStore
    from agentlings.core.sleep import SleepCycle
    from agentlings.core.store import JournalStore
    from agentlings.log import setup_logging

    config = AgentConfig()
    setup_logging(config.agent_log_level)

    memory_store = MemoryFileStore(config.agent_data_dir)
    journal_store = JournalStore(config.agent_data_dir)
    llm = create_llm_client(
        backend=config.agent_llm_backend,
        api_key=config.anthropic_api_key,
        model=config.agent_model,
        max_tokens=config.agent_max_tokens,
    )

    cycle = SleepCycle(
        config=config, llm=llm, memory_store=memory_store, store=journal_store,
    )
    asyncio.run(cycle.run(date=date))


if __name__ == "__main__":
    main()

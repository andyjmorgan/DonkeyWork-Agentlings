"""CLI entry point: subcommand router for ``agentling``.

Subcommands:

- ``run`` (default)  — start the agent server using ``agent.yaml``/``.env`` from CWD or ``--dir``
- ``init <name>``    — scaffold a new agent directory
- ``upgrade``        — apply pending data migrations, advance the framework-version stamp
- ``memory show``    — print the current memory store
- ``sleep [--date]`` — run a sleep cycle for the agent in CWD
- ``--list-tools``   — print bundled tool registry

``--dir <PATH>`` works on ``run`` and ``upgrade`` to operate on a directory
other than CWD.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


def main() -> None:
    """Entry-point dispatcher for the ``agentling`` console script."""
    parser = _build_parser()
    args = parser.parse_args()

    if getattr(args, "legacy_list_tools", False):
        _list_tools()
        return
    if args.command in (None, "run"):
        _run(getattr(args, "dir", None))
    elif args.command == "init":
        _init(args)
    elif args.command == "upgrade":
        _upgrade(args)
    elif args.command == "memory":
        _memory_command(args.subcommand)
    elif args.command == "sleep":
        _sleep_command(args.date)
    elif args.command == "list-tools":
        _list_tools()
    else:
        parser.print_help()
        sys.exit(2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentling", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="start the agent server (default if no subcommand)")
    run_p.add_argument("--dir", type=Path, default=None, help="agent directory (defaults to CWD)")

    init_p = sub.add_parser("init", help="scaffold a new agent directory")
    init_p.add_argument("name", help="agent name (also default directory name)")
    init_p.add_argument("--dir", type=Path, default=None, help="output directory (default: ./<name>)")
    init_p.add_argument("--template", default="default", help="bundled template to scaffold from")
    init_p.add_argument("--api-key", default=None, help="AGENT_API_KEY value (auto-generated if omitted)")
    init_p.add_argument("--anthropic-api-key", default=None, help="pre-populate ANTHROPIC_API_KEY")
    init_p.add_argument("--anthropic-base-url", default=None, help="pre-populate ANTHROPIC_BASE_URL")
    init_p.add_argument("--force", action="store_true", help="overwrite an existing non-empty directory")

    upg_p = sub.add_parser("upgrade", help="apply data migrations against the installed framework")
    upg_p.add_argument("--dir", type=Path, default=None, help="agent directory (defaults to CWD)")
    upg_p.add_argument("--dry-run", action="store_true", help="report pending migrations without running them")

    mem_p = sub.add_parser("memory", help="memory store inspection")
    mem_p.add_argument("subcommand", choices=["show"])

    slp_p = sub.add_parser("sleep", help="run a one-off sleep cycle")
    slp_p.add_argument("--date", default=None, help="YYYY-MM-DD (defaults to today)")

    sub.add_parser("list-tools", help="print bundled tool registry")

    # Back-compat: older docs and habits use `--list-tools` at top level.
    parser.add_argument("--list-tools", dest="legacy_list_tools", action="store_true", help=argparse.SUPPRESS)

    return parser


def _run(agent_dir: Path | None) -> None:
    """Run the server with optional directory pinning.

    Sets ``AGENT_CONFIG`` and ``AGENT_DATA_DIR`` to point at the directory's
    artefacts before constructing ``AgentConfig``. The ``.env`` file inside
    the directory is loaded by ``pydantic-settings`` automatically because
    we change CWD before AgentConfig is created.
    """
    if agent_dir is not None:
        target = agent_dir.resolve()
        os.environ.setdefault("AGENT_CONFIG", str(target / "agent.yaml"))
        os.environ.setdefault("AGENT_DATA_DIR", str(target / "data"))
        os.chdir(target)

    from agentlings.server import run

    run()


def _init(args: argparse.Namespace) -> None:
    from agentlings.cli.init import init_agent

    try:
        result = init_agent(
            args.name,
            dir=args.dir,
            template=args.template,
            api_key=args.api_key,
            anthropic_api_key=args.anthropic_api_key,
            anthropic_base_url=args.anthropic_base_url,
            force=args.force,
        )
    except (FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"created agent at {result.agent_dir}")
    print(f"  template:          {result.template}")
    print(f"  framework version: {result.framework_version}")
    print(f"  generated AGENT_API_KEY (rotate by editing .env)")
    print()
    print("next steps:")
    print(f"  cd {result.agent_dir}")
    print("  # add ANTHROPIC_API_KEY to .env if you need it")
    print("  agentling run")


def _upgrade(args: argparse.Namespace) -> None:
    from agentlings.cli.upgrade import upgrade_agent

    try:
        result = upgrade_agent(args.dir, dry_run=args.dry_run)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"recorded version:  {result.recorded_version or '(none)'}")
    print(f"installed version: {result.installed_version}")
    if not result.applied:
        print("no migrations to apply.")
        return
    label = "would apply" if args.dry_run else "applied"
    for m in result.applied:
        print(f"  {label} {m.id}: {m.description}")


def _list_tools() -> None:
    from agentlings.tools.builtins import BUILTIN_REGISTRY
    from agentlings.tools.memory import MEMORY_TOOL_DEFINITION
    from agentlings.tools.registry import TOOL_GROUPS

    print("Available tool groups:\n")
    all_tools = {**BUILTIN_REGISTRY, MEMORY_TOOL_DEFINITION["name"]: MEMORY_TOOL_DEFINITION}
    for group, tools in TOOL_GROUPS.items():
        print(f"  {group}")
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


def _memory_command(subcommand: str) -> None:
    if subcommand != "show":
        print("Usage: agentling memory show", file=sys.stderr)
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


def _sleep_command(date_str: str | None) -> None:
    from datetime import datetime, timezone

    date = None
    if date_str:
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"Invalid date format: {date_str} (expected YYYY-MM-DD)", file=sys.stderr)
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
        base_url=config.anthropic_base_url,
    )

    cycle = SleepCycle(
        config=config, llm=llm, memory_store=memory_store, store=journal_store,
    )
    asyncio.run(cycle.run(date=date))


if __name__ == "__main__":
    main()

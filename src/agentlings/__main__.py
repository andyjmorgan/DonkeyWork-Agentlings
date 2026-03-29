"""CLI entry point for the agentling server."""


def main() -> None:
    """Start the agentling HTTP server."""
    from agentlings.server import run

    run()


if __name__ == "__main__":
    main()

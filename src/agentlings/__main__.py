"""Entry point for running agentlings as ``python -m agentlings``."""


def main() -> None:
    """Start the agentling server."""
    from agentlings.server import run

    run()


if __name__ == "__main__":
    main()

"""Reference tools demonstrating the ``@tool`` decorator surface.

These are intentionally small and illustrative — they exist to show
authors how to express common patterns:

- ``echo``    — simplest possible tool, no I/O.
- ``http_get``— async I/O + ``Annotated[..., Field(...)]`` per-param descriptions.
- ``set_severity`` — string ``Enum`` parameter.
- ``geocode`` — env-var-driven config + nested ``BaseModel`` parameter.

They are not registered by default; agent configs opt in by name once the
loader supports plugin discovery.
"""

from agentlings.tools.examples.echo import echo
from agentlings.tools.examples.geocode import geocode
from agentlings.tools.examples.http_get import http_get
from agentlings.tools.examples.set_severity import set_severity

EXAMPLE_TOOLS = [echo, http_get, set_severity, geocode]

__all__ = ["EXAMPLE_TOOLS", "echo", "geocode", "http_get", "set_severity"]

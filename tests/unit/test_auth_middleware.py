from __future__ import annotations

import datetime
from types import SimpleNamespace
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from agentlings import server
from agentlings.config import OAuthConfig
from agentlings.server import AuthMiddleware

ISSUER = "https://auth.example.com/realms/Agents"
AUDIENCE = "donkeywork-agents-api"
RESOURCE_METADATA_URL = "https://agent.example.com/.well-known/oauth-protected-resource"


@pytest.fixture(scope="module")
def keypair() -> tuple[Any, Any]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def _mint(private_key: Any, **claims: Any) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + datetime.timedelta(minutes=5),
        **claims,
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


@pytest.fixture
def app_factory(keypair: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch):
    """Build a Starlette app guarded by AuthMiddleware with a stubbed JWKS.

    The fake JWKS client returns the test public key, so the real signature
    verification path in ``jwt.decode`` runs without any network access.
    """
    _, public_key = keypair

    class _FakeJWKClient:
        def __init__(self, uri: str) -> None:
            self._uri = uri

        def get_signing_key_from_jwt(self, token: str) -> Any:
            return SimpleNamespace(key=public_key)

    monkeypatch.setattr(server, "PyJWKClient", _FakeJWKClient)

    def _build(oauth: OAuthConfig | None) -> TestClient:
        async def protected(request: Request) -> PlainTextResponse:
            return PlainTextResponse("ok")

        async def public_doc(request: Request) -> PlainTextResponse:
            return PlainTextResponse("public")

        routes = [
            Route("/mcp", protected, methods=["GET", "POST"]),
            Route(
                "/.well-known/oauth-protected-resource",
                public_doc,
                methods=["GET"],
            ),
        ]
        middleware = [
            Middleware(
                AuthMiddleware,
                api_key="secret-key",
                oauth=oauth,
                resource_metadata_url=RESOURCE_METADATA_URL if oauth else None,
            )
        ]
        app = Starlette(routes=routes, middleware=middleware)
        return TestClient(app)

    return _build


@pytest.fixture
def oauth() -> OAuthConfig:
    return OAuthConfig(
        enabled=True,
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_uri="https://auth.example.com/jwks",
    )


def test_valid_api_key_passes(app_factory, oauth: OAuthConfig) -> None:
    client = app_factory(oauth)
    resp = client.get("/mcp", headers={"X-API-Key": "secret-key"})
    assert resp.status_code == 200


def test_no_credentials_rejected_with_challenge(app_factory, oauth: OAuthConfig) -> None:
    client = app_factory(oauth)
    resp = client.get("/mcp")
    assert resp.status_code == 401
    assert RESOURCE_METADATA_URL in resp.headers.get("www-authenticate", "")


def test_valid_bearer_passes(app_factory, oauth: OAuthConfig, keypair) -> None:
    private_key, _ = keypair
    client = app_factory(oauth)
    token = _mint(private_key)
    resp = client.get("/mcp", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200


def test_valid_bearer_with_array_audience_passes(
    app_factory, oauth: OAuthConfig, keypair
) -> None:
    """Keycloak issues ``aud`` as an array; the configured audience must be
    accepted when it is a member of that array (the live notes-api shape)."""
    private_key, _ = keypair
    client = app_factory(oauth)
    token = _mint(private_key, aud=[AUDIENCE, "account"])
    resp = client.get("/mcp", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200


def test_expired_token_rejected(app_factory, oauth: OAuthConfig, keypair) -> None:
    private_key, _ = keypair
    client = app_factory(oauth)
    past = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
    token = _mint(private_key, exp=past, iat=past)
    resp = client.get("/mcp", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_wrong_audience_rejected(app_factory, oauth: OAuthConfig, keypair) -> None:
    private_key, _ = keypair
    client = app_factory(oauth)
    token = _mint(private_key, aud="some-other-resource")
    resp = client.get("/mcp", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_wrong_issuer_rejected(app_factory, oauth: OAuthConfig, keypair) -> None:
    private_key, _ = keypair
    client = app_factory(oauth)
    token = _mint(private_key, iss="https://evil.example.com/realms/Agents")
    resp = client.get("/mcp", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_bad_signature_rejected(app_factory, oauth: OAuthConfig) -> None:
    """A token signed by a different key fails verification against the JWKS."""
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    client = app_factory(oauth)
    token = _mint(other_key)
    resp = client.get("/mcp", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_malformed_authorization_header_rejected(
    app_factory, oauth: OAuthConfig
) -> None:
    client = app_factory(oauth)
    resp = client.get("/mcp", headers={"Authorization": "Basic abc123"})
    assert resp.status_code == 401


def test_public_path_bypasses_auth(app_factory, oauth: OAuthConfig) -> None:
    client = app_factory(oauth)
    resp = client.get("/.well-known/oauth-protected-resource")
    assert resp.status_code == 200
    assert resp.text == "public"


def test_bearer_ignored_when_oauth_disabled(app_factory, keypair) -> None:
    """With OAuth off, a bearer token is not honoured — only the API key is."""
    private_key, _ = keypair
    client = app_factory(None)
    token = _mint(private_key)
    resp = client.get("/mcp", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    resp = client.get("/mcp", headers={"X-API-Key": "secret-key"})
    assert resp.status_code == 200


def test_resolve_jwks_uri_uses_explicit_value() -> None:
    oauth = OAuthConfig(
        enabled=True, issuer=ISSUER, audience=AUDIENCE, jwks_uri="https://x/jwks"
    )
    assert server._resolve_jwks_uri(oauth) == "https://x/jwks"


def test_resolve_jwks_uri_derives_from_oidc_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When jwks_uri is unset, it is read from the issuer's OIDC discovery doc."""
    derived = ISSUER + "/protocol/openid-connect/certs"
    captured: dict[str, str] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {"jwks_uri": derived}

    class _FakeClient:
        def __init__(self, timeout: float) -> None:
            pass

        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def get(self, url: str) -> _FakeResponse:
            captured["url"] = url
            return _FakeResponse()

    monkeypatch.setattr(server.httpx, "Client", _FakeClient)
    oauth = OAuthConfig(enabled=True, issuer=ISSUER, audience=AUDIENCE, jwks_uri=None)
    assert server._resolve_jwks_uri(oauth) == derived
    assert captured["url"] == ISSUER + "/.well-known/openid-configuration"


def test_protected_resource_metadata_shape() -> None:
    from agentlings.config import AgentConfig

    config = AgentConfig(
        anthropic_api_key="x",
        agent_api_key="k",
        agent_external_url="https://agent.example.com",
        _env_file=None,
    )
    oauth = OAuthConfig(enabled=True, issuer=ISSUER, audience=AUDIENCE)
    doc = server._protected_resource_metadata(config, oauth)
    assert doc["resource"] == "https://agent.example.com/mcp"
    assert doc["authorization_servers"] == [ISSUER]
    assert doc["bearer_methods_supported"] == ["header"]

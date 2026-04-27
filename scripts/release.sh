#!/usr/bin/env bash
# Build and upload the current pyproject version to PyPI.
#
# Usage:
#   PYPI_TOKEN=pypi-... ./scripts/release.sh         # real PyPI
#   PYPI_TOKEN=pypi-... TARGET=test ./scripts/release.sh   # TestPyPI dry-run
#
# The token must be a freshly-generated PyPI API token. Generate one at:
#   https://pypi.org/manage/account/token/
#
# Never commit the token, never paste it anywhere, never put it in your
# shell history (use `read -rs` or set it in a one-shot env). After this
# script succeeds, replace the account-wide token with a project-scoped
# one and revoke the original.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Pull PYPI_TOKEN from .env if not already in the environment. The .env file
# is gitignored; never commit it.
if [[ -z "${PYPI_TOKEN:-}" && -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if [[ -z "${PYPI_TOKEN:-}" ]]; then
    echo "error: PYPI_TOKEN not set (looked in env and .env)" >&2
    echo "" >&2
    echo "Either:" >&2
    echo "  - add 'PYPI_TOKEN=pypi-...' to .env (gitignored), or" >&2
    echo "  - export PYPI_TOKEN inline before running this script" >&2
    echo "" >&2
    echo "Generate tokens at https://pypi.org/manage/account/token/" >&2
    exit 1
fi

TARGET="${TARGET:-pypi}"
case "$TARGET" in
    pypi)        REPO_FLAG=() ;;
    test|testpypi) REPO_FLAG=(--repository-url https://test.pypi.org/legacy/) ;;
    *) echo "error: unknown TARGET '$TARGET' (expected 'pypi' or 'test')" >&2; exit 1 ;;
esac

VERSION="$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')"
echo "==> releasing agentlings $VERSION to $TARGET"

echo "==> ensuring build/twine are installed"
python3 -m pip install --quiet --upgrade build twine

echo "==> cleaning previous artefacts"
rm -rf dist/ build/ src/agentlings.egg-info/

echo "==> building wheel + sdist"
python3 -m build

echo "==> validating artefacts"
python3 -m twine check dist/*

EXPECTED_WHEEL="dist/agentlings-${VERSION}-py3-none-any.whl"
if [[ ! -f "$EXPECTED_WHEEL" ]]; then
    echo "error: expected $EXPECTED_WHEEL not produced" >&2
    exit 1
fi

echo "==> uploading to $TARGET"
TWINE_USERNAME=__token__ \
TWINE_PASSWORD="$PYPI_TOKEN" \
    python3 -m twine upload ${REPO_FLAG[@]+"${REPO_FLAG[@]}"} dist/*

echo ""
echo "==> done."
case "$TARGET" in
    pypi) echo "    https://pypi.org/project/agentlings/$VERSION/" ;;
    test|testpypi) echo "    https://test.pypi.org/project/agentlings/$VERSION/" ;;
esac

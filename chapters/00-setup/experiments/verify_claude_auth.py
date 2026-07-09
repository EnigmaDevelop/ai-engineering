"""Proves Claude Code authenticates via the Pro subscription OAuth token
(CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`), not a pay-per-token API key.

Run: uv run python chapters/00-setup/experiments/verify_claude_auth.py
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"
EXPECTED = "TOKEN_AUTH_OK"


def load_oauth_token() -> str:
    if not ENV_FILE.exists():
        sys.exit(f".env not found at {ENV_FILE} — run `claude setup-token` and save it there first")
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("CLAUDE_CODE_OAUTH_TOKEN="):
            return line.split("=", 1)[1].strip()
    sys.exit("CLAUDE_CODE_OAUTH_TOKEN not found in .env")


def main() -> None:
    token = load_oauth_token()

    # Isolated environment: only the OAuth token plus what the OS needs to find the
    # `claude` binary and its config. No ANTHROPIC_API_KEY, no ANTHROPIC_AUTH_TOKEN —
    # proves the call authenticates via the subscription token alone, not API billing.
    env = {
        "CLAUDE_CODE_OAUTH_TOKEN": token,
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "USERPROFILE": os.environ.get("USERPROFILE", ""),
        "APPDATA": os.environ.get("APPDATA", ""),
        "LOCALAPPDATA": os.environ.get("LOCALAPPDATA", ""),
        "COMSPEC": os.environ.get("COMSPEC", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
    }

    # On Windows, "claude" is an npm .cmd shim, not a PE executable — CreateProcess
    # (what subprocess uses under shell=False) can't launch it directly, so run it
    # through cmd.exe via shell=True. No untrusted input goes into this string.
    result = subprocess.run(
        f'claude -p "Reply with exactly: {EXPECTED}"',
        shell=True,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    output = result.stdout.strip()
    print(f"exit_code={result.returncode} output={output!r}")

    if result.returncode != 0 or EXPECTED not in output:
        print("STDERR:", result.stderr, file=sys.stderr)
        sys.exit("FAILED: subscription OAuth token did not authenticate")

    print("PASSED: Claude Code authenticated via CLAUDE_CODE_OAUTH_TOKEN (subscription, zero extra spend)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Scan VCR cassettes for potential leaked secrets using Claude.

This script reads VCR cassette files and asks Claude to identify any content
that looks like it could be a secret (API keys, tokens, credentials, etc.).

Usage:
    uv run python scripts/check_vcr_secrets.py [--verbose]

Environment:
    ANTHROPIC_API_KEY: Required for Claude API access
"""

import argparse
import sys
from pathlib import Path

from chatlas import ChatAnthropic

VCR_DIR = Path(__file__).parent.parent / "tests" / "_vcr"

SYSTEM_PROMPT = """\
You are a security auditor scanning HTTP recording files (VCR cassettes) for leaked secrets.

Your task is to identify content that could be a REAL secret, including:
- API keys (any format: sk-*, key-*, AIza*, AKIA*, etc.)
- Bearer tokens or JWT tokens
- Passwords or credentials
- Private keys or certificates
- Connection strings with embedded credentials

IGNORE these (they are NOT secrets):
- Cloudflare cookies (__cf_bm, _cfuvid) - these are short-lived CDN cookies, not auth secrets
- Dummy/placeholder values like "dummy-api-key", "test-key", "SCRUBBED_*"
- Empty arrays for filtered headers like "authorization: []"
- Public identifiers (model names, request IDs, response IDs, etc.)
- Base64-encoded request/response bodies that are clearly content (images, PDFs)
- MCP session IDs (mcp-session-id) - these are local test session identifiers

Only flag items that could cause real security issues if exposed publicly.

For each potential secret found, report:
1. The file path
2. The suspicious content (truncated if very long)
3. Why you think it might be a secret
"""

USER_PROMPT_TEMPLATE = """\
Please scan the following VCR cassette files for potential leaked secrets.

If you find ANY potential secrets, respond with:
SECRETS_FOUND: YES
Then list each finding.

If the files appear clean, respond with:
SECRETS_FOUND: NO

Files to scan:

{file_contents}
"""


def read_cassettes(vcr_dir: Path, verbose: bool = False) -> dict[str, str]:
    """Read all VCR cassette files."""
    cassettes = {}

    if not vcr_dir.exists():
        print(f"Warning: VCR directory not found: {vcr_dir}")
        return cassettes

    for yaml_file in vcr_dir.rglob("*.yaml"):
        rel_path = yaml_file.relative_to(vcr_dir)
        if verbose:
            print(f"Reading: {rel_path}")
        cassettes[str(rel_path)] = yaml_file.read_text()

    return cassettes


def chunk_cassettes(cassettes: dict[str, str], max_chars: int = 50000) -> list[str]:
    """
    Split cassettes into chunks that fit within context limits.

    Returns formatted strings ready for the prompt.
    """
    chunks = []
    current_chunk = []
    current_size = 0

    for path, content in cassettes.items():
        entry = f"=== FILE: {path} ===\n{content}\n"
        entry_size = len(entry)

        if current_size + entry_size > max_chars and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(entry)
        current_size += entry_size

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def scan_with_claude(chunks: list[str], verbose: bool = False) -> tuple[bool, list[str]]:
    """
    Scan cassette chunks using Claude.

    Returns (secrets_found, findings).
    """
    chat = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        system_prompt=SYSTEM_PROMPT,
    )

    all_findings = []
    secrets_found = False

    for i, chunk in enumerate(chunks, 1):
        if verbose:
            print(f"Scanning chunk {i}/{len(chunks)}...")

        prompt = USER_PROMPT_TEMPLATE.format(file_contents=chunk)
        response = str(chat.chat(prompt))

        if "SECRETS_FOUND: YES" in response:
            secrets_found = True
            all_findings.append(f"--- Chunk {i} findings ---\n{response}")
        elif verbose:
            print(f"  Chunk {i}: Clean")

        # Reset chat for next chunk (independent analysis)
        chat = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            system_prompt=SYSTEM_PROMPT,
        )

    return secrets_found, all_findings


def main():
    parser = argparse.ArgumentParser(
        description="Scan VCR cassettes for leaked secrets using Claude"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--vcr-dir",
        type=Path,
        default=VCR_DIR,
        help=f"Path to VCR cassettes directory (default: {VCR_DIR})"
    )
    args = parser.parse_args()

    print("Scanning VCR cassettes for potential secrets...")

    # Read all cassettes
    cassettes = read_cassettes(args.vcr_dir, args.verbose)

    if not cassettes:
        print("No cassette files found.")
        return 0

    print(f"Found {len(cassettes)} cassette files")

    # Chunk for API limits
    chunks = chunk_cassettes(cassettes)
    print(f"Split into {len(chunks)} chunks for analysis")

    # Scan with Claude
    secrets_found, findings = scan_with_claude(chunks, args.verbose)

    if secrets_found:
        print("\n" + "=" * 60)
        print("POTENTIAL SECRETS FOUND!")
        print("=" * 60)
        for finding in findings:
            print(finding)
        print("=" * 60)
        print("\nPlease review the findings above and update make_vcr_config()")
        print("in tests/conftest.py to filter any new sensitive headers.")
        return 1
    else:
        print("\nNo secrets found in VCR cassettes.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

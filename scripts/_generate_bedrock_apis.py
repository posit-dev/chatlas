"""Derive the Bedrock model -> API lookup table from litellm's model metadata.

Bedrock reaches most models through the Converse API on the bedrock-runtime
endpoint, so chatlas only needs to know about the models Converse *can't*
serve. Those live on the bedrock-mantle endpoint and speak either the OpenAI
Responses API or the Anthropic Messages API.

Mirrors the derivation in ellmer's data-raw/prices.R so the two projects agree.
"""

import json
import re
import urllib.request
from pathlib import Path

LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/model_prices_and_context_window.json"

OUT_PATH = Path(__file__).parent.parent / "chatlas" / "data" / "bedrock_apis.json"

CROSS_REGION_PREFIX = re.compile(r"^(us|eu|apac|au|jp|ca|global)\.")
VERSION_SUFFIX = re.compile(r"-v?\d+(:\d+)?$")

# litellm records supported_endpoints only for the OpenAI-compatible APIs; it
# has no data on which models speak the Anthropic Messages API, so these are
# maintained by hand. AWS documents them as mantle-only.
MESSAGES_MODELS = {
    "anthropic.claude-mythos-5": "messages",
    "anthropic.claude-mythos-preview": "messages",
}


def main() -> None:
    with urllib.request.urlopen(LITELLM_URL) as resp:
        litellm = json.load(resp)

    rows = list(bedrock_rows(litellm))
    if not any(provider == "bedrock_mantle" for provider, _, _ in rows):
        raise SystemExit("Expected some bedrock_mantle models in litellm data")

    # Match models across endpoints by stripping the cross-region inference
    # prefix and the version suffix, which the two endpoints spell differently
    # (bedrock_converse/openai.gpt-oss-120b-1:0 vs bedrock_mantle/openai.gpt-oss-120b).
    on_runtime = {
        match_key(model) for provider, model, _ in rows if provider != "bedrock_mantle"
    }

    mantle_models = [model for provider, model, _ in rows if provider == "bedrock_mantle"]
    overlap = {model for model in mantle_models if match_key(model) in on_runtime}
    if not overlap:
        raise SystemExit(
            "Expected some bedrock_mantle models to also resolve to a Converse-side "
            "model (e.g. the gpt-oss family), but none matched. This means "
            "match_key()'s cross-region-prefix/version-suffix stripping no longer "
            "aligns bedrock_mantle ids with bedrock/bedrock_converse ids — check "
            "CROSS_REGION_PREFIX and VERSION_SUFFIX against the current litellm id "
            "formats before trusting this table, since a broken match would "
            "silently add Converse-servable models to it."
        )

    apis = dict(MESSAGES_MODELS)
    for provider, model, endpoints in rows:
        if provider != "bedrock_mantle":
            continue
        if match_key(model) in on_runtime:
            continue
        if "/v1/responses" not in endpoints:
            continue
        apis.setdefault(model, "responses")

    if len(apis) < 5:
        raise SystemExit(f"Expected at least 5 non-Converse models, got {len(apis)}")

    table = {model: apis[model] for model in sorted(apis)}
    OUT_PATH.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(table)} entries to {OUT_PATH}")


def bedrock_rows(litellm: dict) -> list[tuple[str, str, list[str]]]:
    rows = []
    for key, spec in litellm.items():
        if key == "sample_spec" or not isinstance(spec, dict):
            continue
        provider = spec.get("litellm_provider") or ""
        if not provider.startswith("bedrock"):
            continue
        prefix = provider + "/"
        model = key[len(prefix) :] if key.startswith(prefix) else key
        rows.append((provider, model, spec.get("supported_endpoints") or []))
    return rows


def match_key(model: str) -> str:
    return VERSION_SUFFIX.sub("", CROSS_REGION_PREFIX.sub("", model))


if __name__ == "__main__":
    main()

"""Print the currently available free OpenRouter models (no API key needed).

Free models are rotated frequently. Run this when answers start failing, then put
the slugs you like into the OPENROUTER_MODELS environment variable.

    python scripts/list_free_models.py
"""

import json
import sys
import urllib.request

URL = "https://openrouter.ai/api/v1/models"


def main() -> int:
    with urllib.request.urlopen(URL, timeout=30) as resp:
        models = json.load(resp)["data"]

    free = [
        m for m in models
        if m["id"].endswith(":free")
        or (m["pricing"].get("prompt") == "0" and m["pricing"].get("completion") == "0")
    ]
    # Text-in/text-out chat models only.
    free = [m for m in free if "text" in m.get("architecture", {}).get("input_modalities", ["text"])
            and m.get("architecture", {}).get("output_modalities", ["text"]) == ["text"]]
    free.sort(key=lambda m: m.get("context_length") or 0, reverse=True)

    print(f"{len(free)} free text models:\n")
    for m in free:
        print(f"{m['id']:<55} ctx={m.get('context_length')}")
    print("\nExample:\n  OPENROUTER_MODELS=" + ",".join(m["id"] for m in free[:3]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

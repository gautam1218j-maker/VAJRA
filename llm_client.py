"""OpenRouter client for VAJRA.

Turns numeric nowcast outputs into a short plain-language advisory.
The ConvLSTM does the forecasting; the LLM only explains the numbers.

Setup (PowerShell, current session only):
    $env:OPENROUTER_API_KEY = "your-new-key"
Or create a .env file in the project root containing:
    OPENROUTER_API_KEY=your-new-key
and make sure .env is listed in .gitignore.
"""

import os
from pathlib import Path

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader (avoids adding python-dotenv as a dependency)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _api_key() -> str:
    _load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Set it as an environment variable "
            "or in a .env file (never hardcode it in source)."
        )
    return key


def chat(messages, model=DEFAULT_MODEL, temperature=0.3, max_tokens=400, timeout=60) -> str:
    """Send a chat request to OpenRouter and return the reply text."""
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


SYSTEM_PROMPT = (
    "You write short severe-weather advisories for a thunderstorm and lightning "
    "nowcasting system. Use ONLY the numbers provided. Do not invent locations, "
    "times, or values. Write 2-3 plain sentences for a non-expert, then one "
    "safety action. Do not claim certainty; this is a probabilistic forecast."
)


def _fallback_advisory(summary: dict) -> str:
    """Template used if the API is down or rate-limited, so the demo never breaks."""
    prob = summary.get("lightning_prob", 0.0)
    lead = summary.get("lead_time_min", "?")
    level = "HIGH" if prob >= 0.7 else "MODERATE" if prob >= 0.4 else "LOW"
    return (
        f"{level} lightning risk in the next {lead} minutes "
        f"(probability {prob:.0%}, peak reflectivity {summary.get('max_dbz', '?')} dBZ). "
        "Move indoors if risk is moderate or higher."
    )


def generate_advisory(summary: dict, model=DEFAULT_MODEL) -> str:
    """summary example:
    {"lead_time_min": 30, "max_dbz": 52.0, "lightning_prob": 0.81,
     "storm_motion": "toward the northeast at ~25 km/h"}
    """
    user_msg = "Nowcast output:\n" + "\n".join(f"- {k}: {v}" for k, v in summary.items())
    try:
        return chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            model=model,
        )
    except Exception as exc:  # network error, rate limit, bad key, etc.
        print(f"[llm_client] LLM call failed ({exc}); using template fallback.")
        return _fallback_advisory(summary)


if __name__ == "__main__":
    demo = {
        "lead_time_min": 30,
        "max_dbz": 52.0,
        "lightning_prob": 0.81,
        "storm_motion": "toward the northeast",
    }
    print(generate_advisory(demo))

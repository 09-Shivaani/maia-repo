from pathlib import Path

_DIR = Path(__file__).parent


def load_prompt(name: str, **fields) -> str:
    """Load maia/prompts/<name>.txt and fill {placeholders} (plain replace, so code braces are safe)."""
    text = (_DIR / f"{name}.txt").read_text(encoding="utf-8")
    for k, v in fields.items():
        text = text.replace("{" + k + "}", str(v))
    return text

"""Optional OpenAI guidance. Called only after the user explicitly opts in."""

import json
import os


def load_config(secrets=None):
    """Load API settings from [openai] secrets or environment variables."""
    values = {str(key).lower(): value for key, value in dict(secrets or {}).items()}
    api_key = os.environ.get("OPENAI_API_KEY") or values.get("api_key")
    model = os.environ.get("OPENAI_MODEL") or values.get("model") or "gpt-6-luna"
    if not api_key or str(api_key).startswith("your-"):
        return {}
    return {"api_key": str(api_key).strip(), "model": str(model).strip()}


def generate(flagged, answers, api_key, model, timeout=20):
    """Generate short, bounded guidance; return (result, safe_error_message)."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=timeout, max_retries=1)
        schema = {
            "type": "object",
            "properties": {
                "personal_note": {"type": "string"},
            },
            "required": ["personal_note"],
            "additionalProperties": False,
        }
        response = client.responses.create(
            model=model,
            store=False,
            max_output_tokens=300,
            instructions=(
                "Write one brief, supportive, plain-language personal note for a school-project diabetes screening aid. "
                "A model flag is not a diagnosis and does not establish that the person has or does not have diabetes. "
                "Use only the supplied result and answers. You may mention one relevant answer without interpreting it "
                "as a cause or diagnosis. If flagged, suggest discussing testing with a qualified health professional. "
                "If not flagged, say this does not rule out diabetes and suggest professional advice if they have concerns. "
                "Do not give any other health, diet, exercise, or treatment advice; do not recommend medication; do not "
                "invent facts or repeat sensitive details unnecessarily. Keep it to two sentences. Avoid emojis and markdown."
            ),
            input=json.dumps({"flagged": bool(flagged), "answers": answers}, ensure_ascii=False),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "screening_guidance",
                    "strict": True,
                    "schema": schema,
                }
            },
        )
        parsed = json.loads(response.output_text)
        note = parsed.get("personal_note")
        if not isinstance(note, str) or not note.strip() or len(note) > 600:
            return None, "the response format was not usable"
        return {"personal_note": note.strip()}, None
    except Exception:
        # Do not show exception details: SDK errors can include request or account metadata.
        return None, "the service was unavailable"

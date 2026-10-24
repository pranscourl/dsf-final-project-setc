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


def generate(flagged, safe_suggestions, api_key, model, timeout=20):
    """Generate short, bounded guidance; return (result, safe_error_message)."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=timeout, max_retries=1)
        schema = {
            "type": "object",
            "properties": {
                "encouragement": {"type": "string"},
                "suggestions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["encouragement", "suggestions"],
            "additionalProperties": False,
        }
        response = client.responses.create(
            model=model,
            store=False,
            max_output_tokens=300,
            instructions=(
                "Write a warm, hopeful, plain-language reflection for a school-project diabetes screening aid. "
                "The result and medical disclaimer are already shown above, so do not repeat the verdict, say 'the model "
                "flagged', or restate that this is not a diagnosis. Acknowledge that taking time to check in is a useful "
                "step. Personalize one or two practical suggestions using ONLY the supplied safe_suggestions; you may "
                "gently rephrase them but must not add new health claims, diagnoses, diet or exercise prescriptions, or "
                "treatment advice. If the safe suggestions are empty, offer a supportive invitation to discuss questions "
                "with a health professional. Never infer causes or future risk. Avoid repeating sensitive answers. "
                "Keep the encouragement to one or two sentences and provide at most two short suggestions. Avoid emojis and markdown."
            ),
            input=json.dumps({
                "flagged": bool(flagged),
                "safe_suggestions": safe_suggestions,
            }, ensure_ascii=False),
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
        encouragement = parsed.get("encouragement")
        suggestions = parsed.get("suggestions")
        if (not isinstance(encouragement, str) or not encouragement.strip()
                or len(encouragement) > 300 or not isinstance(suggestions, list)
                or len(suggestions) > 2
                or not all(isinstance(item, str) and item.strip() and len(item) <= 300 for item in suggestions)):
            return None, "the response format was not usable"
        return {
            "encouragement": encouragement.strip(),
            "suggestions": [item.strip() for item in suggestions],
        }, None
    except Exception:
        # Do not show exception details: SDK errors can include request or account metadata.
        return None, "the service was unavailable"

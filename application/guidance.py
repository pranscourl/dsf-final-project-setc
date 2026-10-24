"""Plain-language, local screening guidance shared by the app and email."""

DISCLAIMER = (
    "This is a school project based on a 2015 U.S. health survey. It estimates whether answers resemble those "
    "of survey respondents who reported a diabetes diagnosis. It cannot diagnose or rule out diabetes, predict "
    "future diabetes, or replace advice from a qualified health professional."
)


def _verdict(flagged, target_recall):
    if flagged:
        return {
            "level": "higher",
            "headline": "It may be worth talking with a health professional",
            "summary": (
                "Based on the answers you shared, a follow-up conversation may be helpful. This screening cannot tell "
                "whether you have diabetes; a health professional can help you decide whether testing makes sense for you."
            ),
        }

    return {
        "level": "lower",
        "headline": "This screening did not suggest a follow-up",
        "summary": (
            "That may feel reassuring, but no questionnaire can rule out diabetes. If you have symptoms or concerns, "
            "talk with a health professional about whether testing is right for you."
        ),
    }


def personal_tips(answers):
    """Offer brief follow-up prompts based only on answers the respondent gave."""
    prompts = []
    for key, label in (
        ("HighBP", "high blood pressure"),
        ("HighChol", "high cholesterol"),
        ("Stroke", "a history of stroke"),
        ("HeartDiseaseorAttack", "coronary heart disease or a heart attack"),
    ):
        if answers.get(key) == 1:
            prompts.append(f"You mentioned {label}. If you speak with a health professional, sharing that history may help give them context.")

    if answers.get("Fruits") == 0:
        prompts.append("You said you eat fruit less than once a day. If it feels doable, one small idea is to add a fruit you enjoy to a usual meal or snack.")
    if answers.get("HvyAlcoholConsump") == 1:
        prompts.append("Your answer matched the survey's higher alcohol-use category. If you want to talk about drinking or making a change, a health professional can help you explore options without judgment.")
    if answers.get("DiffWalk") == 1:
        prompts.append("You reported serious difficulty walking or climbing stairs. Mention this if you ask for advice about activity or screening.")
    if answers.get("PhysHlth", 0) > 0:
        prompts.append(f"You reported {answers['PhysHlth']} days of poor physical health in the past month; consider sharing that with a health professional if it concerns you.")
    if answers.get("MentHlth", 0) > 0:
        prompts.append(f"You reported {answers['MentHlth']} days of poor mental health in the past month. If you would like support, consider talking with a trusted person or health professional.")

    return prompts[:4]


def build(flagged, answers, target_recall=None):
    """Return a result explanation and concise next steps tailored to the flag and answers."""
    out = _verdict(flagged, target_recall)
    sections = []

    if flagged:
        sections.append({
            "title": "A helpful next step",
            "items": [
                "If you are concerned, ask a health professional whether a diabetes test would be appropriate for you. Only clinical assessment can diagnose diabetes.",
                "You can show them this result and the answers you entered. Do not start, stop, or change medication based on this screening result.",
            ],
        })
    else:
        sections.append({
            "title": "If something still worries you",
            "items": [
                "Even when this screening does not suggest follow-up, it cannot rule out diabetes. If you have symptoms or concerns, ask a health professional whether testing would make sense for you.",
                "You can show them this result and the answers you entered. A test and clinical assessment are needed for diagnosis.",
            ],
        })

    personal = personal_tips(answers)
    if personal:
        sections.append({"title": "Follow-up points from your answers", "items": personal})

    out["sections"] = sections
    return out

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
            "headline": "The model flagged these answers for follow-up",
            "summary": (
                "This screening result is not a diagnosis. The model found a pattern similar to some survey "
                "respondents who reported diagnosed diabetes. Consider discussing diabetes testing with a "
                "qualified health professional."
            ),
        }

    recall_note = " The model was set using a training recall target, but it can still miss cases." if target_recall else ""
    return {
        "level": "lower",
        "headline": "The model did not flag these answers",
        "summary": (
            "This does not rule out diabetes, and the result is not a diagnosis."
            + recall_note
            + " If you have symptoms, concerns, or a reason to be screened, discuss testing with a qualified "
            "health professional."
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
            prompts.append(f"You reported {label}; consider mentioning this history if you discuss screening with a health professional.")

    if answers.get("Fruits") == 0:
        prompts.append("You reported eating fruit less than once a day. If you want to make a food change, consider a small, realistic step that fits your preferences and budget.")
    if answers.get("HvyAlcoholConsump") == 1:
        prompts.append("Your answer met the survey's heavy-drinking category. If alcohol is a concern for you, a health professional can discuss support options.")
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
            "title": "Suggested next step",
            "items": [
                "Consider asking a qualified health professional whether a blood test is appropriate. Only a health professional can interpret testing and diagnose diabetes.",
                "You can show them this result and the answers you entered. Do not start, stop, or change medication based on this screening result.",
            ],
        })
    else:
        sections.append({
            "title": "If you still have concerns",
            "items": [
                "A result that was not flagged cannot rule out diabetes. If you have symptoms, concerns, or a reason to be screened, ask a qualified health professional about testing.",
                "You can show them this result and the answers you entered. A test and clinical assessment are needed for diagnosis.",
            ],
        })

    personal = personal_tips(answers)
    if personal:
        sections.append({"title": "Follow-up points from your answers", "items": personal})

    out["sections"] = sections
    return out

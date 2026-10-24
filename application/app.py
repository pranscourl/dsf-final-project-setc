"""
Diabetes Risk Screener: a local Streamlit app.

Run from this folder with:   streamlit run app.py

The questions are built from the features stored inside the saved model file, so the app keeps
working if a different feature set is chosen when the model is retrained.
"""
import sklearn
import streamlit as st

import emailer
import guidance
import ai_guidance
import risk_model as rm

st.set_page_config(page_title="Diabetes Risk Screener", layout="centered")

# ------------------------------------------------------------------------------------------------
# Question catalogue: every one of the 21 survey answers, in the words of the survey.
# ------------------------------------------------------------------------------------------------
YES_NO = {
    "HighBP": ("Has a doctor or other health professional ever told you that you have high blood pressure?", None),
    "HighChol": ("Has a doctor or other health professional ever told you that your blood cholesterol is high?", None),
    "CholCheck": ("Have you had your cholesterol checked in the past 5 years?", None),
    "Stroke": ("Have you ever been told that you had a stroke?", None),
    "HeartDiseaseorAttack": ("Have you ever been told that you had coronary heart disease or a heart attack?", None),
    "Smoker": ("Have you smoked at least 100 cigarettes (about 5 packs) in your lifetime?", None),
    "PhysActivity": ("In the past 30 days, have you done any physical activity or exercise outside your regular job?", None),
    "Fruits": ("Do you eat fruit at least once a day?", None),
    "Veggies": ("Do you eat vegetables at least once a day?", None),
    "HvyAlcoholConsump": ("In a typical week, do you drink more than 14 alcoholic drinks if male, or more than 7 if female?",
                          "This reflects the survey's classification for this feature; it is not a health recommendation."),
    "AnyHealthcare": ("Do you have any kind of health care coverage, including insurance or a government plan?", None),
    "NoDocbcCost": ("In the past 12 months, was there a time you needed to see a doctor but could not because of the cost?", None),
    "DiffWalk": ("Do you have serious difficulty walking or climbing stairs?", None),
}

# Sections of the form, in order. Only questions the loaded model uses are shown.
SECTIONS = [
    ("About you", ["Sex", "Age", "BMI"]),
    ("Medical history", ["HighBP", "HighChol", "CholCheck", "Stroke", "HeartDiseaseorAttack"]),
    ("Day-to-day health", ["GenHlth", "PhysHlth", "MentHlth", "DiffWalk"]),
    ("Lifestyle", ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"]),
    ("Healthcare access", ["AnyHealthcare", "NoDocbcCost"]),
    ("Education and income", ["Education", "Income"]),
]
KNOWN = {f for _, fs in SECTIONS for f in fs}


def get_email_config() -> dict:
    """Email settings from .streamlit/secrets.toml or environment variables. {} means email is not set up."""
    try:
        secrets = st.secrets["email"]
    except Exception:                       # no secrets file, or no [email] table
        secrets = None
    return emailer.load_config(secrets)


def get_openai_config() -> dict:
    """Return optional OpenAI settings without exposing the API key in the interface."""
    try:
        settings = dict(st.secrets.get("openai", {}))
    except Exception:
        settings = {}
    return ai_guidance.load_config(settings)


EMAIL_CFG = get_email_config()
OPENAI_CFG = get_openai_config()


# ------------------------------------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_bundle(path: str) -> dict:
    return rm.load_bundle(path)


chosen = rm.MODELS_DIR / rm.DEFAULT_MODEL
if not chosen.is_file():
    st.error(f"The selected model was not found: `{chosen}`. Rebuild it with "
             "`notebooks/05_modeling_cmb(original70K)/02_train.ipynb`.")
    st.stop()

try:
    bundle = get_bundle(str(chosen))
except Exception as e:                                        # corrupt file, wrong format, ...
    st.error(f"Could not load the selected model: {e}")
    st.stop()

FEATURES = bundle["features"]
unknown = [f for f in FEATURES if f not in KNOWN]
if unknown:
    st.error(f"The model uses features this app has no question for: {unknown}")
    st.stop()

with st.sidebar:
    st.header("About this model")
    st.caption("70k M1 · real survey training data")
    st.write(f"**Questions asked:** {len(FEATURES)}")
    if bundle.get("training_data"):
        st.write(f"**Training data:** {bundle['training_data']}")
    if "target_recall" in bundle:
        st.write(f"**Training recall target:** {bundle['target_recall']:.0%}")
    if bundle.get("sklearn_version") and bundle["sklearn_version"] != sklearn.__version__:
        st.warning(f"The model was saved with scikit-learn {bundle['sklearn_version']}, but this app runs "
                   f"{sklearn.__version__}. Results may be wrong. Install the same version.")
    st.divider()
    st.caption("A school project built on the 2015 BRFSS survey. It is a screening aid, not a medical device, "
               "and it cannot diagnose anything.")

# ------------------------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------------------------
st.title("Diabetes Risk Screener")
st.write("Answer a few questions to see whether your responses resemble those of people in the survey who "
         "reported that a health professional had diagnosed them with diabetes.")
st.info("This school project uses a 2015 U.S. health survey. It does not diagnose diabetes or predict who will "
        "develop it in the future. Your answers stay in this session unless you choose to email a copy or opt in "
        "to OpenAI-generated guidance.",
        )
st.caption(f"The questionnaire uses {len(FEATURES)} answers. You can change any answer before submitting.")

# Units sit outside the form so that the height and weight boxes change as soon as you switch.
units = None
if "BMI" in FEATURES:
    units = st.radio("Which units would you like to use for height and weight?", ["Metric (cm, kg)", "Imperial (ft/in, lb)"], horizontal=True)

# ------------------------------------------------------------------------------------------------
# The questionnaire updates as answers change; scoring happens only after submission.
# ------------------------------------------------------------------------------------------------
answers, shown, missing = {}, {}, []     # model codes, readable values, unanswered questions
progress_slot = st.empty()


def record(feature, label, code, display):
    """Store an answer, or note it as missing."""
    if code is None:
        missing.append(label)
    else:
        answers[feature] = code
        shown[label] = display


def ask_yes_no(feature, column):
    question, help_text = YES_NO[feature]
    with column:
        choice = st.radio(question, ["Yes", "No"], index=None, horizontal=True, help=help_text, key=f"q_{feature}")
    record(feature, question, None if choice is None else int(choice == "Yes"), choice)


def ask_sex():
    choice = st.radio("What is your sex?", ["Female", "Male"], index=None, horizontal=True, key="q_Sex",
                      help="The survey recorded sex as female or male.")
    record("Sex", "Sex category", None if choice is None else int(choice == "Male"), choice)


def ask_age():
    age = st.number_input("How old are you, in years?", min_value=18, max_value=120, value=None, step=1, key="q_Age",
                          placeholder="e.g. 45", help="The survey covers adults aged 18 and over.")
    if age is None:
        record("Age", "Age", None, None)
    else:
        code = rm.age_to_code(int(age))
        record("Age", "Age", code, f"{int(age)} years (survey age group {rm.age_group_label(code)})")


def ask_height_weight():
    c1, c2 = st.columns(2)
    if units.startswith("Metric"):
        with c1:
            h = st.number_input("What is your height in centimeters?", min_value=100.0, max_value=250.0, value=None, step=1.0,
                                key="q_h_cm", placeholder="e.g. 165")
        with c2:
            w = st.number_input("What is your weight in kilograms?", min_value=30.0, max_value=300.0, value=None, step=0.5,
                                key="q_w_kg", placeholder="e.g. 68")
        height_cm = h
        weight_kg = w
    else:
        with c1:
            f1, f2 = st.columns(2)
            ft = f1.number_input("What is your height in feet?", min_value=3, max_value=8, value=None, step=1, key="q_h_ft", placeholder="5")
            inch = f2.number_input("And how many additional inches?", min_value=0, max_value=11, value=None, step=1, key="q_h_in", placeholder="7")
        with c2:
            lb = st.number_input("What is your weight in pounds?", min_value=66.0, max_value=660.0, value=None, step=1.0,
                                 key="q_w_lb", placeholder="150")
        height_cm = None if ft is None or inch is None else rm.feet_inches_to_cm(ft, inch)
        weight_kg = None if lb is None else rm.pounds_to_kg(lb)

    if height_cm is None or weight_kg is None:
        record("BMI", "Height and weight", None, None)
    else:
        bmi = rm.compute_bmi(height_cm, weight_kg)
        if not (rm.BMI_MIN <= bmi <= rm.BMI_MAX):
            st.error(f"Your height and weight give a BMI of {bmi:.1f}, outside the range this tool was built for "
                     f"({rm.BMI_MIN}-{rm.BMI_MAX}). Please check them.")
            record("BMI", "Height and weight", None, None)
        else:
            record("BMI", "Height and weight", rm.bmi_to_code(bmi),
                   f"BMI {bmi:.1f} ({rm.bmi_category(bmi)})")


def ask_general_health():
    options = list(rm.GENERAL_HEALTH.values())
    choice = st.radio("How would you rate your general health?", options, index=None, horizontal=True, key="q_GenHlth")
    code = None if choice is None else options.index(choice) + 1
    record("GenHlth", "General health", code, choice)


def ask_days(feature, label, help_text):
    days = st.number_input(label, min_value=0, max_value=30, value=None, step=1,
                           key=f"q_{feature}", placeholder="Enter 0–30 days", help=help_text)
    record(feature, label, days, None if days is None else f"{days} of 30 days")


def ask_choice(feature, label, options: dict, help_text=None):
    choice = st.selectbox(label, list(options), index=None, format_func=options.get, key=f"q_{feature}",
                          placeholder="Select...", help=help_text)
    record(feature, label, choice, None if choice is None else options[choice])


for title, feats in SECTIONS:
    feats = [f for f in feats if f in FEATURES]
    if not feats:
        continue
    with st.container(border=True):
        st.subheader(title)
        yn = [f for f in feats if f in YES_NO]
        for f in feats:
            if f == "Sex": ask_sex()
            elif f == "Age": ask_age()
            elif f == "BMI": ask_height_weight()
            elif f == "GenHlth": ask_general_health()
            elif f == "PhysHlth":
                ask_days("PhysHlth", "During the past 30 days, for how many days was your physical health not good?",
                         "Include physical illness and injury. 0 means none.")
            elif f == "MentHlth":
                ask_days("MentHlth", "During the past 30 days, for how many days was your mental health not good?",
                         "Include stress, depression and problems with emotions. 0 means none.")
            elif f == "Education":
                ask_choice("Education", "What is the highest level of education you have completed?", rm.EDUCATION)
            elif f == "Income":
                ask_choice("Income", "What is your annual household income before taxes?", rm.INCOME,
                           "Total income of everyone in the household, before taxes. The survey used US dollars.")
        if yn:                                                # Yes/No questions in two columns
            cols = st.columns(2)
            for i, f in enumerate(yn):
                ask_yes_no(f, cols[i % 2])

answered = len(answers)
progress_slot.progress(answered / len(FEATURES), text=f"{answered} of {len(FEATURES)} questions answered")

with st.container(border=True):
    st.subheader("Email me a copy (optional)")
    if EMAIL_CFG:
        st.caption("Your answers and result will be sent only to the address you enter. Email is not a private "
                   "channel, so use an address you are comfortable with. Leave it blank to skip.")
    else:
        st.caption("Email is disabled until SMTP settings are added to `application/.streamlit/secrets.toml`. "
                   "Use `application/secrets.toml.example` as the template, then restart the app.")
    e1, e2 = st.columns(2)
    email_raw = e1.text_input("Email address", key="q_email", placeholder="you@example.com",
                              disabled=not EMAIL_CFG)
    name_raw = e2.text_input("Your name (for the greeting)", key="q_name", placeholder="optional",
                             disabled=not EMAIL_CFG, max_chars=60)

use_openai = False
if OPENAI_CFG:
    use_openai = st.checkbox("Use OpenAI for optional personalized guidance", value=False)
    st.caption("Only if selected, your screening result and answers are sent to OpenAI to draft guidance. "
               "Your name and email are not included. API use may incur charges.")
else:
    st.caption("OpenAI guidance is unavailable until an API key is configured in `application/.streamlit/secrets.toml`.")

submitted = st.button("**Show the Screening Result**", type="primary", use_container_width=True)

email = (email_raw or "").strip() if EMAIL_CFG else ""
name = " ".join((name_raw or "").split()) if EMAIL_CFG else ""

if submitted:
    st.session_state.pop("result", None)
    if email and not emailer.is_valid_email(email):
        st.error("That email address does not look right. Please correct it, or clear the box to skip the email.")
    elif missing:
        st.error("Please answer every question. Still missing: " + ", ".join(missing) + ".")
    else:
        prediction = rm.predict(bundle, answers)
        advice = guidance.build(prediction["flagged"], answers, bundle.get("target_recall"))
        ai_error = None
        if use_openai:
            with st.spinner("Preparing personalized guidance..."):
                personalized, ai_error = ai_guidance.generate(
                    prediction["flagged"], shown, OPENAI_CFG["api_key"], OPENAI_CFG["model"]
                )
            if personalized:
                advice["sections"].append({
                    "title": "Personalized reflection from OpenAI",
                    "items": [personalized["personal_note"]],
                })
        result = {**prediction, **advice, "inputs": shown, "answer_codes": dict(answers),
                  "model": chosen.name, "email_status": None, "ai_error": ai_error,
                  "ai_used": bool(use_openai and not ai_error)}
        if email:
            msg = emailer.build_message(email, name, {**prediction, "level": advice["level"]}, shown, advice,
                                        EMAIL_CFG["sender"], chosen.name)
            with st.spinner("Sending your email..."):
                ok, why = emailer.send(msg, EMAIL_CFG)
            result["email_status"] = (ok, email, why)
        st.session_state["result"] = result

# Inputs rerun the app as they change, so remove an old result as soon as answers no longer match it.
saved_result = st.session_state.get("result")
if saved_result and saved_result.get("answer_codes") != answers:
    st.session_state.pop("result", None)

# ------------------------------------------------------------------------------------------------
# Result
# ------------------------------------------------------------------------------------------------
result = st.session_state.get("result")
if result and result["model"] == chosen.name:
    st.divider()
    st.header("Your result")

    if result["level"] == "higher":
        st.warning(f"### {result['headline']}\n\n{result['summary']}")
    else:
        st.success(f"### {result['headline']}\n\n{result['summary']}")

    status = result["email_status"]
    if status:
        ok, address, why = status
        if ok:
            st.success(f"A copy was sent to **{address}**. If it does not arrive in a few minutes, check your spam folder.",
                       )
        else:
            st.warning(f"We could not send the email, but your result is shown here. {why}")

    if result.get("ai_error"):
        st.warning(f"OpenAI guidance could not be generated ({result['ai_error']}). The local screening guidance is shown instead.")
    elif result.get("ai_used"):
        st.caption("OpenAI drafted the personalized guidance using the answers you chose to share. Review it as general information only.")

    with st.expander("How to read this result"):
        st.write("This is a model screening flag based on 14 answers. The model was trained on a balanced sample "
                 "from a 2015 U.S. survey; this score is not a calibrated personal probability and may not fit "
                 "other populations.")
        st.write(f"Internal model score: **{result['score']:.2f}**; flag threshold: **{result['threshold']:.2f}**. "
                 "These values describe the model calculation, not the chance that you have or will develop diabetes.")

    st.header("What you can do next")
    for i, section in enumerate(result["sections"]):
        with st.expander(section["title"], expanded=(i < 2 or section["title"] == "Personalized reflection from OpenAI")):
            for item in section["items"]:
                st.markdown(f"- {item}")

    with st.expander("Your answers"):
        st.table({"Question": list(result["inputs"]), "Answer": list(result["inputs"].values())})

    st.info(guidance.DISCLAIMER)

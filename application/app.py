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
import risk_model as rm

st.set_page_config(page_title="Diabetes Risk Screener", page_icon="🩺", layout="centered")

# ------------------------------------------------------------------------------------------------
# Question catalogue: every one of the 21 survey answers, in the words of the survey.
# ------------------------------------------------------------------------------------------------
YES_NO = {
    "HighBP": ("High blood pressure", "Has a doctor or other health professional ever told you that you have high blood pressure?"),
    "HighChol": ("High cholesterol", "Has a doctor or other health professional ever told you that your blood cholesterol is high?"),
    "CholCheck": ("Cholesterol check in the last 5 years", "Have you had your cholesterol checked in the past 5 years?"),
    "Stroke": ("Ever had a stroke", "Have you ever been told you had a stroke?"),
    "HeartDiseaseorAttack": ("Heart disease or heart attack", "Have you ever been told you had coronary heart disease or a heart attack?"),
    "Smoker": ("Smoked 100+ cigarettes in your life", "Have you smoked at least 100 cigarettes (5 packs) in your entire life?"),
    "PhysActivity": ("Physical activity in the past 30 days", "Any physical activity or exercise in the past 30 days, other than your regular job?"),
    "Fruits": ("Eats fruit every day", "Do you eat fruit one or more times per day?"),
    "Veggies": ("Eats vegetables every day", "Do you eat vegetables one or more times per day?"),
    "HvyAlcoholConsump": ("Heavy drinking", "Men: more than 14 drinks per week. Women: more than 7 drinks per week."),
    "AnyHealthcare": ("Has health insurance or coverage", "Do you have any kind of health care coverage, including insurance, prepaid plans or government plans?"),
    "NoDocbcCost": ("Skipped a doctor visit because of cost", "In the past 12 months, was there a time you needed to see a doctor but could not because of the cost?"),
    "DiffWalk": ("Serious difficulty walking or climbing stairs", "Do you have serious difficulty walking or climbing stairs?"),
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


EMAIL_CFG = get_email_config()


# ------------------------------------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_bundle(path: str) -> dict:
    return rm.load_bundle(path)


model_files = rm.list_models()
if not model_files:
    st.error(f"No model file found in `{rm.MODELS_DIR}`. Run `07_modeling_cmu/02_train.ipynb` first.")
    st.stop()

with st.sidebar:
    st.header("Model")
    if len(model_files) > 1:
        chosen = st.selectbox("Saved model", model_files, format_func=lambda p: p.name,
                              help="Switch between the saved models to compare their answers.")
    else:
        chosen = model_files[0]
    try:
        bundle = get_bundle(str(chosen))
    except Exception as e:                                    # corrupt file, wrong format, ...
        st.error(f"Could not load `{chosen.name}`: {e}")
        st.stop()

FEATURES = bundle["features"]
unknown = [f for f in FEATURES if f not in KNOWN]
if unknown:
    st.error(f"The model uses features this app has no question for: {unknown}")
    st.stop()

with st.sidebar:
    st.caption(f"`{chosen.name}`")
    st.write(f"**Questions asked:** {len(FEATURES)}")
    st.write(f"**Flag threshold:** {bundle['threshold']:.2f}")
    if "target_recall" in bundle:
        st.write(f"**Recall goal:** {bundle['target_recall']:.0%}")
    if "training_data" in bundle:
        st.write(f"**Trained on:** {bundle['training_data']}")
    if bundle.get("sklearn_version") and bundle["sklearn_version"] != sklearn.__version__:
        st.warning(f"The model was saved with scikit-learn {bundle['sklearn_version']}, but this app runs "
                   f"{sklearn.__version__}. Results may be wrong. Install the same version.")
    st.divider()
    st.caption("A school project built on the 2015 BRFSS survey. It is a screening aid, not a medical device, "
               "and it cannot diagnose anything.")

# ------------------------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------------------------
st.title("🩺 Diabetes Risk Screener")
st.write("Answer the questions about yourself (or the person you are checking). The model estimates whether "
         "a follow-up check for diabetes is worth considering, and gives tips on what to do next. All questions are "
         "required except the email. Your answers are not saved anywhere.")

# Units sit outside the form so that the height and weight boxes change as soon as you switch.
units = None
if "BMI" in FEATURES:
    units = st.radio("Units for height and weight", ["Metric (cm, kg)", "Imperial (ft/in, lb)"], horizontal=True)

# ------------------------------------------------------------------------------------------------
# The form: nothing is calculated until "Check my risk" is pressed.
# ------------------------------------------------------------------------------------------------
answers, shown, missing = {}, {}, []     # model codes, readable values, unanswered questions


def record(feature, label, code, display):
    """Store an answer, or note it as missing."""
    if code is None:
        missing.append(label)
    else:
        answers[feature] = code
        shown[label] = display


def ask_yes_no(feature, column):
    label, help_text = YES_NO[feature]
    with column:
        choice = st.radio(label, ["Yes", "No"], index=None, horizontal=True, help=help_text, key=f"q_{feature}")
    record(feature, label, None if choice is None else int(choice == "Yes"), choice)


def ask_sex():
    choice = st.radio("Sex", ["Female", "Male"], index=None, horizontal=True, key="q_Sex",
                      help="The survey recorded sex as female or male.")
    record("Sex", "Sex", None if choice is None else int(choice == "Male"), choice)


def ask_age():
    age = st.number_input("Age (years)", min_value=18, max_value=120, value=None, step=1, key="q_Age",
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
            h = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=None, step=1.0,
                                key="q_h_cm", placeholder="e.g. 165")
        with c2:
            w = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=None, step=0.5,
                                key="q_w_kg", placeholder="e.g. 68")
        height_cm = h
        weight_kg = w
    else:
        with c1:
            f1, f2 = st.columns(2)
            ft = f1.number_input("Height (ft)", min_value=3, max_value=8, value=None, step=1, key="q_h_ft", placeholder="5")
            inch = f2.number_input("(in)", min_value=0, max_value=11, value=None, step=1, key="q_h_in", placeholder="7")
        with c2:
            lb = st.number_input("Weight (lb)", min_value=66.0, max_value=660.0, value=None, step=1.0,
                                 key="q_w_lb", placeholder="150")
        height_cm = None if ft is None or inch is None else rm.feet_inches_to_cm(ft, inch)
        weight_kg = None if lb is None else rm.pounds_to_kg(lb)

    if height_cm is None or weight_kg is None:
        record("BMI", "Height and weight", None, None)
    else:
        bmi = rm.compute_bmi(height_cm, weight_kg)
        record("BMI", "Height and weight", rm.bmi_to_code(bmi),
               f"BMI {bmi:.1f} ({rm.bmi_category(bmi)})")


def ask_general_health():
    options = list(rm.GENERAL_HEALTH.values())
    choice = st.radio("General health", options, index=None, horizontal=True, key="q_GenHlth",
                      help="In general, how would you rate your overall health?")
    code = None if choice is None else options.index(choice) + 1
    record("GenHlth", "General health", code, choice)


def ask_days(feature, label, help_text):
    days = st.slider(label, 0, 30, 0, key=f"q_{feature}", help=help_text)
    record(feature, label, days, f"{days} of 30 days")


def ask_choice(feature, label, options: dict, help_text=None):
    choice = st.selectbox(label, list(options), index=None, format_func=options.get, key=f"q_{feature}",
                          placeholder="Select...", help=help_text)
    record(feature, label, choice, None if choice is None else options[choice])


with st.form("risk_form"):
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
                    ask_days("PhysHlth", "Days in the past 30 when your physical health was not good",
                             "Include physical illness and injury. 0 means none.")
                elif f == "MentHlth":
                    ask_days("MentHlth", "Days in the past 30 when your mental health was not good",
                             "Include stress, depression and problems with emotions. 0 means none.")
                elif f == "Education":
                    ask_choice("Education", "Highest level of education completed", rm.EDUCATION)
                elif f == "Income":
                    ask_choice("Income", "Annual household income (US dollars)", rm.INCOME,
                               "Total income of everyone in the household, before taxes. The survey used US dollars.")
            if yn:                                            # Yes/No questions in two columns
                cols = st.columns(2)
                for i, f in enumerate(yn):
                    ask_yes_no(f, cols[i % 2])

    with st.container(border=True):
        st.subheader("Email me a copy (optional)")
        if EMAIL_CFG:
            st.caption("Add your email to receive your answers, the result and the advice in one message. "
                       "Leave it empty to skip. Your address is used only to send this one email and is not saved. "
                       "Email is not a private channel, so only use an address you are comfortable with.")
        else:
            st.caption("Email delivery is not set up on this computer yet (see `secrets.toml.example`).")
        e1, e2 = st.columns(2)
        email_raw = e1.text_input("Email address", key="q_email", placeholder="you@example.com",
                                  disabled=not EMAIL_CFG)
        name_raw = e2.text_input("Your name (for the greeting)", key="q_name", placeholder="optional",
                                 disabled=not EMAIL_CFG, max_chars=60)

    submitted = st.form_submit_button("Check my risk", type="primary", use_container_width=True)

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
        result = {**prediction, **advice, "inputs": shown, "model": chosen.name, "email_status": None}
        if email:
            msg = emailer.build_message(email, name, {**prediction, "level": advice["level"]}, shown, advice,
                                        EMAIL_CFG["sender"], chosen.name)
            with st.spinner("Sending your email..."):
                ok, why = emailer.send(msg, EMAIL_CFG)
            result["email_status"] = (ok, email, why)
        st.session_state["result"] = result

# ------------------------------------------------------------------------------------------------
# Result
# ------------------------------------------------------------------------------------------------
result = st.session_state.get("result")
if result and result["model"] == chosen.name:
    st.divider()
    st.header("Your result")

    if result["level"] == "higher":
        st.warning(f"### {result['headline']}\n\n{result['summary']}", icon="⚠️")
    else:
        st.success(f"### {result['headline']}\n\n{result['summary']}", icon="✅")

    status = result["email_status"]
    if status:
        ok, address, why = status
        if ok:
            st.success(f"A copy was sent to **{address}**. If it does not arrive in a few minutes, check your spam folder.",
                       icon="📧")
        else:
            st.warning(f"We could not send the email, but your result is shown here. {why}", icon="📧")

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated chance of diabetes", f"{result['adjusted']:.0%}",
              help=f"The model score rescaled to the real-world rate of diabetes (about {rm.REAL_PREVALENCE:.0%} "
                   "of surveyed adults). It is an approximation.")
    c2.metric("Model score", f"{result['score']:.2f}",
              help="Raw model output on the balanced training scale. It overstates real risk, so use it only "
                   "to compare against the threshold.")
    c3.metric("Flag threshold", f"{result['threshold']:.2f}",
              help="People with a score at or above this are flagged. It is set low on purpose, to catch most "
                   "people who have diabetes, at the cost of some false alarms.")

    # Score bar with the threshold marked
    fill, mark = result["score"] * 100, result["threshold"] * 100
    colour = "#E8A33D" if result["flagged"] else "#3BA55D"
    st.markdown(
        f"""
        <div style="position:relative;height:14px;border-radius:7px;background:rgba(128,128,128,.25);margin:14px 0 4px 0;">
          <div style="width:{fill:.1f}%;height:100%;border-radius:7px;background:{colour};"></div>
          <div style="position:absolute;left:{mark:.1f}%;top:-5px;width:3px;height:24px;background:currentColor;"></div>
        </div>
        <div style="font-size:0.8rem;opacity:.7;">Model score {result['score']:.2f} (bar) against the flag threshold
        {result['threshold']:.2f} (vertical line)</div>
        """,
        unsafe_allow_html=True,
    )

    st.header("What you can do next")
    for i, section in enumerate(result["sections"]):
        with st.expander(f"{section['icon']}  {section['title']}", expanded=(i < 2)):
            for item in section["items"]:
                st.markdown(f"- {item}")

    with st.expander("Your answers"):
        st.table({"Question": list(result["inputs"]), "Answer": list(result["inputs"].values())})

    st.info(guidance.DISCLAIMER)

"""
guidance.py
Plain-language wording of the result and the advice that goes with it. Used by the app screen
and by the email, so both always say the same thing. No Streamlit needed.

The advice follows widely used public-health guidance (ADA, WHO, CDC). It is general information,
not medical advice.
"""

DISCLAIMER = ("This is a school project that uses a statistical model trained on a 2015 US health survey. "
              "It can only suggest whether a check is worth considering. It cannot diagnose diabetes, and it is "
              "not a substitute for a doctor. Only a blood test can confirm diabetes.")

WARNING_SIGNS = [
    "**Common warning signs:** being very thirsty, urinating often (especially at night), feeling very tired, "
    "blurry vision, losing weight without trying, cuts or sores that heal slowly, tingling or numb hands or feet, "
    "and frequent infections. If you notice these, get tested **whatever this result says**.",
    "**Go to an emergency room straight away** if high blood sugar comes with nausea or vomiting, stomach pain, "
    "fast or deep breathing, fruity-smelling breath, or confusion and unusual drowsiness.",
]

HEALTHY_HABITS = [
    "**Move more.** Aim for about 150 minutes a week of moderate activity, for example a brisk 30-minute walk on "
    "5 days. Add muscle-strengthening exercise twice a week, and break up long periods of sitting.",
    "**Build a balanced plate.** Fill half with vegetables, a quarter with lean protein (fish, eggs, beans, "
    "chicken) and a quarter with whole grains or other starchy foods. Choose whole grains, beans and whole fruit.",
    "**Cut back on sugary drinks and refined carbs.** Soft drinks, sweetened juice, milk tea, sweets, white bread "
    "and ultra-processed snacks raise blood sugar quickly. Make water your main drink.",
    "**Keep a healthy weight.** If you are overweight, losing even 5-7% of your body weight lowers the risk of "
    "type 2 diabetes substantially. Small, steady changes work better than crash diets.",
    "**Sleep 7-9 hours and manage stress.** Poor sleep and long-term stress make blood sugar harder to control.",
    "**Do not smoke, and keep alcohol low.** Smoking raises diabetes and heart risk. If you drink, keep to no more "
    "than 1 drink a day (women) or 2 (men).",
    "**Check blood pressure and cholesterol** at least once a year. They are linked to diabetes and heart disease.",
]


def _verdict(flagged, target_recall):
    if flagged:
        return {
            "level": "higher",
            "headline": "Higher risk: it is worth getting tested",
            "summary": ("Your answers look similar to those of many people who have diabetes. This does **not** mean "
                        "you have it, and it is not a diagnosis. A simple blood test can tell you for sure, and "
                        "finding diabetes early makes it much easier to manage."),
        }
    if target_recall:
        missed = round((1 - target_recall) * 100)
        caveat = (f"This tool is set up to catch about {round(target_recall * 100)} in 100 people who have "
                  f"diabetes, so about {missed} in 100 are missed.")
    else:
        caveat = "This tool is set up to catch most people who have diabetes, but some are missed."
    return {
        "level": "lower",
        "headline": "Lower risk: you are unlikely to have diabetes right now",
        "summary": ("Your answers look more like those of people without diabetes, which is good news. It is not a "
                    f"guarantee, though. {caveat} Keep up healthy habits, and get tested if you have symptoms or "
                    "you are 35 or older."),
    }


def personal_tips(a):
    """Tips based on the person's own answers. `a` holds the coded answers; missing keys are skipped."""
    tips = []
    bmi = a.get("BMI")
    if bmi is not None:
        if bmi >= 30:
            tips.append(f"**Weight:** your BMI is about {bmi}, in the obesity range. Losing even 5-7% of your body "
                        "weight makes a real difference to diabetes risk, and also helps blood pressure and cholesterol.")
        elif bmi >= 25:
            tips.append(f"**Weight:** your BMI is about {bmi}, in the overweight range. A modest weight loss of 5-7% "
                        "of your body weight, through food and movement, lowers diabetes risk.")
        elif bmi < 18.5:
            tips.append(f"**Weight:** your BMI is about {bmi}, in the underweight range. If your weight dropped "
                        "without you trying, tell a doctor, because unexplained weight loss can be a sign of diabetes.")
    if a.get("HighBP") == 1:
        tips.append("**Blood pressure:** you reported high blood pressure. It often goes together with diabetes and "
                    "both strain the heart and kidneys. Take any prescribed medicine as directed, eat less salt, and "
                    "have your pressure checked regularly.")
    if a.get("HighChol") == 1:
        tips.append("**Cholesterol:** you reported high cholesterol. Eat less fried and fatty food, add fibre "
                    "(oats, beans, vegetables), and ask your doctor how often to recheck it.")
    if a.get("HeartDiseaseorAttack") == 1 or a.get("Stroke") == 1:
        tips.append("**Heart and brain:** you reported heart disease, a heart attack or a stroke. Ask your doctor what "
                    "level of exercise is safe for you, and keep blood sugar, blood pressure and cholesterol under control.")
    if a.get("HvyAlcoholConsump") == 1:
        tips.append("**Alcohol:** you reported heavy drinking. Cutting down helps blood sugar, weight and blood "
                    "pressure. Alcohol can also cause dangerously low blood sugar in people taking diabetes medicine.")
    if a.get("Smoker") == 1:
        tips.append("**Smoking:** smokers are more likely to develop type 2 diabetes and have more complications. "
                    "Ask a doctor or health centre about help to quit.")
    if a.get("PhysActivity") == 0:
        tips.append("**Activity:** you reported no exercise in the past month. Start small, such as 10-minute walks "
                    "after meals, and build up to 30 minutes on most days.")
    if a.get("Fruits") == 0 or a.get("Veggies") == 0:
        tips.append("**Fruit and vegetables:** you do not eat them daily. Adding a portion of vegetables to each meal "
                    "and choosing whole fruit instead of juice is an easy first step.")
    if a.get("DiffWalk") == 1:
        tips.append("**Movement:** walking is hard for you. Chair exercises, swimming or water exercise, and resistance "
                    "bands are gentler options, and a physiotherapist can design a safe plan.")
    if (a.get("GenHlth") or 0) >= 4 or (a.get("PhysHlth") or 0) >= 14:
        tips.append("**General health:** you rated your health as fair or poor, or reported many unwell days. A "
                    "full check-up with a doctor is a good idea, whatever this result says.")
    if (a.get("MentHlth") or 0) >= 14:
        tips.append("**Mental health:** you reported many days of poor mental health. Stress and low mood affect "
                    "eating, sleep and blood sugar. Talking to a professional or someone you trust can help.")
    if a.get("NoDocbcCost") == 1 or a.get("AnyHealthcare") == 0:
        tips.append("**Cost of care:** cost has been a barrier for you. Ask about community health centres, public "
                    "clinics and free screening drives, and ask for generic medicines. A blood sugar test is usually "
                    "one of the cheaper tests.")
    return tips


def build(flagged, answers, target_recall=None):
    """Everything the screen and the email need: headline, summary and a list of advice sections."""
    out = _verdict(flagged, target_recall)
    sections = []

    if flagged:
        sections.append({"title": "Your next step: get tested", "icon": "🧪", "items": [
            "**See a doctor or visit a community health centre** and ask for a diabetes blood test. Common tests are "
            "fasting blood glucose and HbA1c (a 3-month average), sometimes an oral glucose tolerance test.",
            "For a fasting test, have nothing to eat or drink except water for at least 8 hours beforehand. HbA1c "
            "does not need fasting.",
            "**How results are read:** diabetes is usually diagnosed with a fasting glucose of 126 mg/dL (7.0 mmol/L) "
            "or higher, or an HbA1c of 6.5% or higher, confirmed by a repeat test. Prediabetes is a fasting glucose of "
            "100-125 mg/dL (5.6-6.9 mmol/L) or an HbA1c of 5.7-6.4%.",
            "**Prediabetes is a warning, not a sentence.** Healthy eating, more movement and modest weight loss can "
            "often stop it from becoming type 2 diabetes.",
            "Take this summary with you to the appointment so the doctor can see your answers.",
        ]})
        sections.append({"title": "If you have diabetes (newly confirmed or already diagnosed)", "icon": "📋", "items": [
            "**Stay calm: diabetes is manageable.** With good care, most people live long, healthy lives.",
            "**Follow your care plan.** Take medicine or insulin exactly as prescribed, and never stop or change it "
            "without asking your doctor. Ask to join a diabetes self-management education class.",
            "**Monitor.** Check your blood sugar as advised. HbA1c is usually checked every 3-6 months, and the "
            "target is individual (commonly under 7%), so ask what yours should be.",
            "**Eat regularly and smartly.** Keep regular meal times, use the balanced-plate method, choose high-fibre "
            "carbohydrates, and avoid sugary drinks. A dietitian can build a plan around the food you actually eat.",
            "**Stay active.** About 150 minutes of moderate exercise a week plus strength work twice a week. If you use "
            "insulin, ask your doctor how to adjust for exercise.",
            "**Protect eyes, feet, kidneys and heart.** Get a dilated eye exam every year, check your feet daily for "
            "cuts, blisters and sores, have kidney tests (urine albumin and eGFR) yearly, keep blood pressure and "
            "cholesterol in target, and do not smoke.",
            "**Know low blood sugar (hypoglycaemia).** Signs are shakiness, sweating, hunger, confusion and a pounding "
            "heart. If you take insulin or certain tablets, keep fast sugar with you (glucose tablets or half a glass "
            "of juice), take about 15 g, and recheck in 15 minutes.",
            "**Look after your mind.** Feeling overwhelmed by diabetes is common. Tell your care team if you are struggling.",
            "**Stay up to date on vaccines** (such as flu and pneumonia) and ask your doctor what is right for you.",
        ]})
    else:
        sections.append({"title": "Stay on top of screening", "icon": "🗓️", "items": [
            "**Adults aged 35 or older should be tested for diabetes about every 3 years,** even when healthy.",
            "Get tested **earlier and more often** if you are overweight and also have a parent or sibling with "
            "diabetes, high blood pressure, high cholesterol, little physical activity, polycystic ovary syndrome, "
            "or had diabetes during pregnancy.",
            "This result is based on a questionnaire only. It is **not** a blood test.",
        ]})

    personal = personal_tips(answers)
    if personal:
        sections.append({"title": "Based on your answers", "icon": "🎯", "items": personal})

    sections.append({"title": "Healthy habits that lower diabetes risk (and help if you have it)", "icon": "🥗",
                     "items": HEALTHY_HABITS})
    sections.append({"title": "Know the warning signs", "icon": "🚨", "items": WARNING_SIGNS})

    out["sections"] = sections
    return out

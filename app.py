from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """<style>
/* ---------- Global & Header Fix ---------- */
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b1729 50%, #101f35 100%);
    color: #f5f7fa;
}

header[data-testid="stHeader"] {
    background-color: transparent !important;
    height: 3rem;
}

.block-container {
    padding-top: 4.5rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* ---------- Header ---------- */
.hero {
    background: linear-gradient(135deg, #0f2747, #123b5d);
    padding: 2rem 2.2rem;
    border-radius: 20px;
    border: 1px solid rgba(75, 200, 255, 0.25);
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25);
    margin-bottom: 1.8rem;
}

.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-bottom: 0.4rem;
    color: #ffffff;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #b8d8ee;
    line-height: 1.5;
    margin-bottom: 0;
}

/* ---------- Section Cards ---------- */
.section-card {
    background: rgba(17, 34, 56, 0.88);
    border: 1px solid rgba(120, 170, 210, 0.16);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.18);
}

.section-title {
    font-size: 1.2rem;
    font-weight: 750;
    color: #ffffff;
    margin-bottom: 0.2rem;
}

.section-description {
    font-size: 0.88rem;
    color: #9db5c9;
    margin-bottom: 0;
}

/* ---------- Inputs & Expander ---------- */
label {
    color: #d8e8f5 !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #101f33;
    border: 1px solid #294762;
    border-radius: 10px;
}

input {
    color: #ffffff !important;
}

div[data-testid="stExpander"] {
    background-color: rgba(17, 34, 56, 0.5);
    border: 1px solid rgba(120, 170, 210, 0.2);
    border-radius: 14px;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
}

/* ---------- Button ---------- */
.stButton > button {
    width: 100%;
    height: 3.2rem;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #16b9e8, #247cff);
    color: white;
    font-size: 1.05rem;
    font-weight: 750;
    box-shadow: 0 8px 22px rgba(36, 124, 255, 0.25);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(36, 124, 255, 0.4);
}

/* ---------- Prediction Result ---------- */
.prediction-card {
    background: linear-gradient(135deg, #102d4b, #123e5a);
    border: 1px solid rgba(63, 207, 255, 0.45);
    border-radius: 20px;
    padding: 2.2rem;
    text-align: center;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.3);
}

.prediction-label {
    color: #a9c8dd;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}

.prediction-value {
    font-size: 3.8rem;
    font-weight: 850;
    color: #54d6ff;
    line-height: 1.1;
}

.prediction-scale {
    color: #8faec4;
    font-size: 0.95rem;
    margin-top: 0.5rem;
}

/* ---------- Sidebar & Footer ---------- */
section[data-testid="stSidebar"] {
    background: #081524;
    border-right: 1px solid #19334b;
}

.footer {
    text-align: center;
    color: #6f899e;
    font-size: 0.8rem;
    padding-top: 2rem;
    padding-bottom: 1rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================


@st.cache_resource
def load_model():
    model_path = Path("models/gradient_boosting_model.pkl")

    if not model_path.exists():
        st.error(
            "Model file not found. Please make sure "
            "'models/gradient_boosting_model.pkl' exists."
        )
        st.stop()

    return joblib.load(model_path)


model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """<div style="font-size:1.35rem; font-weight:800; color:#ffffff; padding: 0.2rem 0; letter-spacing:-0.2px;">
🎓 Student Performance AI
</div>
<hr style="border:none; border-top:1px solid #19334b; margin:0.8rem 0 1.2rem 0;">
<div style="display:flex; flex-direction:column; gap:1.1rem; color:#91aabd; font-size:0.88rem; line-height:1.55;">
    <div>
        <span style="color:#e2ecf5; font-size:0.92rem; font-weight:700; display:block; margin-bottom:0.35rem;">About this application</span>
        Predict a student's estimated final grade using academic, demographic, family, and behavioral factors.
    </div>
    <div>
        <span style="color:#e2ecf5; font-weight:600; display:block; margin-bottom:0.2rem;">🤖 Model</span>
        Gradient Boosting Regressor
    </div>
    <div>
        <span style="color:#e2ecf5; font-weight:600; display:block; margin-bottom:0.2rem;">🎯 Prediction Target</span>
        Final Grade (G3), on a 0–20 scale
    </div>
    <div>
        <span style="color:#e2ecf5; font-weight:600; display:block; margin-bottom:0.2rem;">📊 Dataset</span>
        UCI Student Performance Dataset
    </div>
    <div style="background:rgba(255,184,0,0.08); border:1px solid rgba(255,184,0,0.25); border-radius:8px; padding:0.75rem 0.85rem; font-size:0.82rem; line-height:1.45; color:#d8c29d;">
        <span style="color:#ffcc66; font-weight:700; display:block; margin-bottom:0.25rem;">⚠️ Important</span>
        This model provides an estimate based on historical training data and should not replace formal academic assessment.
    </div>
</div>""",
        unsafe_allow_html=True,
    )


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """<div class="hero">
<div class="hero-title">STUDENT PERFORMANCE PREDICTOR</div>
<div class="hero-subtitle">Estimate final academic grade from student information.</div>
</div>""",
    unsafe_allow_html=True,
)


# ============================================================
# INPUT FORM
# ============================================================

with st.form("student_prediction_form"):

    # ========================================================
    # 1. ACADEMIC PERFORMANCE
    # ========================================================

    st.markdown(
        """<div class="section-card">
<div class="section-title">📚 Academic Performance</div>
<div class="section-description">Key historical grades and immediate study metrics.</div>
</div>""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        G1 = st.number_input(
            "G1 (First Period)", min_value=0, max_value=20, value=15
        )

    with c2:
        G2 = st.number_input(
            "G2 (Second Period)", min_value=0, max_value=20, value=16
        )

    with c3:
        failures = st.selectbox("Previous Failures", [0, 1, 2, 3], index=0)

    c4, c5 = st.columns(2)

    with c4:
        absences = st.number_input(
            "Absences", min_value=0, max_value=75, value=4
        )

    with c5:
        studytime = st.selectbox(
            "Study Time",
            [1, 2, 3, 4],
            index=1,
            format_func=lambda x: {
                1: "Less than 2 hours",
                2: "2–5 hours",
                3: "5–10 hours",
                4: "More than 10 hours",
            }[x],
        )

    # ========================================================
    # 2. STUDENT & FAMILY
    # ========================================================

    st.markdown(
        """<div class="section-card">
<div class="section-title">👤 Student & Family</div>
<div class="section-description">Core demographic and household background.</div>
</div>""",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Basic student information
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=15, max_value=22, value=17)

    with c2:
        sex_label = st.selectbox("Sex", ["Female", "Male"])
        sex = "F" if sex_label == "Female" else "M"

    with c3:
        school_label = st.selectbox(
            "School", ["Gabriel Pereira (GP)", "Mousinho da Silveira (MS)"]
        )
        school = "GP" if school_label == "Gabriel Pereira (GP)" else "MS"

    # --------------------------------------------------------
    # Family size and education
    # --------------------------------------------------------

    c4, c5, c6 = st.columns(3)

    with c4:
        famsize_label = st.selectbox(
            "Family Size", ["3 or fewer members", "More than 3 members"]
        )
        famsize = "LE3" if famsize_label == "3 or fewer members" else "GT3"

    education_options = {
        "None": 0,
        "Primary education (up to 4th grade)": 1,
        "5th–9th grade": 2,
        "Secondary education": 3,
        "Higher education": 4,
    }

    with c5:
        Medu_label = st.selectbox(
            "Mother's Education", list(education_options.keys()), index=2
        )
        Medu = education_options[Medu_label]

    with c6:
        Fedu_label = st.selectbox(
            "Father's Education", list(education_options.keys()), index=2
        )
        Fedu = education_options[Fedu_label]

    # --------------------------------------------------------
    # Parent status and family support
    # --------------------------------------------------------

    c7, c8 = st.columns(2)

    with c7:
        Pstatus_label = st.selectbox(
            "Parents' Living Arrangement",
            ["Living together", "Living separately"],
        )
        Pstatus = "T" if Pstatus_label == "Living together" else "A"

    with c8:
        famsup_label = st.selectbox(
            "Family Educational Support", ["Yes", "No"]
        )
        famsup = "yes" if famsup_label == "Yes" else "no"

    # ========================================================
    # 3. SCHOOL & ACTIVITIES
    # ========================================================

    st.markdown(
        """<div class="section-card">
<div class="section-title">🏫 School & Activities</div>
<div class="section-description">Educational assistance and extra-curricular involvement.</div>
</div>""",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        schoolsup = st.selectbox("School Support", ["no", "yes"])
        activities = st.selectbox("Extracurricular Activities", ["yes", "no"])

    with c2:
        internet = st.selectbox("Internet Access", ["yes", "no"])
        paid = st.selectbox("Extra Paid Classes", ["no", "yes"])

    # ========================================================
    # 4. ADVANCED INFORMATION
    # ========================================================

    with st.expander(
        "▼ Advanced Information (Demographics, Lifestyle & Wellbeing)",
        expanded=False,
    ):

        st.caption(
            "Additional information used by the prediction model. Pre-filled values are used if you do not change them."
        )

        # ----------------------------------------------------
        # Demographic information
        # ----------------------------------------------------

        a1, a2, a3 = st.columns(3)

        with a1:
            address_label = st.selectbox(
                "Residential Area", ["Urban", "Rural"]
            )
            address = "U" if address_label == "Urban" else "R"

            guardian = st.selectbox(
                "Guardian", ["Mother", "Father", "Other"]
            )
            guardian = {
                "Mother": "mother",
                "Father": "father",
                "Other": "other",
            }[guardian]

            traveltime = st.selectbox(
                "Travel Time",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Less than 15 minutes",
                    2: "15–30 minutes",
                    3: "30 minutes–1 hour",
                    4: "More than 1 hour",
                }[x],
            )

        # ----------------------------------------------------
        # Parent jobs and school choice
        # ----------------------------------------------------

        with a2:
            Mjob_label = st.selectbox(
                "Mother's Job",
                ["Teacher", "Health", "Services", "At home", "Other"],
            )
            Mjob = {
                "Teacher": "teacher",
                "Health": "health",
                "Services": "services",
                "At home": "at_home",
                "Other": "other",
            }[Mjob_label]

            Fjob_label = st.selectbox(
                "Father's Job",
                ["Teacher", "Health", "Services", "At home", "Other"],
            )
            Fjob = {
                "Teacher": "teacher",
                "Health": "health",
                "Services": "services",
                "At home": "at_home",
                "Other": "other",
            }[Fjob_label]

            reason_label = st.selectbox(
                "School Choice Reason",
                ["Near home", "School reputation", "Course preference", "Other"],
            )
            reason = {
                "Near home": "home",
                "School reputation": "reputation",
                "Course preference": "course",
                "Other": "other",
            }[reason_label]

        # ----------------------------------------------------
        # Education and relationships
        # ----------------------------------------------------

        with a3:
            higher_label = st.selectbox(
                "Wants Higher Education", ["Yes", "No"]
            )
            higher = "yes" if higher_label == "Yes" else "no"

            nursery_label = st.selectbox(
                "Attended Nursery School", ["Yes", "No"]
            )
            nursery = "yes" if nursery_label == "Yes" else "no"

            romantic_label = st.selectbox(
                "In a Romantic Relationship", ["No", "Yes"]
            )
            romantic = "no" if romantic_label == "No" else "yes"

        # ----------------------------------------------------
        # Lifestyle
        # ----------------------------------------------------

        st.markdown("---")
        st.write("**Lifestyle & Wellbeing (Scale 1–5):**")

        l1, l2, l3, l4, l5 = st.columns(5)

        with l1:
            famrel = st.slider("Family Relations", 1, 5, 4)

        with l2:
            freetime = st.slider("Free Time", 1, 5, 3)

        with l3:
            goout = st.slider("Going Out", 1, 5, 3)

        with l4:
            Dalc = st.slider("Workday Alcohol", 1, 5, 1)

        with l5:
            Walc = st.slider("Weekend Alcohol", 1, 5, 1)

        health = st.slider("Current Health Status", 1, 5, 4)

    # ========================================================
    # SUBMIT BUTTON
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🚀 Predict Final Grade")


# ============================================================
# PREDICTION & RESULTS
# ============================================================

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "school": school,
                "sex": sex,
                "age": age,
                "address": address,
                "famsize": famsize,
                "Pstatus": Pstatus,
                "Medu": Medu,
                "Fedu": Fedu,
                "Mjob": Mjob,
                "Fjob": Fjob,
                "reason": reason,
                "guardian": guardian,
                "traveltime": traveltime,
                "studytime": studytime,
                "failures": failures,
                "schoolsup": schoolsup,
                "famsup": famsup,
                "paid": paid,
                "activities": activities,
                "nursery": nursery,
                "higher": higher,
                "internet": internet,
                "romantic": romantic,
                "famrel": famrel,
                "freetime": freetime,
                "goout": goout,
                "Dalc": Dalc,
                "Walc": Walc,
                "health": health,
                "absences": absences,
                "G1": G1,
                "G2": G2,
            }
        ]
    )

    try:
        raw_prediction = model.predict(input_data)[0]
        prediction = max(0.0, min(20.0, float(raw_prediction)))
    except Exception as e:
        st.error(
            "Prediction failed. Please check that the input "
            "features match the features used to train the model."
        )
        st.exception(e)
        st.stop()

    # ========================================================
    # RESULT
    # ========================================================

    st.markdown("---")

    st.markdown(
        f"""<div class="prediction-card">
<div class="prediction-label">PREDICTED FINAL GRADE</div>
<div class="prediction-value">{prediction:.2f} / 20</div>
<div class="prediction-scale">Estimated G3 score</div>
</div>""",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Prediction comparison
    # --------------------------------------------------------

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("G1 Score", f"{G1}")

    with r2:
        st.metric("G2 Score", f"{G2}")

    with r3:
        st.metric("Prediction (G3)", f"{prediction:.2f}")

    st.caption(
        "This prediction is generated by the trained Gradient Boosting regression pipeline. "
        "It should be interpreted as an estimate rather than a guaranteed final grade."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """<div class="footer">
Student Performance Prediction System · Machine Learning Portfolio Project
</div>""",
    unsafe_allow_html=True,
)
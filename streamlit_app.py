# ============================================================
#  DiabetesPredict — Streamlit Web App
#  Run:    streamlit run streamlit_app.py
#  Deploy: share.streamlit.io  (free, from GitHub)
# ============================================================

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import extra_streamlit_components as stx

import cloud_auth as auth

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesPredict — AI Risk Assessment",
    page_icon="🩺",
    layout="wide",
)

# ── Load model + scaler (cached) ─────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
    "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex",
    "Age", "Education", "Income",
]
FEATURE_LABELS = {
    "HighBP": "High Blood Pressure", "HighChol": "High Cholesterol",
    "CholCheck": "Cholesterol Check", "BMI": "Body Mass Index",
    "Smoker": "Smoker", "Stroke": "Stroke History",
    "HeartDiseaseorAttack": "Heart Disease", "PhysActivity": "Physical Activity",
    "Fruits": "Eats Fruits", "Veggies": "Eats Vegetables",
    "HvyAlcoholConsump": "Heavy Alcohol Use", "AnyHealthcare": "Healthcare Coverage",
    "NoDocbcCost": "Skipped Doctor (cost)", "GenHlth": "General Health",
    "MentHlth": "Poor Mental-Health Days", "PhysHlth": "Poor Physical-Health Days",
    "DiffWalk": "Difficulty Walking", "Sex": "Sex",
    "Age": "Age Group", "Education": "Education", "Income": "Income",
}
LABELS = {0: "No Diabetes", 1: "Pre-Diabetes", 2: "Diabetes"}
ADVICE = {
    0: "Your indicators suggest a **low** diabetes risk. Keep it up with regular "
       "exercise, a balanced diet, and routine check-ups.",
    1: "Your indicators suggest **possible pre-diabetes**. Consider consulting a "
       "doctor — early lifestyle changes can prevent Type 2 diabetes.",
    2: "Your indicators suggest an **elevated** diabetes risk. Please consult a "
       "healthcare professional for a proper diagnosis.",
}
RISK_COLOR = {0: "#059669", 1: "#d97706", 2: "#dc2626"}

# ── Custom styling ───────────────────────────────────────────
st.markdown(
    """
    <style>
      /* Bright, colourful background so the frosted glass is visible */
      .stApp { background:
        linear-gradient(135deg, #aac6ff 0%, #c9d6ff 35%, #e3d2ff 70%, #ffd6ec 100%)
        fixed !important; }
      .block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1180px; }
      h1, h2, h3, h4, p, label, span, div { color: #0f2747; }

      /* Hide the "Press Enter to submit" hint under text inputs */
      div[data-testid="InputInstructions"] { display: none !important; }

      /* Hide the invisible cookie-manager component so it takes no space */
      iframe[title="extra_streamlit_components.CookieManager.cookie_manager"] {
        display: none !important;
      }
      div[data-testid="stElementContainer"]:has(
        iframe[title="extra_streamlit_components.CookieManager.cookie_manager"]) {
        display: none !important;
      }

      /* ── Glassmorphism for all boxes (clear frosted glass) ── */
      div[data-testid="stMetric"],
      div[data-testid="stForm"],
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.40) !important;
        backdrop-filter: blur(22px) saturate(180%);
        -webkit-backdrop-filter: blur(22px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.85) !important;
        border-radius: 8px !important;
        box-shadow: 0 8px 30px rgba(31, 67, 120, 0.18),
                    inset 0 1px 0 rgba(255, 255, 255, 0.6);
      }
      div[data-testid="stMetric"] { padding: 16px 18px; }
      div[data-testid="stMetricLabel"] { color: #3b5b85; }
      div[data-testid="stMetricValue"] { color: #1d4ed8; font-weight: 800; }

      .stButton>button, .stFormSubmitButton>button {
        width: 100%; background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
        color: #1d4ed8; border: 1px solid rgba(255, 255, 255, 0.9);
        border-radius: 8px; padding: 0.7rem; font-weight: 700; font-size: 1.02rem;
      }
      .stButton>button:hover, .stFormSubmitButton>button:hover {
        background: rgba(255, 255, 255, 0.85);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Authentication gate ──────────────────────────────────────
cookies = stx.CookieManager(key="dp_cookie_mgr")

if "user" not in st.session_state:
    st.session_state.user = None

# "Remember me" — restore the session from a browser cookie
# (skip restore right after an explicit logout to avoid a stale re-read)
if st.session_state.user is None and not st.session_state.get("just_logged_out", False):
    _saved = cookies.get("dp_user")
    if _saved:
        st.session_state.user = _saved

if st.session_state.user is None:
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        st.title("Diabetes Risk Assessment")
        st.caption("Please log in or create an account to continue.")

        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

        with tab_login:
            with st.form("login_form"):
                le = st.text_input("Email", key="login_email")
                lp = st.text_input("Password", type="password", key="login_pass")
                remember = st.checkbox("Remember me", value=True)
                if st.form_submit_button("Login", icon=":material/login:"):
                    ok, msg = auth.login(le, lp)
                    if ok:
                        email = le.strip().lower()
                        st.session_state.user = email
                        st.session_state.just_logged_out = False
                        if remember:
                            cookies.set("dp_user", email, key="cookie_set_login")
                        st.rerun()
                    else:
                        st.error(msg)

        with tab_signup:
            with st.form("signup_form"):
                se = st.text_input("Email", key="su_email")
                sp = st.text_input("Choose a Password", type="password", key="su_pass")
                sp2 = st.text_input("Confirm Password", type="password", key="su_pass2")
                if st.form_submit_button("Create Account", icon=":material/person_add:"):
                    if sp != sp2:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = auth.sign_up(se, sp)
                        if ok:
                            email = se.strip().lower()
                            st.session_state.user = email
                            st.session_state.just_logged_out = False
                            cookies.set("dp_user", email, key="cookie_set_signup")
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

    st.stop()

# ── Logged-in: sidebar profile avatar + bottom logout ────────
with st.sidebar:
    _email = st.session_state.user
    _initial = _email[0].upper() if _email else "U"
    st.markdown(
        f"""
        <style>
          /* Make the sidebar a full-height column so logout can sit at the bottom */
          section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:first-of-type {{
            min-height: calc(100vh - 4rem);
          }}
          section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:first-of-type
            > div:last-child {{
            margin-top: auto;
          }}
          .profile-avatar {{
            width: 48px; height: 48px; border-radius: 50%;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: #fff; display: flex; align-items: center; justify-content: center;
            font-weight: 800; font-size: 1.3rem;
            box-shadow: 0 4px 14px rgba(37,99,235,0.35);
          }}
        </style>
        <div class="profile-avatar" title="{_email}">{_initial}</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(_email)

    st.divider()
    st.markdown("#### :material/psychology: About this tool")
    st.markdown(
        "Uses a **Random Forest** model trained on **236,000+** real health "
        "records (CDC BRFSS 2021) to estimate diabetes risk across 3 levels."
    )
    st.markdown(
        ":material/target: **Accuracy:** ~80%  \n"
        ":material/database: **Records:** 236K  \n"
        ":material/category: **Features:** 21"
    )

    st.divider()
    st.markdown("#### :material/lightbulb: Did you know?")
    st.markdown(
        "- Type 2 diabetes is largely **preventable** with a healthy lifestyle.\n"
        "- **High BP, cholesterol & BMI** are top risk factors.\n"
        "- **Early detection** helps prevent serious complications.\n"
        "- 30 min of daily activity can **lower your risk** significantly."
    )

    st.caption(":material/medical_services: Educational use only — "
               "not a medical diagnosis.")
    if st.button("Logout", icon=":material/logout:", use_container_width=True):
        try:
            # Overwrite with an empty value (more reliable than delete across sessions)
            cookies.set("dp_user", "", key="cookie_clear_logout")
        except Exception:
            pass
        st.session_state.user = None
        st.session_state.just_logged_out = True
        st.rerun()

# ── Header ───────────────────────────────────────────────────
st.title("Diabetes Risk Assessment")
st.caption("Based on 236,000+ real health records from the BRFSS 2021 survey.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", "80%")
c2.metric("Training Records", "236K")
c3.metric("Health Features", "21")
c4.metric("Algorithm", "RF")

st.divider()

# ── Input form ───────────────────────────────────────────────
left, right = st.columns([1.2, 1], gap="large")

with left:
    st.subheader(":material/health_and_safety: Health Assessment")

    with st.form("predict_form"):
        tab_cardio, tab_life, tab_care, tab_demo = st.tabs([
            ":material/favorite: Heart",
            ":material/fitness_center: Lifestyle",
            ":material/local_hospital: Healthcare",
            ":material/person: About You",
        ])

        # ── Tab 1 — Cardiovascular ──
        with tab_cardio:
            a, b = st.columns(2)
            HighBP    = a.toggle("High Blood Pressure")
            HighChol  = b.toggle("High Cholesterol")
            CholCheck = a.toggle("Cholesterol Checked (5y)", value=True)
            HeartDis  = b.toggle("Heart Disease / Attack")
            Stroke    = a.toggle("Stroke History")

        # ── Tab 2 — Body & Lifestyle ──
        with tab_life:
            BMI = st.slider("Body Mass Index (BMI)", 12.0, 50.0, 26.5, 0.1)
            a, b = st.columns(2)
            Smoker   = a.toggle("Smoker (100+ cigs)")
            PhysAct  = b.toggle("Physically Active", value=True)
            Fruits   = a.toggle("Eats Fruits Daily", value=True)
            Veggies  = b.toggle("Eats Veggies Daily", value=True)
            HvyAlc   = a.toggle("Heavy Alcohol Use")
            DiffWalk = b.toggle("Difficulty Walking")

        # ── Tab 3 — Healthcare Access ──
        with tab_care:
            a, b = st.columns(2)
            AnyHC   = a.toggle("Has Healthcare Coverage", value=True)
            NoDoc   = b.toggle("Skipped Doctor (cost)")
            GenHlth = st.select_slider(
                "General Health", options=[1, 2, 3, 4, 5], value=3,
                format_func=lambda x: {1: "Excellent", 2: "Very Good", 3: "Good",
                                       4: "Fair", 5: "Poor"}[x])
            MentHlth = st.slider("Poor Mental-Health Days (last 30)", 0, 30, 0)
            PhysHlth = st.slider("Poor Physical-Health Days (last 30)", 0, 30, 0)

        # ── Tab 4 — Demographics ──
        with tab_demo:
            AGE_LABELS = ["18–24", "25–29", "30–34", "35–39", "40–44", "45–49",
                          "50–54", "55–59", "60–64", "65–69", "70–74", "75–79", "80+"]
            Sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
            Age = st.select_slider("Age Group", options=list(range(1, 14)), value=7,
                                   format_func=lambda x: AGE_LABELS[x - 1])
            Education = st.slider("Education Level", 1, 6, 5)
            Income    = st.slider("Income Scale", 1, 11, 6)

        st.markdown("")  # small spacer above the button
        submitted = st.form_submit_button("Run Risk Assessment", icon=":material/search:")

# ── Prediction + results ─────────────────────────────────────
with right.container(border=True):
    st.subheader(":material/monitoring: Result")

    if submitted:
        row = {
            "HighBP": int(HighBP), "HighChol": int(HighChol), "CholCheck": int(CholCheck),
            "BMI": BMI, "Smoker": int(Smoker), "Stroke": int(Stroke),
            "HeartDiseaseorAttack": int(HeartDis), "PhysActivity": int(PhysAct),
            "Fruits": int(Fruits), "Veggies": int(Veggies),
            "HvyAlcoholConsump": int(HvyAlc), "AnyHealthcare": int(AnyHC),
            "NoDocbcCost": int(NoDoc), "GenHlth": GenHlth, "MentHlth": MentHlth,
            "PhysHlth": PhysHlth, "DiffWalk": int(DiffWalk),
            "Sex": 1 if Sex == "Male" else 0, "Age": Age,
            "Education": Education, "Income": Income,
        }
        X = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)
        Xs = scaler.transform(X)
        pred  = int(model.predict(Xs)[0])
        proba = model.predict_proba(Xs)[0]
        risk_score = round((proba[1] * 0.5 + proba[2]) * 100, 1)

        # Gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"suffix": "%", "font": {"size": 40, "color": "#0f2747"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#3b5b85"},
                "bar": {"color": RISK_COLOR[pred]},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33],  "color": "rgba(5,150,105,0.22)"},
                    {"range": [33, 66], "color": "rgba(217,119,6,0.22)"},
                    {"range": [66, 100],"color": "rgba(220,38,38,0.22)"},
                ],
            },
            title={"text": "Risk Score", "font": {"size": 14, "color": "#3b5b85"}},
        ))
        gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10),
                            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True,
                        config={"displayModeBar": False, "staticPlot": True})

        st.markdown(
            f"<h2 style='text-align:center;color:{RISK_COLOR[pred]};margin-top:-10px'>"
            f"{LABELS[pred]}</h2>", unsafe_allow_html=True)
        st.info(ADVICE[pred])

        st.markdown("**Class Probabilities**")
        st.progress(int(proba[0] * 100), text=f"No Diabetes — {proba[0]*100:.1f}%")
        st.progress(int(proba[1] * 100), text=f"Pre-Diabetes — {proba[1]*100:.1f}%")
        st.progress(int(proba[2] * 100), text=f"Diabetes — {proba[2]*100:.1f}%")
    else:
        st.info(":material/science: Fill in the form and press "
                "**Run Risk Assessment** to see your result here.")

# ── Feature importance ───────────────────────────────────────
st.divider()
st.subheader(":material/insights: What drives the model")
st.caption("Top factors the Random Forest weighs most, learned from the data.")

if hasattr(model, "feature_importances_"):
    imp = (pd.Series(model.feature_importances_, index=FEATURES)
           .sort_values(ascending=False).head(10))
    pairs = [(FEATURE_LABELS[i], v * 100) for i, v in imp.items()]
    max_v = max(v for _, v in pairs)

    rows_html = ""
    for name, val in pairs:
        width = val / max_v * 100
        rows_html += (
            '<div style="display:flex;align-items:center;gap:12px;margin-bottom:11px;">'
            f'<div style="width:150px;flex-shrink:0;font-size:0.86rem;color:#3b5b85;'
            f'text-align:right;">{name}</div>'
            '<div style="flex:1;background:rgba(31,67,120,0.10);border-radius:999px;'
            'height:14px;overflow:hidden;">'
            f'<div style="height:100%;width:{width:.1f}%;border-radius:999px;'
            'background:linear-gradient(90deg,#6366f1,#22d3ee);"></div></div>'
            f'<div style="width:48px;flex-shrink:0;font-size:0.82rem;font-weight:700;'
            f'color:#1d4ed8;">{val:.1f}%</div>'
            '</div>'
        )

    st.markdown(f'<div style="margin-top:16px;">{rows_html}</div>',
                unsafe_allow_html=True)

st.divider()
st.caption("DiabetesPredict · Built with Streamlit, scikit-learn & BRFSS 2021. "
           "For educational purposes only — not a medical diagnosis.")

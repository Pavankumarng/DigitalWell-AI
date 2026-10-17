import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="DigitalWell AI", page_icon="🌿", layout="wide")

st.title("🌿 DigitalWell AI — LinuxONE Community Edition")
st.caption("Native s390x Machine Learning Inference running on IBM Z / LinuxONE")
st.write("Explore model-estimated digital-habit risk and compare hypothetical habit changes.")
st.caption("Educational prototype only — not a medical or psychological diagnosis.")

# Load the trained model from beside this file, not from a machine-specific path.
model_path = Path(__file__).resolve().parent / "digitalwell_model.pkl"
if not model_path.exists():
    st.error(
        "Model file not found. Run the training/export cells in "
        "`Digitalwell_Ai.ipynb` to create `digitalwell_model.pkl`, "
        "then place it beside `app.py`."
    )
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as exc:
    st.error(f"Could not load the model artifact: {exc}")
    st.stop()

st.sidebar.header("User Habits Profile")
age = st.sidebar.number_input("Age", 10, 80, 22)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
daily_screen = st.sidebar.slider("Daily Screen Time (hours)", 0.0, 16.0, 6.0, step=0.1)
social_media = st.sidebar.slider("Social Media Time (hours)", 0.0, 12.0, 3.0, step=0.1)
gaming = st.sidebar.slider("Gaming Hours", 0.0, 10.0, 1.0, step=0.1)
work_study = st.sidebar.slider("Work/Study Hours", 0.0, 12.0, 4.0, step=0.1)
sleep = st.sidebar.slider("Sleep Hours", 0.0, 12.0, 7.0, step=0.1)
notifs = st.sidebar.number_input("Notifications per Day", 0, 1000, 120)
app_opens = st.sidebar.number_input("App Opens per Day", 0, 500, 80)
weekend_screen = st.sidebar.slider("Weekend Screen Time (hours)", 0.0, 20.0, 7.0, step=0.1)
stress = st.sidebar.selectbox("Stress Level", ["Low", "Medium", "High"])
impact = st.sidebar.selectbox("Academic / Work Impact", ["No", "Yes", "Not Applicable"])

input_df = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "daily_screen_time_hours": daily_screen,
    "social_media_hours": social_media,
    "gaming_hours": gaming,
    "work_study_hours": work_study,
    "sleep_hours": sleep,
    "notifications_per_day": notifs,
    "app_opens_per_day": app_opens,
    "weekend_screen_time": weekend_screen,
    "stress_level": stress,
    "academic_work_impact": impact,
}])

class_labels = list(model.classes_)
if 1 not in class_labels:
    st.error("The loaded model does not contain the expected positive class label `1`.")
    st.stop()

positive_class_index = class_labels.index(1)
prob = float(model.predict_proba(input_df)[0][positive_class_index]) * 100
pred = int(model.predict(input_df)[0])

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Model-Estimated Risk Score",
        value=f"{prob:.1f}%",
        delta="Elevated model score" if pred == 1 else "Lower model score",
        delta_color="inverse",
    )

with col2:
    if pred == 1:
        st.warning("The model classified this profile as label 1. This is not a diagnosis.")
    else:
        st.success("The model classified this profile as label 0. This is not a diagnosis.")

st.markdown("---")
st.subheader("💡 What-If Habit Simulation")
st.write(
    "Compare how the model output changes under hypothetical input edits. "
    "These are not causal estimates or predictions of guaranteed outcomes."
)

scenarios = {
    "Current Habits": input_df.copy(),
    "Screen Time -2h": input_df.copy(),
    "Social Media -1h": input_df.copy(),
    "Sleep +1h": input_df.copy(),
}

scenarios["Screen Time -2h"]["daily_screen_time_hours"] = max(0.0, float(daily_screen) - 2.0)
scenarios["Social Media -1h"]["social_media_hours"] = max(0.0, float(social_media) - 1.0)
scenarios["Sleep +1h"]["sleep_hours"] = min(24.0, float(sleep) + 1.0)

simulation_rows = []
for scenario_name, profile in scenarios.items():
    scenario_probability = (
        float(model.predict_proba(profile)[0][positive_class_index]) * 100
    )
    simulation_rows.append({
        "Scenario": scenario_name,
        "Model-Estimated Score (%)": round(scenario_probability, 1),
        "Change vs Current (percentage points)": round(scenario_probability - prob, 1),
    })

st.dataframe(pd.DataFrame(simulation_rows), use_container_width=True, hide_index=True)

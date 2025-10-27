import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# Load
@st.cache_resource
def load_artifacts():
    model = joblib.load('student_risk_model.joblib')
    encoder = joblib.load('encoder.joblib')
    return model, encoder

@st.cache_data
def load_data():
    df = pd.read_csv('student_data_cleaned.csv')
    # Pre-compute risks (cached)
    _, encoder = load_artifacts()
    cat_cols = ['gender', 'department', 'parental_education']
    encoded_cats = encoder.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out())
    df_encoded = pd.concat([df.drop(cat_cols, axis=1), encoded_df], axis=1)
    model, _ = load_artifacts()
    df['risk_score'] = model.predict_proba(df_encoded.drop(['student_id', 'dropout'], axis=1))[:, 1]
    return df

model, encoder = load_artifacts()
df = load_data()

st.title("Student Dropout Risk Dashboard")
st.markdown("Real-data AI for early dropout detection (28% rate). Risk: Prob. of dropout.")

# KPIs
col1, col2, col3, col4 = st.columns(4)
total = len(df)
dropout_rate = (df['dropout'].sum() / total) * 100
high_risk = len(df[df['risk_score'] > 0.5])
avg_risk = df['risk_score'].mean() * 100
with col1: st.metric("Students", total)
with col2: st.metric("Dropout Rate", f"{dropout_rate:.1f}%")
with col3: st.metric("High-Risk (>50%)", high_risk)
with col4: st.metric("Avg Risk", f"{avg_risk:.1f}%")

# Trend Chart
st.subheader("Risk by Department")
dept_risk = df.groupby('department')['risk_score'].mean().sort_values(ascending=False)
st.bar_chart(dept_risk)

# Students Section
st.header("Students Overview")
search_id = st.text_input("Search Student ID")
search_dept = st.selectbox("Filter Department", ['All'] + sorted(df['department'].unique()))
filtered = df
if search_id: filtered = filtered[filtered['student_id'].astype(str).str.contains(search_id)]
if search_dept != 'All': filtered = filtered[filtered['department'] == search_dept]
display_cols = ['student_id', 'department', 'cgpa', 'attendance_rate', 'risk_score', 'dropout']
st.dataframe(filtered[display_cols].sort_values('risk_score', ascending=False), use_container_width=True)

# Student Detail
st.header("Student Details")
selected_id = st.selectbox("Select ID", df['student_id'].sort_values().unique())
student_row = df[df['student_id'] == selected_id].iloc[0]
st.json({k: v for k, v in student_row.items() if k != 'risk_score' and k != 'dropout'})

# Risk & SHAP
risk = student_row['risk_score']
observed = student_row['dropout']
st.metric("Predicted Risk Score", f"{risk:.3f}", delta=f"Observed: {observed}")

# SHAP (on encoded)
st.subheader("Feature Impact (SHAP)")
cat_cols = ['gender', 'department', 'parental_education']
student_cats = student_row[cat_cols].values.reshape(1, -1)
encoded_cats = encoder.transform(student_cats)
encoded_student = pd.concat([student_row.drop(cat_cols + ['student_id', 'dropout', 'risk_score']).values.reshape(1, -1), 
                             pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out())], axis=1)
encoded_student.columns = list(model.feature_names_in_)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(encoded_student)[1]  # Dropout class
fig, ax = plt.subplots()
shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value[1], 
                                      data=encoded_student.iloc[0], feature_names=encoded_student.columns))
st.pyplot(fig)

# Simulation
st.header("What-If Simulation")
with st.expander("Adjust Features"):
    sim_cgpa = st.slider("CGPA", 0.0, 10.0, student_row['cgpa'])
    sim_att = st.slider("Attendance %", 0.0, 100.0, student_row['attendance_rate'])
    sim_study = st.slider("Study Hours/Week", 0.0, 50.0, student_row['study_hours_per_week'])
    sim_income = st.slider("Family Income", df['family_income'].min(), df['family_income'].max(), student_row['family_income'])
    # Re-encode (full for accuracy)
    sim_row = student_row.copy()
    sim_row['cgpa'] = sim_cgpa
    sim_row['attendance_rate'] = sim_att
    sim_row['study_hours_per_week'] = sim_study
    sim_row['family_income'] = sim_income
    sim_cats = sim_row[cat_cols].values.reshape(1, -1)
    sim_encoded_cats = encoder.transform(sim_cats)
    sim_encoded = pd.concat([sim_row.drop(cat_cols + ['student_id', 'dropout', 'risk_score']).values.reshape(1, -1), 
                             pd.DataFrame(sim_encoded_cats, columns=encoder.get_feature_names_out())], axis=1)
    sim_encoded.columns = list(model.feature_names_in_)
    sim_risk = model.predict_proba(sim_encoded)[:, 1][0]
    st.metric("Simulated Risk", f"{sim_risk:.3f}", delta=f"Δ{risk - sim_risk:+.3f}")

# Batch Upload (Bonus)
uploaded = st.file_uploader("Upload CSV (must match schema)")
if uploaded:
    batch_df = pd.read_csv(uploaded)
    # Clean/encode batch (simplified; add full cleaning in prod)
    batch_df['scholarship'] = batch_df['scholarship'].map({'y':1, 'yes':1, 'n':0, 'no':0}).fillna(0)
    # ... (apply similar cleaning)
    batch_encoded = pd.get_dummies(batch_df.drop(['student_id', 'dropout'], axis=1), drop_first=True)
    # Align columns
    batch_encoded = batch_encoded.reindex(columns=X.columns, fill_value=0)
    batch_df['risk_score'] = model.predict_proba(batch_encoded)[:, 1]
    st.dataframe(batch_df[['student_id', 'risk_score']])

# Intervention Suggester (Bonus Chatbot)
st.header("Intervention Assistant")
query = st.text_input("e.g., 'Low CGPA advice'")
if query:
    risk_level = "Low" if risk < 0.3 else "Medium" if risk < 0.7 else "High"
    top_driver = "CGPA" if "cgpa" in query.lower() else "Attendance" if "attend" in query.lower() else "General"
    if top_driver == "CGPA":
        st.write(f"**{risk_level} Risk**: Assign peer tutoring (NPTEL resources). Goal: +0.5 GPA in 1 semester.")
    elif top_driver == "Attendance":
        st.write(f"**{risk_level} Risk**: Weekly check-ins + flexible online sessions. Impact: -15% risk.")
    else:
        st.write(f"**{risk_level} Risk ({risk:.2f})**: If >0.5, alert counselor. General: Boost activities for +10% engagement.")
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load the trained model and scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize session state for storing patient records
if "patient_data" not in st.session_state:
    st.session_state.patient_data = []

# Streamlit Dashboard UI
st.title("📊 Advanced Patient Health Monitoring Dashboard")
st.write("Enter patient details to predict lab results and track health trends.")

# Input fields (6 vital signs)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=130, value=80)
temperature = st.number_input("Temperature (°F)", min_value=95.0, max_value=105.0, value=98.6)
heart_rate = st.number_input("Heart Rate", min_value=50, max_value=150, value=75)
oxygen_saturation = st.number_input("Oxygen Saturation", min_value=85, max_value=100, value=98)

# Placeholder values for missing features (15 additional features)
missing_features = [0] * 15

# Prepare input data
input_data = np.array([[age, systolic_bp, diastolic_bp, temperature, heart_rate, oxygen_saturation] + missing_features])

# Ensure correct feature count before transformation
if input_data.shape[1] != 21:
    st.error(f"Feature mismatch: Expected 21, but got {input_data.shape[1]}. Check input features!")
else:
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    if st.button("Predict Lab Result"):
        prediction = model.predict(input_data_scaled)
        result = "Normal" if prediction[0] == 1 else "Abnormal"

        # Store patient data for real-time visualization
        st.session_state.patient_data.append({
            "Age": age,
            "Systolic BP": systolic_bp,
            "Diastolic BP": diastolic_bp,
            "Temperature": temperature,
            "Heart Rate": heart_rate,
            "Oxygen Saturation": oxygen_saturation,
            "Prediction": result
        })

        # Display result
        st.subheader(f"🩺 Predicted Lab Result: **{result}**")

# Convert session state to DataFrame
if st.session_state.patient_data:
    df = pd.DataFrame(st.session_state.patient_data)

    # --- 1. **Bar Chart - Normal vs Abnormal Cases** ---
    st.subheader("📊 Prediction Comparison")
    pred_counts = df["Prediction"].value_counts()
    fig = px.bar(pred_counts, x=pred_counts.index, y=pred_counts.values,
                 labels={"x": "Lab Result", "y": "Count"},
                 color=pred_counts.index, title="🔬 Normal vs Abnormal Cases",
                 color_discrete_map={"Normal": "green", "Abnormal": "red"})
    st.plotly_chart(fig)

    # --- 2. **Box Plot - Distribution of Vitals (Normal vs Abnormal)** ---
    st.subheader("📦 Vitals Distribution by Prediction Category")
    fig = px.box(df, x="Prediction", y=["Systolic BP", "Diastolic BP", "Temperature", "Heart Rate", "Oxygen Saturation"],
                 points="all", color="Prediction", title="Vitals Comparison for Normal vs Abnormal",
                 color_discrete_map={"Normal": "blue", "Abnormal": "red"})
    st.plotly_chart(fig)

    # --- 3. **Violin Plot - Heart Rate Spread** ---
    st.subheader("🎻 Heart Rate Spread by Prediction")
    fig = px.violin(df, x="Prediction", y="Heart Rate", box=True, points="all",
                    color="Prediction", title="Heart Rate Analysis",
                    color_discrete_map={"Normal": "blue", "Abnormal": "red"})
    st.plotly_chart(fig)

    # --- 4. **Scatter Plot - Age vs. Systolic BP** ---
    st.subheader("📌 Age vs Systolic BP Comparison")
    fig = px.scatter(df, x="Age", y="Systolic BP", color="Prediction", size="Systolic BP",
                     title="Age vs. Systolic BP",
                     color_discrete_map={"Normal": "green", "Abnormal": "red"})
    st.plotly_chart(fig)

    # --- 5. **Pie Chart - Percentage of Normal vs Abnormal** ---
    st.subheader("🥧 Health Condition Distribution")
    fig = px.pie(df, names="Prediction", title="Percentage of Normal vs Abnormal Patients",
                 color="Prediction", color_discrete_map={"Normal": "green", "Abnormal": "Red"})
    st.plotly_chart(fig)

    # --- 6. **Enhanced Trend Line - Patient Vitals Over Time (With Linear Regression)** ---
    st.subheader("📈 Patient Vitals Over Time (With Linear Trend)")

    df["Index"] = range(1, len(df) + 1)

    # Define vitals to plot
    vitals = ["Systolic BP", "Diastolic BP", "Temperature", "Heart Rate", "Oxygen Saturation"]

    # Create subplots for better visualization
    fig = go.Figure()

    colors = {
        "Systolic BP": "red",
        "Diastolic BP": "blue",
        "Temperature": "orange",
        "Heart Rate": "green",
        "Oxygen Saturation": "purple"
    }

    # Add traces for each vital sign
    for vital in vitals:
        fig.add_trace(go.Scatter(
            x=df["Index"],
            y=df[vital],
            mode='lines+markers',
            name=vital,
            line=dict(color=colors[vital], width=2)
        ))

        # Add linear trend line
        fig.add_trace(go.Scatter(
            x=df["Index"],
            y=np.poly1d(np.polyfit(df["Index"], df[vital], 1))(df["Index"]),
            mode='lines',
            name=f"{vital} Trend",
            line=dict(color=colors[vital], width=1.5, dash='dot')
        ))

    # Layout customization
    fig.update_layout(
        title="📊 Patient Vitals Trend Over Time (with Linear Regression)",
        xaxis_title="Patient Entry",
        yaxis_title="Vital Sign Values",
        legend_title="Vitals",
        template="plotly_dark",
        hovermode="x"
    )

    st.plotly_chart(fig)

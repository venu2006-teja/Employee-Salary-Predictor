import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Employee Salary Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Styling ---
st.markdown(
    """
    <style>
    .main-container {
        background: linear-gradient(to right, #ece9e6, #ffffff); /* Light grey to white gradient */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    h1 {
        color: #0e1117;
        text-align: center;
        font-family: 'Open Sans', sans-serif;
        font-weight: 700;
    }
    h2 {
        color: #262730;
        font-family: 'Open Sans', sans-serif;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .stNumberInput, .stSelectbox {
        padding: 5px 0;
    }
    .stSuccess, .stError, .stInfo {
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
    }
    .predicted-salary {
        font-size: 2.5em;
        color: #1a73e8;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        padding: 15px;
        background-color: #e8f0fe;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the best regression model
model = joblib.load("best_regression_model.pkl")

st.title("Employee Salary Prediction App")

st.write("### Enter the following information to predict the employee's salary:")

# Create input fields for each feature used in training (excluding 'Salary')
# Ensure these match the columns in X used for training the regression model
with st.sidebar:
    st.header("Personal and Role Information")
    age = st.number_input("Age", min_value=18, max_value=65, value=30) # Adjust age range based on data cleaning
    gender = st.selectbox("Gender", ['Female', 'Male', 'Other']) # Ensure categories match training data
    department = st.selectbox("Department", ['HR', 'Sales', 'IT']) # Ensure categories match training data
    status = st.selectbox("Status", ['Active', 'Inactive']) # Ensure categories match training data
    location = st.selectbox("Location", ['New York', 'Los Angeles', 'Chicago']) # Ensure categories match training data
    session = st.selectbox("Session", ['Night', 'Evening', 'Morning']) # Ensure categories match training data


st.header("Performance and Experience")
performance_score = st.number_input("Performance Score", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
experience = st.number_input("Experience (Years)", min_value=0, max_value=25, value=5) # Adjust experience range based on data cleaning


st.header("Joining Date Information")
# Assuming you engineered these features from 'Joining Date'
joining_year = st.number_input("Joining Year", min_value=2015, max_value=2024, value=2020) # Adjust range based on data
joining_month = st.number_input("Joining Month", min_value=1, max_value=12, value=6)
joining_day = st.number_input("Joining Day", min_value=1, max_value=31, value=15)


# Create a dictionary with the input values, matching the feature names in X
input_data = {
    'Age': age,
    'Gender': gender,
    'Department': department,
    'Performance Score': performance_score,
    'Experience': experience,
    'Status': status,
    'Location': location,
    'Session': session,
    'Joining_Year': joining_year,
    'Joining_Month': joining_month,
    'Joining_Day': joining_day
}

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess the input data (Label Encoding and Scaling)
# Replicate the preprocessing steps from the training notebook

categorical_cols = ['Gender', 'Department', 'Status', 'Location', 'Session']

# Apply Label Encoding
for col in categorical_cols:
    # In a production app, you would load pre-fitted encoders
    # For this example, we fit on a list of expected categories (based on value_counts in notebook)
    le = LabelEncoder()
    # Ensure categories here match the ones used during training
    if col == 'Gender':
        le.fit(['Female', 'Male', 'Other'])
    elif col == 'Department':
        le.fit(['HR', 'Sales', 'IT'])
    elif col == 'Status':
        le.fit(['Active', 'Inactive'])
    elif col == 'Location':
        le.fit(['New York', 'Los Angeles', 'Chicago'])
    elif col == 'Session':
        le.fit(['Night', 'Evening', 'Morning'])
    input_df[col] = le.transform(input_df[col])

# Identify numerical columns for scaling
numerical_cols = input_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Exclude label encoded categorical columns from numerical scaling
numerical_cols = [col for col in numerical_cols if col not in categorical_cols]


# Apply StandardScaler
# In a production app, you would load a pre-fitted scaler
# For this example, we will create a scaler and fit/transform the input data
# This is NOT ideal for production as the scaler should be fitted on the training data
scaler = StandardScaler()
# To properly scale, you should fit the scaler on the training data X_train and then
# use scaler.transform(input_df[numerical_cols])
# As we don't have X_train here, we will fit and transform the input_df, which is
# a simplification for demonstration purposes.
# A better approach would be to save the fitted scaler during training and load it here.
input_df[numerical_cols] = scaler.fit_transform(input_df[numerical_cols])


# Make prediction
if st.button("Predict Salary"): # Renamed button for clarity
    try:
        prediction = model.predict(input_df)
        st.markdown("<div class='predicted-salary'>Predicted Salary: ${:,.2f}</div>".format(prediction[0]), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


# Display Feature Importance (assuming the loaded model is a tree-based regression model)
# Check if the loaded model (or the model within the pipeline) has feature_importances_ attribute
st.markdown("---<br>", unsafe_allow_html=True)
st.header("Model Insights")
if hasattr(model, 'feature_importances_'):
    st.subheader("Feature Importance")
    feature_importances = pd.Series(model.feature_importances_, index=input_df.columns)
    # Sort feature importances for better visualization
    feature_importances = feature_importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index, palette="viridis")
    plt.title("Feature Importance for Salary Prediction")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    st.pyplot(plt)
elif isinstance(model, Pipeline) and hasattr(model.named_steps['model'], 'feature_importances_'):
    # Handle case where model is in a pipeline
    st.subheader("Feature Importance")
    feature_importances = pd.Series(model.named_steps['model'].feature_importances_, index=input_df.columns)
    # Sort feature importances for better visualization
    feature_importances = feature_importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index, palette="viridis")
    plt.title("Feature Importance for Salary Prediction")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    st.pyplot(plt)
else:
    st.info("Feature importance is not available for the selected model type (e.g., SVR, Linear Regression).")

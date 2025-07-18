%%writefile /content/app.py
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the current directory (for reference, not displayed)
current_dir = os.getcwd()
# Removed debugging print statements
# st.write(f"Current working directory: {current_dir}")
# st.write(f"Files in current directory: {os.listdir(current_dir)}")

# Model file path - assuming best_model.pkl is in the same directory as app.py
model_path = os.path.join(current_dir, "best_model.pkl")

# Load the best model
try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Model file not found at {model_path}. Please ensure 'best_model.pkl' exists in the same directory as app.py.")
    st.stop() # Stop the app if the model file is not found
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


st.markdown("<h1 style='text-align: center; color: #1E90FF;'>Income Prediction App</h1>", unsafe_allow_html=True)

st.markdown("---") # Add a horizontal rule

st.write("Enter the following information to predict income:")

# Create input fields for each feature
# Move some inputs to the sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #1E90FF;'>Personal Information</h2>", unsafe_allow_html=True) # Styled header
    age = st.slider("Age", min_value=17, max_value=90, value=30) # Changed to slider for better user experience
    gender = st.selectbox("Gender", ['Male', 'Female'])
    race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    native_country = st.selectbox("Native Country", ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', 'Thailand', 'Guatemala', 'Nicaragua', 'Scotland', 'Columbia', 'Laos', 'Taiwan', 'Haiti', 'Hungary', 'atemala-Total', 'Cuba-Total', 'England-Total', 'Germany-Total', 'Greece-Total', 'India-Total', 'Iran-Total', 'Ireland-Total', 'Italy-Total', 'Jamaica-Total', 'Japan-Total', 'Laos-Total', 'Mexico-Total', 'Nicaragua-Total', 'Peru-Total', 'Philippines-Total', 'Poland-Total', 'Puerto-Rico-Total', 'Scotland-Total', 'South-Total', 'Taiwan-Total', 'Thailand-Total', 'Trinadad&Tobago-Total', 'United-States-Total', 'Vietnam-Total', 'Yugoslavia', '?'])


encoders = {}
for col, categories in zip(categorical_cols, [workclass_categories, marital_status_categories, occupation_categories, relationship_categories, race_categories, gender_categories, native_country_categories]):
    encoder = LabelEncoder()
    encoder.fit(categories)
    input_df[col] = encoder.transform(input_df[col])


# Make prediction
if st.button("Predict Income"):
    try:
        prediction = model.predict(input_df)
        st.subheader("Prediction:")
        if prediction[0] == '<=50K':
            st.success("The predicted income is <=50K") # Use st.success for positive outcome
        else:
            st.warning("The predicted income is >50K") # Use st.warning for potentially higher income
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Display Feature Importance (assuming the best model is GradientBoostingClassifier)
if hasattr(model, 'feature_importances_'):
    st.subheader("Feature Importance")
    feature_importances = pd.Series(model.feature_importances_, index=input_df.columns)
    # Sort feature importances for better visualization
    feature_importances = feature_importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index, palette='viridis') # Changed color palette
    plt.title("Feature Importance for Income Prediction")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    st.pyplot(plt)
else:
    st.info("Feature importance is not available for the selected model.")

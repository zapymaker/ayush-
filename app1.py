
import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest model and encoders
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_salary_model.joblib')
    gender_encoder = joblib.load('gender_encoder.joblib')
    education_encoder = joblib.load('education_encoder.joblib')
    job_encoder = joblib.load('job_encoder.joblib')
    return model, gender_encoder, education_encoder, job_encoder

rf_model, gender_encoder, education_encoder, job_encoder = load_model()

st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# User input fields
years_experience = st.slider('Years of Experience', 0, 30, 5)
gender = st.selectbox('Gender', gender_encoder.classes_)
education = st.selectbox('Education Level', education_encoder.classes_)
job_title = st.selectbox('Job Title', job_encoder.classes_)

if st.button('Predict Salary'):
    try:
        # Encode categorical features
        gender_encoded = gender_encoder.transform([gender])[0]
        education_encoded = education_encoder.transform([education])[0]
        job_title_encoded = job_encoder.transform([job_title])[0]

        # Create DataFrame for prediction
        input_data = pd.DataFrame([[years_experience, gender_encoded, education_encoded, job_title_encoded]],
                                  columns=['YearsExperience', 'Gender_encoded', 'Education_encoded', 'JobTitle_encoded'])

        # Make prediction
        predicted_salary = rf_model.predict(input_data)[0]

        st.success(f'Predicted Salary: ${predicted_salary:,.2f}')
    except ValueError as e:
        st.error(f"Error in prediction: {e}. Please ensure all fields are correctly selected and encoders are loaded properly.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

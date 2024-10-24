import numpy as np
import pandas as pd
import pickle as pkl 
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Tải mô hình và scaler đã lưu
forest = pkl.load(open('MIPML.pkl', 'rb'))
scaler = pkl.load(open('scaler.pkl', 'rb'))  # Giả sử bạn đã lưu scaler vào file này

# Streamlit UI setup
st.header('Medical Insurance Premium Predictor')

# Input fields for the user
gender = st.selectbox('Choose Gender', ['Female', 'Male'])
smoker = st.selectbox('Are you a smoker?', ['Yes', 'No'])
region = st.selectbox('Choose Region', ['SouthEast', 'SouthWest', 'NorthEast', 'NorthWest'])
age = st.slider('Enter Age', 5, 80)
bmi = st.number_input('Enter BMI', min_value=5.0, max_value=100.0, value=25.0, step=0.1)
children = st.slider('Choose No of Children', 0, 5)

# Logic to handle predictions when the button is clicked
if st.button('Predict'):
    # Convert categorical inputs into numerical format
    gender = 0 if gender == 'Female' else 1
    smoker = 1 if smoker == 'Yes' else 0

    # Encode region as one-hot encoding
    region_encoded = pd.get_dummies(pd.Series([region]), drop_first=True)
    SouthEast = region_encoded[0] if 'SouthEast' in region_encoded else 0
    SouthWest = region_encoded[1] if 'SouthWest' in region_encoded else 0
    NorthEast = region_encoded[2] if 'NorthEast' in region_encoded else 0
    NorthWest = region_encoded[3] if 'NorthWest' in region_encoded else 0

    # Prepare input for the model
    input_data = (age, gender, bmi, children, smoker, SouthEast, SouthWest, NorthEast, NorthWest)
    input_data_array = np.asarray(input_data).reshape(1, -1)

    # Chuẩn hóa dữ liệu
    input_data_array = scaler.transform(input_data_array)

    # Dự đoán
    predicted_prem_log = forest.predict(input_data_array)
    insurance_premium = np.exp(predicted_prem_log) - 1

    display_string = 'Insurance Premium will be ' + str(round(insurance_premium[0], 2)) + ' USD Dollars'
    st.markdown(display_string)

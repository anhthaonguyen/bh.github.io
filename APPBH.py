#importing Necessary Libraries
import numpy as np
import pandas as pd
import pickle as pkl 
import streamlit as st

# Load the pre-trained model from the pickle file
model = pkl.load(open('MIPML.PKL', 'rb'))

# Streamlit UI setup
st.header('Medical Insurance Premium Predictor')

# Input fields for the user
gender = st.selectbox('Choose Gender', ['Female', 'Male'])
smoker = st.selectbox('Are you a smoker?', ['Yes', 'No'])
region = st.selectbox('Choose Region', ['SouthEast', 'SouthWest', 'NorthEast', 'NorthWest'])
age = st.slider('Enter Age', 5, 80)
bmi = st.slider('Enter BMI', 5, 100)
children = st.slider('Choose No of Children', 0, 5)

# Logic to handle predictions when the button is clicked
if st.button('Predict'):
    # Convert categorical inputs into numerical format
    gender = 0 if gender == 'Female' else 1
    smoker = 1 if smoker == 'Yes' else 0

    # Encode region as one-hot encoding
    if region == 'SouthEast':
        SouthEast, SouthWest, NorthEast, NorthWest = 1, 0, 0, 0
    elif region == 'SouthWest':
        SouthEast, SouthWest, NorthEast, NorthWest = 0, 1, 0, 0
    elif region == 'NorthEast':
        SouthEast, SouthWest, NorthEast, NorthWest = 0, 0, 1, 0
    else:  # NorthWest
        SouthEast, SouthWest, NorthEast, NorthWest = 0, 0, 0, 1

    # Prepare input for the model
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    input_data = (age, gender, bmi, children, smoker, SouthEast, SouthWest, NorthEast, NorthWest)
    input_data_array = np.asarray(input_data).reshape(1, -1)

    input_data_scaled = scaler.transform(input_data_array)

    # Make prediction
    insurance_premium_log = model.predict(input_data_scaled)
    
    insurance_premium = np.exp(insurance_premium_log) - 1

    # Display the result
    display_string = 'Insurance Premium will be ' + str(insurance_premium[0]) + ' USD'
    st.markdown(display_string)

import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Streamlit UI components
st.title('Car Price Prediction')

# File uploader to allow the user to upload their model.pkl file
uploaded_model = st.file_uploader("Upload your trained model (model.pkl)", type="pkl")

# Check if the user uploaded the model
if uploaded_model is not None:
    # Load the uploaded model
    model = joblib.load(uploaded_model)

    # LabelEncoder for encoding categorical variables
    label_encoder = LabelEncoder()

    # Collect input features from the user
    company = st.selectbox('Car Company', ['Company A', 'Company B', 'Company C']) 
    year = st.number_input('Car Year', min_value=1990, max_value=2025, value=2015)
    present_price = st.number_input('Present Price (in INR)', min_value=0.0, value=500000)
    kms_driven = st.number_input('Kilometers Driven', min_value=0, value=50000)
    fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel'])
    seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer'])
    transmission = st.selectbox('Transmission Type', ['Manual', 'Automatic'])
    owner = st.number_input('Number of Owners', min_value=1, max_value=5, value=1)

    # Prepare the input data for prediction 
    company_encoded = label_encoder.transform([company])[0]
    fuel_type_encoded = label_encoder.transform([fuel_type])[0]
    seller_type_encoded = label_encoder.transform([seller_type])[0]
    transmission_encoded = label_encoder.transform([transmission])[0]

    input_data = np.array([[company_encoded, year, present_price, kms_driven, fuel_type_encoded, seller_type_encoded, transmission_encoded, owner]])

    # Make prediction when the user presses the button
    if st.button('Predict'):
        prediction = model.predict(input_data)
        st.write(f'The predicted selling price of the car is: ₹{prediction[0]:,.2f}')
else:
    st.write("Please upload your trained model (model.pkl) to make predictions.")

import streamlit as st
import numpy as np
import joblib

# function to make crop predictions using the loaded model
def predict_crop(model,input_data):
    try:
        prediction=model.predict(input_data)
        return prediction[0]  
    except Exception as e:
        return str(e)

def main():
    # page title and icon
    st.set_page_config(page_title="Crop Prediction App", page_icon="🌾")

    # loading the saved model
    model=joblib.load('gnnb_model.joblib')

    # app title and description
    st.title("Crop Prediction App 🌱")
    st.write("Enter the following parameters to get a crop prediction:")

    # input data for parameters
    N = st.number_input("Nitrogen (N)", 1, 10000)
    P = st.number_input("Phosphorus (P)", 1, 10000)
    K = st.number_input("Potassium (K)", 1, 10000)
    temp = st.number_input("Temperature (°C)", 0.0, 100000.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0)
    ph = st.number_input("pH", 0.0, 14.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 100000.0)

    # features list from input data
    input_data=np.array([[N, P, K, temp, humidity, ph, rainfall]])

    # predict button
    if st.button('Predict Crop'):
        try:
            # crop prediction using the loaded model
            prediction=predict_crop(model, input_data)

            # display the recommended crop
            st.success(f"The recommended crop for your farm is: {prediction}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # info section
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a machine learning model to recommend crops based on input parameters."
        " It's for educational purpose only and should not be relied upon for real-world decisions."
    )

if __name__=='__main__':
    main()
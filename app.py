#Import libraries
import streamlit as st
import pandas as pd
import joblib
from datetime import date


# Load model, scaler, and feature names
model = joblib.load("rf.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.title("Paris Housing Price Predictor")

st.markdown("""
Please enter the following details to predict the house price.
""")

# Inputs
squaremeters = st.number_input("What is the size of the house(m²)?", min_value=10, max_value=100000, value=100)
rooms = st.number_input("How many rooms are there?", min_value=1, max_value=100, value=5)
floors = st.number_input("How many floors are there?", min_value=1, max_value=100, value=1)
basement = st.number_input("What is the size of the basement(m²)?", min_value=0, max_value=10000, value=20)
attic = st.number_input("What is the size of the attic(m²)?", min_value=0, max_value=10000, value=20)
garage = st.number_input("What is the size of the garage(m²)?", min_value=0, max_value=1000, value=20)
citycode = st.number_input("What is the city code?", min_value=1000, max_value=9999, value=5000)
citypartrange = st.slider("What is the city part range?", min_value=1, max_value=10, value=5)
made = st.number_input("Which year was the house built?", min_value=1900, max_value=date.today().year, value=2000)
guest_room = st.checkbox("Is there a guest room?", value=False)
storm_protector = st.checkbox("Is there a storm protector?", value=False)
storage_room = st.checkbox("Is there a storage room?", value=False)
has_pool = st.checkbox("Is there a pool?", value=False)
has_yard = st.checkbox("Is there a yard?", value=False)
prev_owners = st.number_input("How many people have previously owned the house?", min_value=0, max_value=20, value=1)


# features used in the model
#numberOfRooms, hasYard,hasPool,cityCode,cityPartRange,numPrevOwners,hasStormProtector,hasStorageRoom,
#hasGuestRoom,age,floor_size,basement_size_ratio,attic_size_ratio',garage_size_ratio

if st.button("Predict Price"):
    input_dict = {
        "numberOfRooms": rooms,
        "floor_size": squaremeters / floors,
        "basement_size_ratio": basement / squaremeters,
        "attic_size_ratio": attic / squaremeters,
        "garage_size_ratio": garage / squaremeters,
        "cityCode": citycode,
        "cityPartRange": citypartrange,
        "age": date.today().year - made,
        "hasGuestRoom": int(guest_room),
        "hasStormProtector": int(storm_protector),
        "hasStorageRoom": int(storage_room),
        "hasPool": int(has_pool),
        "hasYard": int(has_yard),
        "numPrevOwners": prev_owners
    }

    # Scaling and prediction
    df_input = pd.DataFrame([input_dict])[features]
    df_scaled = scaler.transform(df_input)
    prediction = model.predict(df_scaled)[0]

    st.success(f"Estimated House Price: €{prediction:,.2f}")





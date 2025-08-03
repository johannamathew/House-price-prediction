
import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and feature names
model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("Paris Housing Price Predictor (Linear Model)")

st.markdown("""
Enter the details of the house below to predict its selling price.
""")

# Input
square_meters = st.number_input("Square Meters", min_value=10, max_value=10000, value=100)
num_rooms = st.number_input("Number of Rooms", min_value=1, max_value=20, value=3)
floors = st.number_input("Floors", min_value=1, max_value=5, value=1)
basement = st.number_input("Basement (m²)", min_value=0, max_value=200, value=20)
attic = st.number_input("Attic (m²)", min_value=0, max_value=200, value=20)
garage = st.number_input("Garage (m²)", min_value=0, max_value=100, value=15)
city_code = st.number_input("City Code", min_value=1000, max_value=9999, value=7500)
city_part = st.slider("City Part Range", min_value=1, max_value=10, value=5)
house_age = st.number_input("House Age (years)", min_value=0, max_value=200, value=10)

# Integer features
guest_room = st.checkbox("Has Guest Room", value=False)
storm_protector = st.checkbox("Has Storm Protector", value=False)
storage_room = st.checkbox("Has Storage Room", value=False)
is_new = st.checkbox("Is Newly Built", value=False)
has_pool = st.checkbox("Has Pool", value=False)
has_yard = st.checkbox("Has Yard", value=False)
num_prev_owners = st.number_input("Number of Previous Owners", min_value=0, max_value=20, value=1)

# Predict button
if st.button("Predict Price"):
    input_dict = {
        "squareMeters": square_meters,
        "numberOfRooms": num_rooms,
        "floors": floors,
        "basement": basement,
        "attic": attic,
        "garage": garage,
        "cityCode": city_code,
        "cityPartRange": city_part,
        "houseAge": house_age,
        "hasGuestRoom": int(guest_room),
        "hasStormProtector": int(storm_protector),
        "hasStorageRoom": int(storage_room),
        "isNewBuilt": int(is_new),
        "hasPool": int(has_pool),
        "hasYard": int(has_yard),
        "numPrevOwners": num_prev_owners,
        "made": 2025 - house_age
    }

    # correct order
    input_df = pd.DataFrame([input_dict])[feature_names]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.success(f"Estimated House Price: €{prediction:,.2f}")

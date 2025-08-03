{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 AppleColorEmoji;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import pandas as pd\
import joblib\
\
# Load model and scaler\
model = joblib.load("rf_model.pkl")\
scaler = joblib.load("scaler.pkl")\
\
st.title("
\f1 \uc0\u55356 \u57312 
\f0  Paris Housing Price Predictor")\
\
st.markdown("""\
Enter the details of the house below to predict its selling price.\
""")\
\
# Input fields\
square_meters = st.number_input("Square Meters", min_value=10, max_value=1000, value=100)\
num_rooms = st.number_input("Number of Rooms", min_value=1, max_value=20, value=3)\
floors = st.number_input("Floors", min_value=1, max_value=5, value=1)\
basement = st.number_input("Basement (m\'b2)", min_value=0, max_value=200, value=20)\
attic = st.number_input("Attic (m\'b2)", min_value=0, max_value=200, value=20)\
garage = st.number_input("Garage (m\'b2)", min_value=0, max_value=100, value=15)\
city_code = st.number_input("City Code", min_value=1000, max_value=9999, value=7500)\
city_part = st.slider("City Part Range", min_value=1, max_value=10, value=5)\
house_age = st.number_input("House Age (years)", min_value=0, max_value=200, value=10)\
\
# Boolean features\
guest_room = st.checkbox("Has Guest Room", value=False)\
storm_protector = st.checkbox("Has Storm Protector", value=False)\
storage_room = st.checkbox("Has Storage Room", value=False)\
is_new = st.checkbox("Is Newly Built", value=False)\
\
# Predict button\
if st.button("Predict Price"):\
    input_data = pd.DataFrame([\{\
        "squareMeters": square_meters,\
        "numberOfRooms": num_rooms,\
        "floors": floors,\
        "basement": basement,\
        "attic": attic,\
        "garage": garage,\
        "cityCode": city_code,\
        "cityPartRange": city_part,\
        "houseAge": house_age,\
        "hasGuestRoom": int(guest_room),\
        "hasStormProtector": int(storm_protector),\
        "hasStorageRoom": int(storage_room),\
        "isNewBuilt": int(is_new),\
        "made": 2025 - house_age  # reverse engineer year built\
    \}])\
\
    # Reorder and scale\
    input_scaled = scaler.transform(input_data)\
    prediction = model.predict(input_scaled)[0]\
\
    st.success(f"
\f1 \uc0\u55356 \u57335 \u65039 
\f0  Estimated House Price: \'80\{prediction:,.2f\}")\
}
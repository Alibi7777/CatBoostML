import streamlit as st
import requests

st.set_page_config(page_title="Apartment Price Category Predictor")
st.title("🏢 Apartment Price Category Predictor")

# Input fields
number_of_rooms = st.number_input("Number of Rooms", min_value=1, max_value=10)
district = st.selectbox("District", list(range(1, 11)))
structure_type = st.selectbox("Structure Type", list(range(1, 6)))
year_of_construction = st.slider("Year of Construction", 1960, 2025)
floor = st.number_input("Floor", min_value=1)
area = st.number_input("Area (m²)", min_value=10.0)
quality = st.selectbox("Quality", list(range(1, 6)))

# Format helper
def format_price(price):
    millions = int(price) // 1_000_000
    remain = (int(price) % 1_000_000) 
    return f"{millions} million {remain}  tenge"

# Prediction button logic
if st.button("Predict"):
    payload = {
        "number_of_rooms": number_of_rooms,
        "district": district,
        "structure_type": structure_type,
        "year_of_construction": year_of_construction,
        "floor": floor,
        "area": area,
        "quality": quality,
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            if "predicted_price" in result:
                formatted = format_price(result['predicted_price'])
                st.success(f"💰 Predicted Price: {formatted}")
            else:
                st.warning("⚠️ No 'predicted_price' key in the response.")
                st.json(result)
        else:
            st.error("❌ Prediction request failed.")
    except Exception as e:
        st.error(f"🚫 An unexpected error occurred: {e}")

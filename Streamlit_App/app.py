"""
Indian House Price Predictor — Streamlit App
Run: python -m streamlit run app.py
(Keep this file in the SAME folder as the .pkl files)
"""

import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Indian House Price Predictor", page_icon="🏠", layout="centered")

BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model       = joblib.load(os.path.join(BASE, "house_price_model.pkl"))
    scaler      = joblib.load(os.path.join(BASE, "scaler.pkl"))
    le_city     = joblib.load(os.path.join(BASE, "le_city.pkl"))
    le_locality = joblib.load(os.path.join(BASE, "le_locality.pkl"))
    le_furnish  = joblib.load(os.path.join(BASE, "le_furnish.pkl"))
    return model, scaler, le_city, le_locality, le_furnish

model, scaler, le_city, le_locality, le_furnish = load_artifacts()

def format_inr(amount):
    if amount >= 1_00_00_000:
        return f"₹{amount/1_00_00_000:.2f} Cr"
    elif amount >= 1_00_000:
        return f"₹{amount/1_00_000:.2f} Lakh"
    else:
        return f"₹{amount:,.0f}"

st.markdown("""
<h1 style='text-align:center; color:#0B3D2E;'>House Price Predictor Model</h1>
<p style='text-align:center; color:#555; font-size:15px;'>
AI-powered property valuation across major Indian cities
</p><hr/>
""", unsafe_allow_html=True)

st.subheader("📋 Enter Property Details")
col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("🏙️ City", [
        "Mumbai", "Delhi NCR", "Bangalore", "Pune", "Hyderabad",
        "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"
    ])
    locality = st.selectbox("📍 Locality Type", [
        "Prime/CBD", "Premium Residential", "Mid-tier",
        "Developing Suburb", "Outskirts"
    ])
    area = st.number_input("📐 Area (sq ft)", min_value=300, max_value=4000, value=1100, step=50)
    bhk  = st.slider("🛏️ BHK", 1, 5, 2)
    bathrooms = st.slider("🚿 Bathrooms", 1, 4, 2)

with col2:
    age      = st.slider("🏗️ Age of Property (years)", 0, 40, 5)
    floor_no = st.slider("🏢 Floor Number", 0, 25, 4)
    parking  = st.selectbox("🚗 Parking Slots", [0, 1, 2])
    furnishing = st.selectbox("🛋️ Furnishing", ["Unfurnished", "Semi-Furnished", "Fully-Furnished"])
    clubhouse = st.radio("🏋️ Clubhouse/Gym/Amenities", ["No", "Yes"], horizontal=True)
    gated = st.radio("🚪 Gated Community", ["No", "Yes"], horizontal=True)

st.markdown("<hr/>", unsafe_allow_html=True)

if st.button("🔮 Predict Price", use_container_width=True, type="primary"):
    clubhouse_val = 1 if clubhouse == "Yes" else 0
    gated_val     = 1 if gated == "Yes" else 0

    city_enc     = le_city.transform([city])[0]
    locality_enc = le_locality.transform([locality])[0]
    furnish_enc  = le_furnish.transform([furnishing])[0]

    total_rooms   = bhk + bathrooms
    amenity_score = parking + clubhouse_val * 2 + gated_val * 2
    is_new        = 1 if age <= 5 else 0

    features = np.array([[area, bhk, bathrooms, age, floor_no, parking,
                          city_enc, locality_enc, furnish_enc, clubhouse_val,
                          gated_val, total_rooms, amenity_score, is_new]])
    price = model.predict(scaler.transform(features))[0]

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0B3D2E,#1B6B45);
                border-radius:16px;padding:30px;text-align:center;margin:20px 0;'>
        <p style='color:#BFEBD8;font-size:16px;margin:0;'>Estimated Market Value</p>
        <h1 style='color:white;font-size:48px;margin:10px 0;'>{format_inr(price)}</h1>
        <p style='color:#BFEBD8;font-size:13px;margin:0;'>₹{price:,.0f}</p>
        <p style='color:#BFEBD8;font-size:12px;margin-top:8px;'>
            Model: Random Forest Regressor &nbsp;|&nbsp; R² Score: 0.867
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("City",        city)
    c2.metric("Area",        f"{area:,} sqft")
    c3.metric("Config",      f"{bhk} BHK")
    c4.metric("₹ / sqft",    f"₹{price/area:,.0f}")

    st.subheader("💰 Price in Different Units")
    b1, b2, b3 = st.columns(3)
    b1.metric("Lakhs", f"₹{price/100000:.1f} L")
    b2.metric("Crores", f"₹{price/10000000:.3f} Cr")
    b3.metric("Exact (₹)", f"₹{price:,.0f}")

    with st.expander("🔍 View Engineered Features Used"):
        st.write({
            "Total Rooms": total_rooms,
            "Amenity Score": amenity_score,
            "Is New (≤5 yrs)": bool(is_new),
            "City Encoded": int(city_enc),
            "Locality Encoded": int(locality_enc),
        })

st.markdown("""
<hr/>
<p style='text-align:center;color:#999;font-size:12px;'>
Indian House Price Prediction System &nbsp;|&nbsp; ML Project &nbsp;|&nbsp;
Cities: Mumbai · Delhi NCR · Bangalore · Pune · Hyderabad · Chennai · Kolkata · Ahmedabad · Jaipur · Lucknow
</p>
""", unsafe_allow_html=True)

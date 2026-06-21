HOUSE_PREDICTION_MODEL/
├── Dataset/
│   └── india_house_prices.csv        # 1,200-row dataset (Indian housing market)
├── Notebook/
│   ├── house_prediction_model.py     # Full ML pipeline (all 8 phases)
│   ├── eda_univariate.png
│   ├── eda_bivariate.png
│   ├── eda_boxplots.png
│   ├── eda_heatmap.png
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── model_results.csv
├── Model/
│   ├── house_price_model.pkl         # Trained Random Forest model
│   ├── scaler.pkl                    # StandardScaler
│   ├── le_city.pkl                   # City label encoder
│   ├── le_locality.pkl               # Locality type label encoder
│   └── le_furnish.pkl                # Furnishing label encoder
├── Streamlit_App/
│   ├── app.py                        # Deployed web application
│   └── *.pkl                         # Model files (copied here for easy local run)
├── Documentation/
│   ├── House_Prediction_Model_Report.docx
│   └── House_Prediction_Model_Presentation.pptx
└── README.md
```

## ⚙️ Setup & Run

### 1. Install Dependencies
```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib streamlit
```

### 2. (Optional) Retrain Models
The trained `.pkl` files are already included in `/Model`. To regenerate them from scratch:
```bash
cd Notebook
python house_prediction_model.py
```

### 3. Launch the Web App
```bash
cd Streamlit_App
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

> **Note:** The `.pkl` model files are already included inside `Streamlit_App/`, so the app runs immediately without retraining.

## 🌆 Cities Covered
Mumbai · Delhi NCR · Bangalore · Pune · Hyderabad · Chennai · Kolkata · Ahmedabad · Jaipur · Lucknow

## 🧠 ML Models Trained & Compared
| Model | R² Score | RMSE (₹) | MAE (₹) |
|-------|----------|----------|---------|
| Linear Regression | 0.4796 | 72,80,776 | 57,49,728 |
| Random Forest **(Deployed)** | 0.8667 | 36,85,209 | 27,53,281 |
| Gradient Boosting | 0.9340 | 25,92,931 | 18,07,460 |

> **Note:** Random Forest was selected for deployment over Gradient Boosting for better cross-version compatibility (joblib/scikit-learn portability), at a modest accuracy trade-off.

## 📊 Features Used
- **Location:** City, Locality Type (Prime/CBD, Premium Residential, Mid-tier, Developing Suburb, Outskirts)
- **Property:** Area (sq ft), BHK, Bathrooms, Age, Floor Number
- **Amenities:** Parking, Furnishing, Clubhouse, Gated Community
- **Engineered:** Total Rooms, Amenity Score, Is New (≤5 yrs)

## 📈 ML Lifecycle Followed
1. Problem Understanding → 2. Data Collection → 3. Data Preprocessing →
4. Exploratory Data Analysis → 5. Feature Engineering → 6. Model Building →
7. Model Evaluation → 8. Model Deployment (Streamlit)

## ⚠️ Data Disclaimer
The dataset is **synthetically generated** to realistically simulate Indian residential property pricing patterns (city-tier multipliers, locality premiums, amenity effects) for academic demonstration. It is not sourced from a live real-estate listing platform.
"# HOUSE_PRICE_PRIDICTION_MODEL" 

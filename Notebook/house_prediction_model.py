"""
HOUSE PREDICTION MODEL — India Edition
Complete ML Pipeline: Data Generation -> Preprocessing -> EDA -> Feature Engineering
                       -> Model Building -> Evaluation -> Model Saving
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')

OUT = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# PHASE 1 & 2: Problem Understanding & Data Collection
# ─────────────────────────────────────────────
print("=" * 60)
print("PHASE 1 & 2: HOUSE PREDICTION MODEL - Dataset Creation")
print("=" * 60)

N = 1200
cities = {
    'Mumbai': 2.6, 'Delhi NCR': 2.0, 'Bangalore': 1.8, 'Pune': 1.4,
    'Hyderabad': 1.3, 'Chennai': 1.3, 'Kolkata': 1.0, 'Ahmedabad': 0.9,
    'Jaipur': 0.75, 'Lucknow': 0.65,
}
city_list = list(cities.keys())
city_weights = [0.15, 0.15, 0.13, 0.10, 0.09, 0.09, 0.09, 0.08, 0.07, 0.05]
city = np.random.choice(city_list, N, p=city_weights)

locality_types = ['Prime/CBD', 'Premium Residential', 'Mid-tier', 'Developing Suburb', 'Outskirts']
locality_mult = {'Prime/CBD': 1.6, 'Premium Residential': 1.35, 'Mid-tier': 1.1,
                  'Developing Suburb': 0.9, 'Outskirts': 0.7}
locality = np.random.choice(locality_types, N, p=[0.12, 0.2, 0.33, 0.22, 0.13])

area_sqft = np.random.randint(400, 3500, N)
bhk       = np.random.randint(1, 6, N)
bathrooms = np.clip(bhk - np.random.randint(0, 2, N), 1, None)
age       = np.random.randint(0, 40, N)
floor_no  = np.random.randint(0, 25, N)
parking   = np.random.randint(0, 3, N)
furnished = np.random.choice(['Unfurnished', 'Semi-Furnished', 'Fully-Furnished'], N, p=[0.4, 0.4, 0.2])
furnish_mult = {'Unfurnished': 1.0, 'Semi-Furnished': 1.07, 'Fully-Furnished': 1.15}
clubhouse = np.random.choice([0, 1], N, p=[0.5, 0.5])
gated     = np.random.choice([0, 1], N, p=[0.35, 0.65])

base_rate = 4500
price_per_sqft = np.array([
    base_rate * cities[city[i]] * locality_mult[locality[i]] * furnish_mult[furnished[i]]
    for i in range(N)
])
price_per_sqft *= (1 - age * 0.006)
price_per_sqft *= (1 + floor_no * 0.0015)
price_per_sqft *= (1 + clubhouse * 0.05)
price_per_sqft *= (1 + gated * 0.04)
price_per_sqft *= (1 + parking * 0.015)
price_per_sqft *= (1 + (bhk - 2) * 0.02)
price_per_sqft *= np.random.normal(1, 0.08, N)

price = np.clip(price_per_sqft * area_sqft, 800_000, 120_000_000).astype(int)

df = pd.DataFrame({
    'City': city, 'Locality_Type': locality, 'Area_sqft': area_sqft, 'BHK': bhk,
    'Bathrooms': bathrooms, 'Age_years': age, 'Floor_No': floor_no, 'Parking': parking,
    'Furnishing': furnished, 'Clubhouse_Amenities': clubhouse, 'Gated_Community': gated,
    'Price_INR': price,
})

miss_age = np.random.choice(N, 45, replace=False)
miss_bath = np.random.choice(N, 30, replace=False)
df.loc[miss_age, 'Age_years'] = np.nan
df.loc[miss_bath, 'Bathrooms'] = np.nan

df.to_csv(os.path.join(OUT, '..', 'Dataset', 'india_house_prices.csv'), index=False)
print(f"Dataset saved: {df.shape[0]} rows x {df.shape[1]} cols")
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# ─────────────────────────────────────────────
# PHASE 3: Data Preprocessing
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3: Data Preprocessing")
print("=" * 60)

df['Age_years'] = df['Age_years'].fillna(df['Age_years'].median())
df['Bathrooms'] = df['Bathrooms'].fillna(df['Bathrooms'].mode()[0])
df.drop_duplicates(inplace=True)
print(f"After cleaning: {df.shape[0]} rows, missing: {df.isnull().sum().sum()}")

Q1, Q3 = df['Price_INR'].quantile(0.25), df['Price_INR'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Price_INR'] >= Q1 - 1.5*IQR) & (df['Price_INR'] <= Q3 + 1.5*IQR)].copy()
print(f"After outlier removal: {df.shape[0]} rows")

le_city = LabelEncoder()
le_locality = LabelEncoder()
le_furnish = LabelEncoder()
df['City_enc'] = le_city.fit_transform(df['City'])
df['Locality_enc'] = le_locality.fit_transform(df['Locality_Type'])
df['Furnish_enc'] = le_furnish.fit_transform(df['Furnishing'])

# ─────────────────────────────────────────────
# PHASE 4: EDA Visualizations
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 4: Exploratory Data Analysis")
print("=" * 60)

NB = os.path.join(OUT, '..', 'Notebook')

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Univariate Analysis - HOUSE PREDICTION MODEL', fontsize=16, fontweight='bold')
axes[0, 0].hist(df['Price_INR']/1e7, bins=30, color='#0B3D2E', edgecolor='white')
axes[0, 0].set_title('Price Distribution (Cr Rs)'); axes[0, 0].set_xlabel('Price (Cr)')
axes[0, 1].hist(df['Area_sqft'], bins=30, color='#1B6B45', edgecolor='white')
axes[0, 1].set_title('Area Distribution'); axes[0, 1].set_xlabel('Area (sq ft)')
axes[0, 2].hist(df['Age_years'], bins=20, color='#D97706', edgecolor='white')
axes[0, 2].set_title('Property Age Distribution'); axes[0, 2].set_xlabel('Age (years)')
df['BHK'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0], color='#7C3AED')
axes[1, 0].set_title('BHK Configuration Count'); axes[1, 0].set_xlabel('BHK')
df['City'].value_counts().plot(kind='bar', ax=axes[1, 1], color='#0369A1')
axes[1, 1].set_title('City-wise Listing Count'); axes[1, 1].set_xlabel('City')
df['Furnishing'].value_counts().plot(kind='bar', ax=axes[1, 2], color='#BE123C')
axes[1, 2].set_title('Furnishing Type Count'); axes[1, 2].set_xlabel('Furnishing')
plt.tight_layout()
plt.savefig(os.path.join(NB, 'eda_univariate.png'), dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Bivariate Analysis', fontsize=16, fontweight='bold')
axes[0].scatter(df['Area_sqft'], df['Price_INR']/1e7, alpha=0.4, color='#0B3D2E')
axes[0].set_title('Area vs Price'); axes[0].set_xlabel('Area (sq ft)'); axes[0].set_ylabel('Price (Cr Rs)')
avg_by_city = df.groupby('City')['Price_INR'].mean().sort_values(ascending=False)/1e7
avg_by_city.plot(kind='bar', ax=axes[1], color='#1B6B45')
axes[1].set_title('Avg Price by City'); axes[1].set_ylabel('Price (Cr Rs)'); axes[1].set_xlabel('')
avg_by_bhk = df.groupby('BHK')['Price_INR'].mean()/1e7
avg_by_bhk.plot(kind='bar', ax=axes[2], color='#D97706')
axes[2].set_title('Avg Price by BHK'); axes[2].set_ylabel('Price (Cr Rs)')
plt.tight_layout()
plt.savefig(os.path.join(NB, 'eda_bivariate.png'), dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Boxplot Analysis', fontsize=16, fontweight='bold')
df.boxplot(column='Price_INR', by='City', ax=axes[0], rot=45)
axes[0].set_title('Price by City'); axes[0].set_ylabel('Price (Rs)')
df.boxplot(column='Price_INR', by='Locality_Type', ax=axes[1], rot=20)
axes[1].set_title('Price by Locality Type'); axes[1].set_ylabel('Price (Rs)')
plt.suptitle('')
plt.tight_layout()
plt.savefig(os.path.join(NB, 'eda_boxplots.png'), dpi=150, bbox_inches='tight')
plt.close()

numeric_cols = ['Area_sqft', 'BHK', 'Bathrooms', 'Age_years', 'Floor_No', 'Parking',
                 'Clubhouse_Amenities', 'Gated_Community', 'Price_INR']
corr = df[numeric_cols].corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='YlGnBu', center=0, linewidths=0.5, annot_kws={'size': 9})
plt.title('Feature Correlation Heatmap', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(NB, 'eda_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("EDA plots saved.")

# ─────────────────────────────────────────────
# PHASE 5: Feature Engineering
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 5: Feature Engineering")
print("=" * 60)

df['Total_rooms'] = df['BHK'] + df['Bathrooms']
df['Amenity_score'] = df['Parking'] + df['Clubhouse_Amenities']*2 + df['Gated_Community']*2
df['Is_new'] = (df['Age_years'] <= 5).astype(int)
df['Price_per_sqft'] = df['Price_INR'] / df['Area_sqft']

features = ['Area_sqft', 'BHK', 'Bathrooms', 'Age_years', 'Floor_No', 'Parking',
            'City_enc', 'Locality_enc', 'Furnish_enc', 'Clubhouse_Amenities',
            'Gated_Community', 'Total_rooms', 'Amenity_score', 'Is_new']
X = df[features]
y = df['Price_INR']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────
# PHASE 6 & 7: Model Building & Evaluation
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 6 & 7: Model Building & Evaluation")
print("=" * 60)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    cv = cross_val_score(model, X_scaled, y, cv=5, scoring='r2').mean()
    results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'CV_R2': cv}
    print(f"\n{name}:")
    print(f"  MAE:   Rs {mae:,.0f}")
    print(f"  RMSE:  Rs {rmse:,.0f}")
    print(f"  R2:    {r2:.4f}")
    print(f"  CV R2: {cv:.4f}")

results_df = pd.DataFrame(results).T
results_df.to_csv(os.path.join(NB, 'model_results.csv'))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
colors = ['#BE123C', '#0369A1', '#1B6B45']
names = list(results.keys())
for ax, metric, title in zip(axes, ['R2', 'RMSE', 'MAE'],
                              ['R2 Score (higher=better)', 'RMSE (lower=better)', 'MAE (lower=better)']):
    vals = [results[n][metric] for n in names]
    bars = ax.bar(names, vals, color=colors)
    ax.set_title(title); ax.set_xticklabels(names, rotation=15, ha='right')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.98,
                f'{val:.2f}' if metric == 'R2' else f'{val:,.0f}',
                ha='center', va='top', fontsize=9, color='white', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(NB, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

best_model = models['Random Forest']
y_pred_rf = best_model.predict(X_test)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Random Forest - Prediction Analysis', fontsize=14, fontweight='bold')
axes[0].scatter(y_test/1e7, y_pred_rf/1e7, alpha=0.4, color='#0369A1')
mn, mx = y_test.min()/1e7, y_test.max()/1e7
axes[0].plot([mn, mx], [mn, mx], 'r--', lw=2)
axes[0].set_xlabel('Actual Price (Cr Rs)'); axes[0].set_ylabel('Predicted Price (Cr Rs)')
axes[0].set_title('Actual vs Predicted')
residuals = y_test - y_pred_rf
axes[1].scatter(y_pred_rf/1e7, residuals/100000, alpha=0.4, color='#D97706')
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_xlabel('Predicted Price (Cr Rs)'); axes[1].set_ylabel('Residual (Lakh Rs)')
axes[1].set_title('Residual Plot')
plt.tight_layout()
plt.savefig(os.path.join(NB, 'actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
plt.close()

importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=True)
plt.figure(figsize=(8, 6))
importances.plot(kind='barh', color='#0B3D2E')
plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(NB, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\nAll plots saved.")

MODEL_DIR = os.path.join(OUT, '..', 'Model')
joblib.dump(best_model, os.path.join(MODEL_DIR, 'house_price_model.pkl'), protocol=4)
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'), protocol=4)
joblib.dump(le_city, os.path.join(MODEL_DIR, 'le_city.pkl'), protocol=4)
joblib.dump(le_locality, os.path.join(MODEL_DIR, 'le_locality.pkl'), protocol=4)
joblib.dump(le_furnish, os.path.join(MODEL_DIR, 'le_furnish.pkl'), protocol=4)
print("Models saved to /Model/")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE - HOUSE PREDICTION MODEL")
print("=" * 60)
print(results_df[['MAE', 'RMSE', 'R2', 'CV_R2']].round(4))

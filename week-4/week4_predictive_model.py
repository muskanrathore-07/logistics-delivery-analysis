# ============================================
# WEEK 4: PREDICTIVE MODELING & OPTIMIZATION
# Logistics Delivery Analysis
# ============================================

# ============================================
# 1. Import Libraries
# ============================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================
# 2. Load Cleaned Dataset
# ============================================

df = pd.read_csv("data/cleaned_delivery_logistics.csv")

print("Dataset Shape:", df.shape)


# ============================================
# 3. Remove Unnecessary / Constant Columns
# ============================================

df = df.drop(columns=[
    "delivery_id",
    "delivery_time_hours",
    "expected_time_hours"
])


# ============================================
# 4. Define Features and Target
# ============================================

X = df.drop(columns=["delivery_cost"])
y = df["delivery_cost"]


# ============================================
# 5. Identify Feature Types
# ============================================

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================
# 6. Create Preprocessing Pipeline
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ============================================
# 7. Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records:", X_train.shape[0])
print("Testing Records:", X_test.shape[0])


# ============================================
# 8. Apply Preprocessing
# ============================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Training Shape:",
      X_train_processed.shape)

print("Processed Testing Shape:",
      X_test_processed.shape)


# ============================================
# 9. Linear Regression Model
# ============================================

linear_model = LinearRegression()

linear_model.fit(
    X_train_processed,
    y_train
)


# ============================================
# 10. Linear Regression Predictions
# ============================================

y_pred_linear = linear_model.predict(
    X_test_processed
)


# ============================================
# 11. Evaluate Linear Regression
# ============================================

mae_linear = mean_absolute_error(
    y_test,
    y_pred_linear
)

mse_linear = mean_squared_error(
    y_test,
    y_pred_linear
)

rmse_linear = np.sqrt(mse_linear)

r2_linear = r2_score(
    y_test,
    y_pred_linear
)


print("\n============================================")
print("LINEAR REGRESSION RESULTS")
print("============================================")

print(f"MAE  : {mae_linear:.2f}")
print(f"MSE  : {mse_linear:.2f}")
print(f"RMSE : {rmse_linear:.2f}")
print(f"R²   : {r2_linear:.4f}")


# ============================================
# 12. Random Forest Regression Model
# ============================================

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(
    X_train_processed,
    y_train
)


# ============================================
# 13. Random Forest Predictions
# ============================================

y_pred_rf = random_forest_model.predict(
    X_test_processed
)


# ============================================
# 14. Evaluate Random Forest
# ============================================

mae_rf = mean_absolute_error(
    y_test,
    y_pred_rf
)

mse_rf = mean_squared_error(
    y_test,
    y_pred_rf
)

rmse_rf = np.sqrt(mse_rf)

r2_rf = r2_score(
    y_test,
    y_pred_rf
)


print("\n============================================")
print("RANDOM FOREST RESULTS")
print("============================================")

print(f"MAE  : {mae_rf:.2f}")
print(f"MSE  : {mse_rf:.2f}")
print(f"RMSE : {rmse_rf:.2f}")
print(f"R²   : {r2_rf:.4f}")


# ============================================
# 15. Model Comparison
# ============================================

print("\n============================================")
print("MODEL COMPARISON")
print("============================================")

print("\nLinear Regression:")
print(f"MAE  : {mae_linear:.2f}")
print(f"RMSE : {rmse_linear:.2f}")
print(f"R²   : {r2_linear:.4f}")

print("\nRandom Forest:")
print(f"MAE  : {mae_rf:.2f}")
print(f"RMSE : {rmse_rf:.2f}")
print(f"R²   : {r2_rf:.4f}")

# ============================================
# 16. Actual vs Predicted Visualization
# ============================================

import matplotlib.pyplot as plt


plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred_linear,
    alpha=0.5,
    label="Linear Regression"
)

plt.scatter(
    y_test,
    y_pred_rf,
    alpha=0.5,
    label="Random Forest"
)

# Perfect prediction reference line
min_value = min(y_test.min(), y_pred_linear.min(), y_pred_rf.min())
max_value = max(y_test.max(), y_pred_linear.max(), y_pred_rf.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Delivery Cost")
plt.ylabel("Predicted Delivery Cost")
plt.title("Actual vs Predicted Delivery Cost")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "week-4/actual_vs_predicted.png",
    dpi=300
)

plt.show()

# ============================================
# 17. Prediction Error Analysis
# ============================================

# Calculate prediction errors
linear_errors = y_test - y_pred_linear
rf_errors = y_test - y_pred_rf

# Absolute errors
linear_abs_errors = np.abs(linear_errors)
rf_abs_errors = np.abs(rf_errors)


# ============================================
# 18. Error Statistics
# ============================================

print("\n============================================")
print("PREDICTION ERROR ANALYSIS")
print("============================================")

print("\nLinear Regression:")
print(f"Mean Error          : {linear_errors.mean():.2f}")
print(f"Mean Absolute Error : {linear_abs_errors.mean():.2f}")
print(f"Maximum Absolute Error : {linear_abs_errors.max():.2f}")

print("\nRandom Forest:")
print(f"Mean Error          : {rf_errors.mean():.2f}")
print(f"Mean Absolute Error : {rf_abs_errors.mean():.2f}")
print(f"Maximum Absolute Error : {rf_abs_errors.max():.2f}")


# ============================================
# 19. Error Distribution Visualization
# ============================================

plt.figure(figsize=(10, 6))

plt.hist(
    linear_errors,
    bins=50,
    alpha=0.5,
    label="Linear Regression"
)

plt.hist(
    rf_errors,
    bins=50,
    alpha=0.5,
    label="Random Forest"
)

plt.axvline(
    0,
    linestyle="--",
    label="Zero Error"
)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "week-4/prediction_error_distribution.png",
    dpi=300
)

plt.show()

# ============================================
# 20. Random Forest Feature Importance
# ============================================

# Get feature names after one-hot encoding
encoded_feature_names = preprocessor.named_transformers_[
    "categorical"
].get_feature_names_out(categorical_features)

# Combine encoded categorical features with numerical features
feature_names = np.concatenate([
    encoded_feature_names,
    numerical_features
])

# Get feature importance values
importance_values = random_forest_model.feature_importances_

# Create feature importance DataFrame
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Display top 15 features
print("\n============================================")
print("TOP 15 FEATURE IMPORTANCE")
print("============================================")

print(
    feature_importance_df.head(15).to_string(index=False)
)


# ============================================
# 21. Feature Importance Visualization
# ============================================

top_features = feature_importance_df.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features Influencing Delivery Cost")

plt.tight_layout()

plt.savefig(
    "week-4/feature_importance.png",
    dpi=300
)

plt.show()

# ============================================
# 22. Grouped Feature Importance
# ============================================

grouped_importance = {}

for feature, importance in zip(
    feature_names,
    importance_values
):
    original_feature = None

    # Match encoded feature to its original categorical column
    for category in categorical_features:
        prefix = category + "_"

        if feature.startswith(prefix):
            original_feature = category
            break

    # Numerical feature
    if original_feature is None:
        original_feature = feature

    grouped_importance[original_feature] = (
        grouped_importance.get(original_feature, 0)
        + importance
    )

grouped_importance_df = pd.DataFrame(
    list(grouped_importance.items()),
    columns=["Feature", "Importance"]
)

grouped_importance_df = grouped_importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n============================================")
print("GROUPED FEATURE IMPORTANCE")
print("============================================")

print(
    grouped_importance_df.to_string(index=False)
)


# ============================================
# 23. Grouped Feature Importance Visualization
# ============================================

plt.figure(figsize=(10, 6))

plt.barh(
    grouped_importance_df["Feature"],
    grouped_importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance by Original Variable")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "week-4/grouped_feature_importance.png",
    dpi=300
)

plt.show()

# ============================================
# 24. Distance-Based Cost Analysis
# ============================================

# Create distance bands
df_analysis = pd.read_csv(
    "data/cleaned_delivery_logistics.csv"
)

df_analysis["distance_band"] = pd.cut(
    df_analysis["distance_km"],
    bins=[0, 50, 100, 150, 200, float("inf")],
    labels=[
        "0-50 km",
        "50-100 km",
        "100-150 km",
        "150-200 km",
        "200+ km"
    ],
    include_lowest=True
)

# Calculate cost per km
df_analysis["cost_per_km"] = (
    df_analysis["delivery_cost"]
    / df_analysis["distance_km"]
)

# Group by distance band
distance_analysis = (
    df_analysis
    .groupby("distance_band", observed=True)
    .agg(
        deliveries=("delivery_cost", "count"),
        average_distance_km=("distance_km", "mean"),
        average_delivery_cost=("delivery_cost", "mean"),
        average_cost_per_km=("cost_per_km", "mean")
    )
    .reset_index()
)

print("\n============================================")
print("DISTANCE-BASED COST ANALYSIS")
print("============================================")

print(
    distance_analysis.to_string(index=False)
)


# ============================================
# 25. Distance Band vs Average Delivery Cost
# ============================================

plt.figure(figsize=(9, 6))

plt.bar(
    distance_analysis["distance_band"],
    distance_analysis["average_delivery_cost"]
)

plt.xlabel("Distance Band")
plt.ylabel("Average Delivery Cost")
plt.title("Average Delivery Cost by Distance Band")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "week-4/average_cost_by_distance_band.png",
    dpi=300
)

plt.show()

# ============================================
# 26. Package Weight-Based Cost Analysis
# ============================================

df_analysis["weight_band"] = pd.cut(
    df_analysis["package_weight_kg"],
    bins=[0, 10, 20, 30, 40, float("inf")],
    labels=[
        "0-10 kg",
        "10-20 kg",
        "20-30 kg",
        "30-40 kg",
        "40+ kg"
    ],
    include_lowest=True
)

weight_analysis = (
    df_analysis
    .groupby("weight_band", observed=True)
    .agg(
        deliveries=("delivery_cost", "count"),
        average_weight_kg=("package_weight_kg", "mean"),
        average_delivery_cost=("delivery_cost", "mean")
    )
    .reset_index()
)

print("\n============================================")
print("PACKAGE WEIGHT-BASED COST ANALYSIS")
print("============================================")

print(
    weight_analysis.to_string(index=False)
)


# ============================================
# 27. Weight Band vs Average Delivery Cost
# ============================================

plt.figure(figsize=(9, 6))

plt.bar(
    weight_analysis["weight_band"],
    weight_analysis["average_delivery_cost"]
)

plt.xlabel("Package Weight Band")
plt.ylabel("Average Delivery Cost")
plt.title("Average Delivery Cost by Package Weight")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "week-4/average_cost_by_weight_band.png",
    dpi=300
)

plt.show()
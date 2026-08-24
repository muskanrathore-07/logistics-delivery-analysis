# ============================================================
# WEEK 2 - DATA COLLECTION, CLEANING & PREPROCESSING
# Logistics Delivery Analysis
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# 1. Load Dataset
# ============================================================

df = pd.read_csv("../data/Delivery_Logistics.csv")

print("Dataset loaded successfully!")
print()


# ============================================================
# 2. Basic Dataset Exploration
# ============================================================

print("First 5 Rows:")
print(df.head())
print()

print("Dataset Shape:")
print(df.shape)
print()

print("Column Names:")
print(df.columns.tolist())
print()


# ============================================================
# 3. Dataset Information
# ============================================================

print("Dataset Information:")
df.info()
print()


# ============================================================
# 4. Statistical Summary
# ============================================================

print("Statistical Summary:")
print(df.describe(include="all"))
print()


# ============================================================
# 5. Check Missing Values
# ============================================================

print("Missing Values:")
print(df.isnull().sum())
print()


# ============================================================
# 6. Check Duplicate Rows
# ============================================================

print("Number of Duplicate Rows:")
print(df.duplicated().sum())
print()


# ============================================================
# 7. Remove Duplicate Rows
# ============================================================

df = df.drop_duplicates()

print("Duplicate Rows After Removal:")
print(df.duplicated().sum())
print()


# ============================================================
# 8. Clean Column Names
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("Cleaned Column Names:")
print(df.columns.tolist())
print()


# ============================================================
# 9. Convert Delivery ID
# ============================================================

df["delivery_id"] = df["delivery_id"].astype(int)


# ============================================================
# 10. Check Original Time Values
# ============================================================

print("Original Delivery Time Values:")
print(df["delivery_time_hours"].head(10))
print()

print("Original Expected Time Values:")
print(df["expected_time_hours"].head(10))
print()


# ============================================================
# 11. Convert Time Columns
# ============================================================

def convert_time_to_hours(value):
    try:
        value = str(value)

        # Format: HH:MM.S
        if ":" in value and "." in value:
            parts = value.split(":")

            hours = float(parts[0])
            minutes = float(parts[1])

            return hours + (minutes / 60)

        # If value is already numeric
        return float(value)

    except:
        return np.nan


df["delivery_time_hours"] = df["delivery_time_hours"].apply(
    convert_time_to_hours
)

df["expected_time_hours"] = df["expected_time_hours"].apply(
    convert_time_to_hours
)


# ============================================================
# 12. Check Converted Values
# ============================================================

print("Converted Delivery Time:")
print(df["delivery_time_hours"].head(10))
print()

print("Converted Expected Time:")
print(df["expected_time_hours"].head(10))
print()


# ============================================================
# 13. Check Data Types
# ============================================================

print("Data Types After Conversion:")
print(df.dtypes)
print()


# ============================================================
# 14. Check Missing Values After Preprocessing
# ============================================================

print("Missing Values After Preprocessing:")
print(df.isnull().sum())
print()


# ============================================================
# 15. Final Dataset Shape
# ============================================================

print("Final Dataset Shape:")
print(df.shape)
print()


# ============================================================
# 16. Save Cleaned Dataset
# ============================================================

df.to_csv(
    "../data/cleaned_delivery_logistics.csv",
    index=False
)

print("Cleaned dataset saved successfully!")

# ============================================================
# 17. Check Categorical Values
# ============================================================

print("Unique Delivery Partners:")
print(df["delivery_partner"].unique())
print()

print("Unique Package Types:")
print(df["package_type"].unique())
print()

print("Unique Vehicle Types:")
print(df["vehicle_type"].unique())
print()

print("Unique Delivery Modes:")
print(df["delivery_mode"].unique())
print()

print("Unique Regions:")
print(df["region"].unique())
print()

print("Unique Weather Conditions:")
print(df["weather_condition"].unique())
print()

print("Unique Delayed Values:")
print(df["delayed"].unique())
print()

print("Unique Delivery Status:")
print(df["delivery_status"].unique())
print()

# ============================================================
# 18. Numeric Data Validation
# ============================================================

print("Numeric Columns Summary:")
print(df[
    [
        "distance_km",
        "package_weight_kg",
        "delivery_rating",
        "delivery_cost"
    ]
].describe())
print()


# ============================================================
# 19. Check Invalid Values
# ============================================================

print("Invalid Distance Values:")
print((df["distance_km"] <= 0).sum())
print()

print("Invalid Package Weight Values:")
print((df["package_weight_kg"] <= 0).sum())
print()

print("Invalid Delivery Rating Values:")
print(
    ((df["delivery_rating"] < 1) |
     (df["delivery_rating"] > 5)).sum()
)
print()

print("Invalid Delivery Cost Values:")
print((df["delivery_cost"] <= 0).sum())
print()

# ============================================================
# 20. Final Data Quality Check
# ============================================================

print("========== FINAL DATA QUALITY CHECK ==========")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())

print()
print("Final Cleaned Dataset:")
print(df.head())

print()
print("Week 2 Data Cleaning & Preprocessing Completed!")

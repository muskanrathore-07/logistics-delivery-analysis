# Week 3: Advanced Data Analysis and Visualization
# Logistics Delivery Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Load Cleaned Dataset
# --------------------------------------------------

df = pd.read_csv("../data/cleaned_delivery_logistics.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# --------------------------------------------------
# 2. Missing Values
# --------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# --------------------------------------------------
# 3. Central Tendency
# --------------------------------------------------

print("\nMean:")
print(df.mean(numeric_only=True))

print("\nMedian:")
print(df.median(numeric_only=True))

print("\nStandard Deviation:")
print(df.std(numeric_only=True))

# --------------------------------------------------
# 4. Correlation Analysis
# --------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)

print("\nCorrelation Matrix:")
print(numeric_df.corr())

# --------------------------------------------------
# 5. Visualization 1 - Distance Distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["distance_km"],
    bins=20,
    kde=True
)

plt.title("Distribution of Delivery Distance")
plt.xlabel("Distance (km)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("distance_distribution.png")
plt.show()

# --------------------------------------------------
# 6. Visualization 2 - Delivery Cost Distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["delivery_cost"],
    bins=20,
    kde=True
)

plt.title("Distribution of Delivery Cost")
plt.xlabel("Delivery Cost")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("delivery_cost_distribution.png")
plt.show()

# --------------------------------------------------
# 7. Visualization 3 - Delivery Cost by Vehicle Type
# --------------------------------------------------

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="vehicle_type",
    y="delivery_cost"
)

plt.title("Delivery Cost by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Delivery Cost")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("cost_by_vehicle_type.png")
plt.show()

# --------------------------------------------------
# 8. Visualization 4 - Distance vs Delivery Cost
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="distance_km",
    y="delivery_cost"
)

plt.title("Distance vs Delivery Cost")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Cost")

plt.tight_layout()
plt.savefig("distance_vs_delivery_cost.png")
plt.show()

# --------------------------------------------------
# 9. Visualization 5 - Correlation Heatmap
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# --------------------------------------------------
# 10. Average Cost by Delivery Partner
# --------------------------------------------------

partner_cost = (
    df.groupby("delivery_partner")["delivery_cost"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))

partner_cost.plot(kind="bar")

plt.title("Average Delivery Cost by Delivery Partner")
plt.xlabel("Delivery Partner")
plt.ylabel("Average Delivery Cost")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("average_cost_by_partner.png")
plt.show()

# --------------------------------------------------
# 11. Key Logistics Insights
# --------------------------------------------------

print("\n----- KEY LOGISTICS INSIGHTS -----")

print(
    "Average Distance:",
    round(df["distance_km"].mean(), 2), "km"
)

print(
    "Average Package Weight:",
    round(df["package_weight_kg"].mean(), 2), "kg"
)

print(
    "Average Delivery Cost:",
    round(df["delivery_cost"].mean(), 2)
)

print(
    "Average Delivery Rating:",
    round(df["delivery_rating"].mean(), 2)
)

print(
    "Maximum Delivery Distance:",
    round(df["distance_km"].max(), 2), "km"
)

print(
    "Maximum Delivery Cost:",
    round(df["delivery_cost"].max(), 2)
)

print(
    "Most Expensive Delivery Partner:",
    partner_cost.idxmax()
)

print(
    "Lowest Average Delivery Cost Partner:",
    partner_cost.idxmin()
)

print("\nWeek 3 analysis completed successfully!")
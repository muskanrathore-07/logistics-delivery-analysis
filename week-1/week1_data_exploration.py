import pandas as pd

# Load the logistics dataset
df = pd.read_csv("Delivery_Logistics.csv")

# Display first five records
print(df.head())

# Check number of rows and columns
print("Dataset shape:", df.shape)

# Check data types and basic information
print(df.info())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# Check number of unique values in each column
print("\nUnique values:")
print(df.nunique())

# Basic statistical summary
print("\nStatistical summary:")
print(df.describe())

# Check categorical distributions
print("\nDelivery partners:")
print(df["delivery_partner"].value_counts())

print("\nDelivery modes:")
print(df["delivery_mode"].value_counts())

print("\nDelay status:")
print(df["delayed"].value_counts())

# Check delivery cost
print("\nDelivery cost summary:")
print(df["delivery_cost"].describe())

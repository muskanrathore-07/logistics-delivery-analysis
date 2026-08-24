# Week 2 - Data Collection, Cleaning & Preprocessing

## Project
Logistics Delivery Analysis

## Objective

The objective of Week 2 was to collect, inspect, clean, and preprocess the logistics delivery dataset before performing further analysis and visualization.

## Dataset

The dataset contains logistics and delivery-related information.

- Total Records: 25,000
- Total Columns: 15
- Dataset Type: Logistics Delivery Data

## Data Exploration

The dataset was loaded using Pandas and examined using:

- `head()`
- `shape`
- `info()`
- `describe()`
- Column name inspection
- Missing value analysis
- Duplicate value analysis

## Data Cleaning Performed

The following cleaning and preprocessing steps were performed:

1. Loaded the original CSV dataset using Pandas.
2. Checked the structure and dimensions of the dataset.
3. Checked for missing values.
4. Checked for duplicate rows.
5. Removed duplicate rows if present.
6. Cleaned column names by removing spaces and converting them to lowercase.
7. Converted `delivery_id` to integer format.
8. Converted delivery time fields into numeric hour format.
9. Checked data types after preprocessing.
10. Validated categorical values.
11. Validated numerical columns for invalid values.
12. Saved the cleaned dataset as `cleaned_delivery_logistics.csv`.

## Missing Values

No missing values were found in the original dataset.

After preprocessing:

- Missing Values: 0

## Duplicate Records

No duplicate records were found.

- Duplicate Rows: 0

## Categorical Data Validation

The following categorical columns were checked:

- Delivery Partner
- Package Type
- Vehicle Type
- Delivery Mode
- Region
- Weather Condition
- Delayed
- Delivery Status

No obvious inconsistent categorical values were found.

## Numerical Data Validation

The following numerical columns were validated:

- Distance
- Package Weight
- Delivery Rating
- Delivery Cost

Validation results:

- Invalid Distance Values: 0
- Invalid Package Weight Values: 0
- Invalid Delivery Rating Values: 0
- Invalid Delivery Cost Values: 0

## Time Column Note

The original `delivery_time_hours` and `expected_time_hours` fields were stored in the dataset as `00:00.0` values. These values were converted into numeric hour format during preprocessing.

The source dataset therefore does not provide meaningful variation in these two time fields.

## Final Dataset

After cleaning and preprocessing:

- Rows: 25,000
- Columns: 15
- Missing Values: 0
- Duplicate Rows: 0

The cleaned dataset was saved as:

`cleaned_delivery_logistics.csv`

## Tools & Technologies

- Python
- Pandas
- NumPy
- VS Code
- GitHub

## Conclusion

The logistics delivery dataset was successfully inspected, cleaned, validated, and prepared for further analysis. The cleaned dataset can be used for visualization, exploratory data analysis, and further logistics performance analysis.

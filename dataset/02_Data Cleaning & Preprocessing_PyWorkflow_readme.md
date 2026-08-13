##  PHASE 2 : Data Cleaning & Preprocessing

* The 2nd phase of the machine learning workflow   

* focuses on transforming raw, messy data into a clean, structured, and machine-readable format. Because algorithms only understand numbers and well-defined rules, this phase bridges the gap between raw data and model training.


---

PYTHON WORKFLOW

## 1. Handling Missing Values (`NaN` or `None`)

Real-world datasets often have missing entries , which can cause errors or bias in machine learning models.

 ```python

print(df.isnull().sum()) # Check total missing values per column


df_clean = df.dropna() # Option A: Drop rows with missing values


df['Age'] = df['Age'].fillna(df['Age'].mean()) # Option B: Fill missing values with a statistic (e.g., mean)

```


* **Explanation:** `isnull().sum()` identifies gaps in your data.
*  `1. drop incomplete rows using dropna()`
*     Behavior: It drops any row that contains at least one missing value (NaN) across the entire DataFrame.

      When to use: Use this if your dataset is small, or if a missing value anywhere in a row makes that row completely useless for your model.

      Example: If Row 5 has a missing value in the Age column, the entire Row 5 is deleted
*  `OR * 2. impute (fill) them using fillna()`
*     Behavior : with measures like the mean, median, or a default string.

---

## 2. Removing Duplicate Rows

Duplicate data can skew learning by artificially inflating the importance of certain records.

```python
# Check number of duplicates
print(df.duplicated().sum())

# Drop duplicate rows, keeping the first occurrence
df.drop_duplicates(inplace=True)

```


* **Explanation:** `duplicated().sum()` counts identical rows, and `drop_duplicates()` removes the repetitions to ensure each observation is unique.

---

## 3. Encoding Categorical Variables (Text to Numbers)

Most ML algorithms cannot process text values (like `"Red"`, `"Blue"`, `"Toyota"`). We must convert them into numerical form.

* **Syntax (One-Hot Encoding):**
```python
import pandas as pd

# Convert categorical text columns into binary (0 or 1) columns
df_encoded = pd.get_dummies(df, columns=['Brand'], drop_first=True)

```


* **Explanation:** `pd.get_dummies()` splits a text column into multiple binary columns. For example, a `Brand` column with "Toyota", "Honda", and "Ford" becomes separate columns where a row has a `1` for its matching brand and `0` for others. (`drop_first=True` avoids multicollinearity).

---

## 4. Feature Scaling (Normalization / Standardization)

Features with vastly different scales (e.g., Year ranging from 2010 to 2026 vs. Mileage ranging from 0 to 150,000) can cause gradient-descent algorithms to perform poorly or overemphasize larger numbers.

* **Syntax:**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Scale specific numerical columns (e.g., Year and Mileage for car prediction)
df[['Year', 'Mileage']] = scaler.fit_transform(df[['Year', 'Mileage']])

```

* **Explanation:** `StandardScaler` centers the data around a mean of $0$ with a standard deviation of $1$, ensuring all numerical features contribute equally to the model.
Scaling **both** columns together rather than just one is a crucial step for most machine learning algorithms. Here is why you scale all numerical features together:

      * When you pass multiple features into a model, they need to speak the "same mathematical language.
      * Mileage` ranges from $0$ to $150,000$, while `Year` ranges from roughly $2010$ to $2026$ (a spread of only $16$).
      * If left unscaled, algorithms that rely on gradient descent or distance metrics will treat a change of $1,000$ in
        mileage as massive, while treating a change of $5$ years as tiny. The model will accidentally overemphasize
        Mileage and virtually ignore Year, even if the Year is equally important for predicting car prices.

         df[['Mileage']] = scaler.fit_transform(df[['Mileage']])
      * If you are using a decision-tree-based model (like Random Forest or XGBoost), scaling isn't actually required at all
        because trees split data based on thresholds, not magnitudes.
        However, for algorithms sensitive to feature magnitudes (like Linear Regression), scaling **all** your numerical input
        features together is standard practice to ensure fair and balanced learning.
---

## 5. Handling Outliers

Extreme values (outliers) can heavily distort linear models and distance-based algorithms.

* **Syntax (Using Interquartile Range - IQR):**
```python
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1

# Keep only rows within acceptable bounds
df = df[(df['Price'] >= Q1 - 1.5 * IQR) & (df['Price'] <= Q3 + 1.5 * IQR)]

```


* **Explanation:** The IQR method calculates statistical boundaries and filters out extreme high or low values that sit outside normal distribution limits.

```python
import pandas as pd
df = pd.read_csv('cars.csv')
print(df.info())
print(df.head())
print(df.describe(include='all'))


```

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1725 entries, 0 to 1724
Data columns (total 11 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   Id                       1725 non-null   int64  
 1   year                     1725 non-null   int64  
 2   brand                    1725 non-null   object 
 3   full_model_name          1725 non-null   object 
 4   model_name               1725 non-null   object 
 5   price                    1725 non-null   float64
 6   distance_travelled(kms)  1725 non-null   float64
 7   fuel_type                1725 non-null   object 
 8   city                     1725 non-null   object 
 9   brand_rank               1725 non-null   int64  
 10  car_age                  1725 non-null   float64
dtypes: float64(3), int64(3), object(5)
memory usage: 148.4+ KB
None
   Id  year          brand                                      full_model_name model_name      price  distance_travelled(kms) fuel_type    city  brand_rank  car_age
0   0  2016          Honda                                      Honda Brio S MT       Brio   425000.0                   9680.0    Petrol  Mumbai           7      5.0
1   1  2012         Nissan                               Nissan Sunny XV Diesel      Sunny   325000.0                 119120.0    Diesel  Mumbai          11      9.0
2   2  2017         Toyota               Toyota Fortuner 2.8 4x2 MT [2016-2020]   Fortuner  2650000.0                  64593.0    Diesel   Thane           1      4.0
3   3  2017  Mercedes-Benz  Mercedes-Benz E-Class E 220d Expression [2019-2019]    E-Class  4195000.0                  25000.0    Diesel  Mumbai           2      4.0
4   4  2012        Hyundai                    Hyundai Verna Fluidic 1.6 CRDi SX      Verna   475000.0                  23800.0    Diesel  Mumbai          14      9.0
                 Id         year    brand                   full_model_name model_name         price  distance_travelled(kms) fuel_type     city   brand_rank      car_age
count   1725.000000  1725.000000     1725                              1725       1725  1.725000e+03              1725.000000      1725     1725  1725.000000  1725.000000
unique          NaN          NaN       31                               750        169           NaN                      NaN         5       15          NaN          NaN
top             NaN          NaN  Hyundai  Ford EcoSport Titanium 1.5L TDCi      Creta           NaN                      NaN    Diesel  Chennai          NaN          NaN
freq            NaN          NaN      297                                12         71           NaN                      NaN       922      493          NaN          NaN
mean     862.000000  2015.390725      NaN                               NaN        NaN  1.494837e+06             53848.256232       NaN      NaN    15.731014     5.609275
std      498.108924     3.207504      NaN                               NaN        NaN  1.671658e+06             44725.541963       NaN      NaN    12.951122     3.207504
min        0.000000  1990.000000      NaN                               NaN        NaN  6.250000e+04               350.000000       NaN      NaN     1.000000     0.000000
25%      431.000000  2013.000000      NaN                               NaN        NaN  5.450000e+05             29000.000000       NaN      NaN     5.000000     3.000000
50%      862.000000  2016.000000      NaN                               NaN        NaN  8.750000e+05             49000.000000       NaN      NaN    14.000000     5.000000
75%     1293.000000  2018.000000      NaN                               NaN        NaN  1.825000e+06             70500.000000       NaN      NaN    24.000000     8.000000
max     1724.000000  2021.000000      NaN                               NaN        NaN  1.470000e+07            790000.000000       NaN      NaN    81.000000    31.000000


```

```python
import pandas as pd
df = pd.read_csv('cars.csv')
print("Missing values:")
print(df.isnull().sum())
print("\nUnique value counts for categorical variables:")
print(df[['brand', 'fuel_type', 'city']].nunique())
print("\nFuel types breakdown:")
print(df['fuel_type'].value_counts())
print("\nTop 5 brands:")
print(df['brand'].value_counts().head())


```

```text
Missing values:
Id                         0
year                       0
brand                      0
full_model_name            0
model_name                 0
price                      0
distance_travelled(kms)    0
fuel_type                  0
city                       0
brand_rank                 0
car_age                    0
dtype: int64

Unique value counts for categorical variables:
brand        31
fuel_type     5
city         15
dtype: int64

Fuel types breakdown:
fuel_type
Diesel        922
Petrol        788
CNG + 1         8
Petrol + 1      6
Hybrid          1
Name: count, dtype: int64

Top 5 brands:
brand
Hyundai          297
Maruti Suzuki    275
Honda            153
Mercedes-Benz    131
Toyota           117
Name: count, dtype: int64


```

## Exploratory Data Analysis (EDA) Report: `cars.csv`

### 1. Dataset Overview & Structure

* **Total Rows (Observations):** 1,725 records
* **Total Columns (Features):** 11 attributes
* **Missing Values:** Zero missing values detected across all columns, ensuring a clean dataset ready for modeling or deeper analysis.
* **Data Types:**
* **Numeric (6):** `Id`, `year`, `price`, `distance_travelled(kms)`, `brand_rank`, `car_age`
* **Categorical (5):** `brand`, `full_model_name`, `model_name`, `fuel_type`, `city`



---

### 2. Statistical Summary of Numerical Features

| Feature | Count | Mean | Standard Deviation | Min | 25% (Q1) | Median (Q2) | 75% (Q3) | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **year** | 1,725 | 2015.39 | 3.21 | 1990.00 | 2013.00 | 2016.00 | 2018.00 | 2021.00 |
| **price** | 1,725 | 1,494,837 | 1,671,658 | 62,500 | 545,000 | 875,000 | 1,825,000 | 14,700,000 |
| **distance_travelled (kms)** | 1,725 | 53,848 | 44,725 | 350.00 | 29,000 | 49,000 | 70,500 | 790,000 |
| **car_age** | 1,725 | 5.61 | 3.21 | 0.00 | 3.00 | 5.00 | 8.00 | 31.00 |
| **brand_rank** | 1,725 | 15.73 | 12.95 | 1.00 | 5.00 | 14.00 | 24.00 | 81.00 |

* **Price Distribution:** Highly right-skewed. The median price sits at **₹875,000**, while the mean price is significantly higher at **₹1,494,837**, driven by high-end luxury listings reaching up to **₹14,700,000**.
* **Vehicle Age:** The dataset represents relatively modern pre-owned vehicles. The middle 50% of cars range between 3 to 8 years of age (median age of 5 years), with the oldest record being 31 years old (1990 model).
* **Distance Travelled:** Spans from near-zero use (350 kms) to high mileage (790,000 kms), with a median utilization of **49,000 kms**.

---

### 3. Categorical Breakdown & Cardinality

* **Brands (31 unique):** Dominated heavily by mass-market and popular luxury labels. The top represented brands are **Hyundai** (297 listings), **Maruti Suzuki** (275), **Honda** (153), **Mercedes-Benz** (131), and **Toyota** (117).
* **Fuel Types (5 unique):**
* **Diesel:** 922 entries (~53.4%)
* **Petrol:** 788 entries (~45.7%)
* **CNG + 1 / Petrol + 1 / Hybrid:** 15 entries combined (~0.9%)


* **Geographic Spread (15 unique cities):** Listings are spread across metropolitan hubs, with **Chennai** (493 listings) capturing the largest share in the city field.
* **Models:** Contains 169 unique base model names (e.g., *Creta*, *Fortuner*, *E-Class*) expanded across 750 distinct full variants (`full_model_name`).

---

### 4. Key Insights & Analytical Takeaways

* **Price vs. Age/Mileage Correlation Potential:** Strong inverse relationship expected between `price` and both `car_age` and `distance_travelled(kms)`.
* **Powertrain Preference:** Diesel listings outpace petrol vehicles in this secondary market sample, reflecting preferences in the regional segments recorded.
* **Outliers:** Upper-bound price and mileage metrics exhibit heavy tails (max price of ~1.47 Cr and max mileage of 790k kms), which should be treated appropriately via scaling or robust modeling techniques if building predictive valuation pipelines.

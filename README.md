# Machine Learning-Based Car Price Prediction: Benchmarking Linear, Ensemble, and Non-Parametric Models

 

### 1. Data Preprocessing & Categorical Encoding

* **Label Encoding**: Non-numeric categorical features (`brand`, `model_name`, `city`, `fuel_type`, and `full_model_name`) are converted into numerical format using `sklearn.preprocessing.LabelEncoder`.


* **Dataset Structure**: The dataset consists of 1,725 rows and 11 initial columns (including `Id`, `year`, `price`, `distance_travelled(kms)`, and `car_age`).



### 2. Variance Analysis

* **Variance Computation**: The script calculates the variance across all numerical features using `df_2.var()` to detect columns with minimal variability.


* **Feature Retention & Dropping**:
* The `city` column is dropped into a separate DataFrame variant (`df_var`) due to low perceived effectiveness or variance.


* Experimentation notes within the code indicate that dropping `fuel_type` reduced model accuracy from 79% to 75%, confirming its predictive importance.


### 3. Correlation Analysis

* **Correlation Matrix**: A complete correlation matrix is generated using `df_2.corr()` to identify multicollinearity and linear dependencies between features.


* **Key Observations**:
* **Multicollinearity**: `brand` and `full_model_name` exhibit an extremely high positive correlation of **0.983**, indicating redundancy between brand categories and specific model titles.


* **Inverse Relationship**: `year` and `car_age` show a perfect negative correlation of **-1.0** (since car age is derived directly from the manufacturing year).


* **Price Drivers**: Car `price` shows positive correlation with `year` (**0.288**) and negative correlations with `car_age` (**-0.288**), `fuel_type` (**-0.259**), and `distance_travelled(kms)` (**-0.137**).




### File Title

* **File 1 (`cars.ipynb`)**: Car Price Prediction and Exploratory Data Analysis using Linear Regression



---

### README: Machine Learning Algorithm Comparison

#### Overview

This project processes a used car dataset (`cars.csv`) to predict vehicle prices using machine learning. The current implementation employs **Linear Regression** as the core predictive algorithm.

Below is a comparison of **Linear Regression** (currently implemented) with other standard machine learning regression algorithms that can be applied to this dataset for performance benchmarking.

---

#### Algorithm Comparison Table

| Algorithm | Model Type | Strengths | Weaknesses | Suitability for Car Price Data |
| --- | --- | --- | --- | --- |
| **Linear Regression**<br> | Parametric / Linear | Highly interpretable, fast to train, works well with linearly related features. | Sensitive to outliers, assumes a linear relationship between features and target price. | **Good baseline**, but car depreciation and pricing often involve non-linear interactions. |
| **Decision Tree Regressor** | Non-parametric / Tree-based | Captures non-linear relationships, easy to visualize and interpret, requires minimal feature scaling. | Prone to overfitting on noisy data, unstable with small variations in the training set. | **Moderate**, handles categorical-turned-numeric features well via splits. |
| **Random Forest Regressor** | Ensemble (Bagging) | High accuracy, reduces overfitting compared to single trees, handles non-linearities and interactions well. | Slower inference time, less interpretable "black box" model. | **High**, excellent for handling tabular pricing data with mixed feature types. |
| **Gradient Boosting / XGBoost** | Ensemble (Boosting) | State-of-the-art predictive accuracy, handles complex feature interactions efficiently. | Requires careful hyperparameter tuning, computationally expensive. | **High**, typically yields the lowest error metrics (such as MAPE) for real-world regression tasks. |

---

#### Key Implementation Details  

* Data Preprocessing: Categorical features (`brand`, `model_name`, `city`, `fuel_type`, `full_model_name`) are transformed into numeric values using `LabelEncoder`.


* Feature Selection: Variance and correlation matrices are evaluated; features like `city` are dropped to optimize model accuracy.


* Evaluation Metric: Model performance is evaluated using **Mean Absolute Percentage Error (MAPE)**.


## Feasible Regression Algorithms for Car Price Prediction

When predicting a continuous variable like car price based on mixed tabular features (e.g., brand, year, distance traveled, fuel type, and age), several machine learning algorithms are typically evaluated. Below is a detailed comparison of the most feasible models:

---

### 1. Linear & Regularized Models

* **Linear Regression**
* **How it works:** Fits a linear equation to minimize the residual sum of squares between observed and predicted targets.
* **Pros:** Highly interpretable, fast to train, and acts as a strong baseline.
* **Cons:** Prone to poor performance if relationships are non-linear; sensitive to multicollinearity (such as high correlation between `year` and `car_age` or `brand` and `full_model_name`).


* **Ridge and Lasso Regression**
* **How it works:** Variants of linear regression that add L2 (Ridge) or L1 (Lasso) regularization penalties.
* **Pros:** Helps mitigate multicollinearity; Lasso can perform automatic feature selection by shrinking coefficients of irrelevant features to zero.
* **Cons:** Fundamentally limited to linear relationships unless polynomial features are explicitly engineered.



---

### 2. Instance-Based Models

* **K-Neighbors Regressor (KNN)**
* **How it works:** Predicts the target by finding the $k$-closest training samples in the feature space and averaging their prices.
* **Pros:** Non-parametric and simple to understand; captures local data patterns without assuming a global functional form.
* **Cons:** Computationally expensive during inference for large datasets; highly sensitive to feature scaling and the curse of dimensionality when dealing with numerous encoded categorical columns.



---

### 3. Tree-Based & Ensemble Models

* **Random Forest Regressor**
* **How it works:** Builds multiple randomized decision trees in parallel using bagging and averages their predictions.
* **Pros:** Handles non-linear relationships and interactions between features (e.g., how vehicle age interacts with brand) exceptionally well; robust to outliers and multicollinearity; requires minimal feature scaling.
* **Cons:** Can overfit if hyperparameters (such as `max_depth` and `n_estimators`) are untuned; less interpretable than single linear models.


* **Bagging & Boosting Regressors (AdaBoost, XGBoost)**
* **How it works:** Sequentially corrects errors of previous models (boosting) or trains base estimators on random subsets (bagging).
* **Pros:** Frequently achieves the highest predictive accuracy on tabular real-world datasets like automotive listings; handles heterogeneous feature types effectively.
* **Cons:** Prone to overfitting if learning rates and tree depths are not properly regularized; computationally heavier to train.




### Key Evaluation Considerations

* **Target Distribution:** Car prices are typically right-skewed (many economy cars, few luxury supercars). Applying a log transformation to the target variable and evaluating using Root Mean Squared Logarithmic Error (RMSLE) or $R^2$ score usually yields more stable and accurate model comparisons.
* **Multicollinearity Impact:** Algorithms like Linear Regression require feature dropping or regularization when features like `brand` and `full_model_name` exhibit near-perfect correlation, whereas tree-based ensemble models handle this redundancy naturally.




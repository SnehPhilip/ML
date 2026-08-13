The **data splitting phase** is where you divide your cleaned dataset into training and testing sets. This ensures that your model trains on a portion of the data and is evaluated on completely unseen data to test its real-world performance.

Here are the Python syntactical steps for the data splitting phase using **scikit-learn**:

---

### Step 1: Separate Features ($X$) and Target ($y$)

Before splitting, you must separate your dataset into **features** (the independent variables the model will learn from) and the **target** (the dependent variable or label the model will try to predict).

* **Syntax:**
```python
# X contains all columns except the target column
X = df.drop('Price', axis=1)

# y contains only the target column we want to predict
y = df['Price']

```


* **Explanation:** `df.drop('Price', axis=1)` removes the target column from the feature set, while `y = df['Price']` isolates the target variable.

---

### Step 2: Import the Splitting Function

Import the `train_test_split` module from scikit-learn's model selection library.

* **Syntax:**
```python
from sklearn.model_selection import train_test_split

```


* **Explanation:** `train_test_split` is the standard function used to randomly partition arrays or matrices into training and testing subsets.

---

### Step 3: Execute the Split

Unpack the dataset into training and testing components by specifying the split ratio and a random state.

* **Syntax:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

```


* **Explanation of Parameters:**
* `test_size=0.2`: Allocates 20% of the data for testing and the remaining 80% for training. (You can also use `train_size=0.8`).
* `random_state=42`: Sets a seed for the random number generator. Using a fixed number ensures that your data split is reproducible—meaning every time you run the code, the data will be shuffled and split in the exact same way.

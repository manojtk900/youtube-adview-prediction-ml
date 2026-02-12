import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("Loading dataset...")

train = pd.read_csv("train_list.csv")
test = pd.read_csv("test_list.csv")

print(train.head())


# ---------------------------------------------------
# Convert numeric columns from string → numbers
# ---------------------------------------------------

numeric_cols = ["views", "likes", "dislikes", "comment"]

for col in numeric_cols:
    train[col] = pd.to_numeric(train[col], errors='coerce')
    test[col] = pd.to_numeric(test[col], errors='coerce')

train[numeric_cols] = train[numeric_cols].fillna(0)
test[numeric_cols] = test[numeric_cols].fillna(0)


# ---------------------------------------------------
# Convert Duration (PT7M37S → seconds)
# ---------------------------------------------------

def convert_duration(duration):
    pattern = re.compile(r'PT(\d+)M(\d+)S')
    match = pattern.match(str(duration))
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return minutes * 60 + seconds
    return 0

train["duration"] = train["duration"].apply(convert_duration)
test["duration"] = test["duration"].apply(convert_duration)


# ---------------------------------------------------
# Convert Published Date
# ---------------------------------------------------

train["published"] = pd.to_datetime(train["published"])
test["published"] = pd.to_datetime(test["published"])

train["year"] = train["published"].dt.year
train["month"] = train["published"].dt.month
train["day"] = train["published"].dt.day

test["year"] = test["published"].dt.year
test["month"] = test["published"].dt.month
test["day"] = test["published"].dt.day

train.drop("published", axis=1, inplace=True)
test.drop("published", axis=1, inplace=True)


# ---------------------------------------------------
# Drop Video ID
# ---------------------------------------------------

train.drop("vidid", axis=1, inplace=True)
test.drop("vidid", axis=1, inplace=True)


# ---------------------------------------------------
# One-Hot Encode Category
# ---------------------------------------------------

train = pd.get_dummies(train, columns=["category"], drop_first=True)
test = pd.get_dummies(test, columns=["category"], drop_first=True)

# Ensure same columns
test = test.reindex(columns=train.columns, fill_value=0)


# ---------------------------------------------------
# Separate Features & Target
# ---------------------------------------------------

X = train.drop("adview", axis=1)

# 🔥 Log transformation (important improvement)
y = np.log1p(train["adview"])


# ---------------------------------------------------
# Train-Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------------------------------------------
# Train Model
# ---------------------------------------------------

print("\nTraining model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ---------------------------------------------------
# Evaluate Model
# ---------------------------------------------------

# Predict (log scale)
y_pred_log = model.predict(X_test)

# Convert back to original scale
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

print("\nModel Performance:")
print("R2 Score:", r2_score(y_test_actual, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test_actual, y_pred)))

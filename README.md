# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1

2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.

3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:

~~~
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    LabelEncoder
)

from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    RFE,
    SelectFromModel
)

from sklearn.linear_model import LogisticRegression

# STEP 1: Read Dataset
df = pd.read_csv("C:\\Users\\acer\\Downloads\\income(1) (1).csv")

# STEP 2: Data Cleaning
df = df.drop_duplicates()
df = df.dropna()

# Remove extra spaces
df = df.apply(
    lambda x: x.str.strip()
    if x.dtype=="object"
    else x
)

print("Original Dataset")
print(df.head())
~~~
<img width="461" height="274" alt="image" src="https://github.com/user-attachments/assets/f2c9c816-d9d7-42de-b95b-f841cd48275b" />

~~~

# Encode categorical columns
encoder = LabelEncoder()

for col in df.select_dtypes(include="object"):
    df[col] = encoder.fit_transform(df[col])

# Separate features and target
X = df.drop("SalStat", axis=1)
y = df["SalStat"]

# STEP 3: Feature Scaling

# Standard Scaling
std = StandardScaler()
X_std = std.fit_transform(X)

# MinMax Scaling
minmax = MinMaxScaler()
X_min = minmax.fit_transform(X)

# Max Absolute Scaling
maxabs = MaxAbsScaler()
X_max = maxabs.fit_transform(X)

# Robust Scaling
robust = RobustScaler()
X_robust = robust.fit_transform(X)

# Use Standard Scaled data
X_scaled = pd.DataFrame(
    X_std,
    columns=X.columns
)

# STEP 4: Feature Selection

# Filter Method
filter_select = SelectKBest(
    score_func=chi2,
    k=5
)

X_filter = filter_select.fit_transform(
    abs(X_scaled),
    y
)

selected_filter = X.columns[
    filter_select.get_support()
]

print("\nFilter Selected Features")
print(selected_filter)
~~~
<img width="537" height="46" alt="image" src="https://github.com/user-attachments/assets/c58e7b78-aed2-4bb8-8749-df1709cc292f" />

~~~


# Wrapper Method (RFE)

model = LogisticRegression(
    max_iter=1000
)

rfe = RFE(
    model,
    n_features_to_select=5
)

rfe.fit(X_scaled, y)

selected_wrapper = X.columns[
    rfe.support_
]

print("\nWrapper Selected Features")
print(selected_wrapper)
~~~
<img width="570" height="45" alt="image" src="https://github.com/user-attachments/assets/adf95979-3e2c-4b24-859e-e0d517393b23" />

~~~


# Embedded Method

embed = SelectFromModel(
    LogisticRegression(
        penalty="l1",
        solver="liblinear"
    )
)

embed.fit(X_scaled, y)

selected_embed = X.columns[
    embed.get_support()
]

print("\nEmbedded Selected Features")
print(selected_embed)
~~~
<img width="417" height="73" alt="image" src="https://github.com/user-attachments/assets/7080376b-57d6-4d22-bfc2-b2fdc8ddcc51" />


~~~
# Save Output

output = X_scaled.copy()
output["SalStat"] = y

output.to_csv(
    "Scaled_Selected_Output.csv",
    index=False
)

print("\nData saved as Scaled_Selected_Output.csv")
~~~
<img width="246" height="21" alt="image" src="https://github.com/user-attachments/assets/76e1bb7f-7bfb-4ceb-a0f0-fca8aed54bbd" />





       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
# RESULT:
       # INCLUDE YOUR RESULT HERE

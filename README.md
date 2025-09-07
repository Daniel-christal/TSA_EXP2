# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:06.09.2025
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
file_path = "bmw.csv"
df = pd.read_csv(file_path)

# Show columns to verify
print("Available columns:", df.columns)

# Convert month to datetime if available
if "Month" in df.columns:
    df['Month'] = pd.to_datetime(df['Month'])

# Create time index
df['t'] = np.arange(1, len(df) + 1)

# Identify passenger column (choose 2nd column if unknown)
passenger_col = df.columns[1]   # first is Month, second is usually passenger count
y = df[passenger_col].values
X = df[['t']].values

# ----- Linear Regression -----
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

# ----- Polynomial Regression (degree 2) -----
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

# ----- Linear Plot -----
plt.figure(figsize=(10, 5))
plt.scatter(df['t'], y, color="blue", label="Actual Data")
plt.plot(df['t'], y_linear_pred, color="black", linestyle="--", label="Linear Trend")
plt.xlabel("Time (Months)")
plt.ylabel("Passengers")
plt.title("Linear Trend Estimation")
plt.legend()
plt.show()

# ----- Polynomial Plot -----
plt.figure(figsize=(10, 5))
plt.scatter(df['t'], y, color="blue", label="Actual Data")
sorted_zip = sorted(zip(df['t'], y_poly_pred))
X_sorted, y_poly_sorted = zip(*sorted_zip)
plt.plot(X_sorted, y_poly_sorted, color="red", linestyle="--", label="Polynomial Trend (Degree 2)")
plt.xlabel("Time (Months)")
plt.ylabel("Passengers")
plt.title("Polynomial Trend Estimation (Degree 2)")
plt.legend()
plt.show()

##powerlifting:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
file_path = "openpowerlifting.csv"
df = pd.read_csv(file_path)

# ---- Select relevant columns (adjust names if needed) ----
# Drop rows with NaN in either BodyweightKg or TotalKg
data = df[['BodyweightKg', 'TotalKg']].dropna()

X = data[['BodyweightKg']].values
y = data['TotalKg'].values

# ----- Linear Regression -----
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

# ----- Polynomial Regression (degree 2) -----
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

# ----- Linear Plot -----
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="blue", s=10, alpha=0.3, label="Actual Data")
plt.plot(X, y_linear_pred, color="black", linestyle="--", label="Linear Trend")
plt.xlabel("Bodyweight (Kg)")
plt.ylabel("Total Lifted (Kg)")
plt.title("Linear Trend Estimation")
plt.legend()
plt.show()

# ----- Polynomial Plot -----
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="blue", s=10, alpha=0.3, label="Actual Data")
sorted_zip = sorted(zip(X.flatten(), y_poly_pred))
X_sorted, y_poly_sorted = zip(*sorted_zip)
plt.plot(X_sorted, y_poly_sorted, color="red", linestyle="--", label="Polynomial Trend (Degree 2)")
plt.xlabel("Bodyweight (Kg)")
plt.ylabel("Total Lifted (Kg)")
plt.title("Polynomial Trend Estimation (Degree 2)")
plt.legend()
plt.show()
```


### OUTPUT
A - LINEAR TREND ESTIMATION

<img width="937" height="472" alt="image" src="https://github.com/user-attachments/assets/0cc2f109-bd56-44d4-a9c0-0fd4c01e4997" />

B- POLYNOMIAL TREND ESTIMATION

<img width="857" height="463" alt="image" src="https://github.com/user-attachments/assets/66486a70-62f4-40dd-ac23-45e441f320d4" />


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.

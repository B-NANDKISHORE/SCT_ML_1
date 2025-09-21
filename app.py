import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# =====================
# 1. Load dataset
# =====================
# Use the path where your train.csv was extracted
train_data = pd.read_csv("C:/Users/HOME/OneDrive/Desktop/ML/train.csv")

# =====================
# 2. Select features and target
# =====================
features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
X = train_data[features]
y = train_data["SalePrice"]

# =====================
# 3. Split into training and test sets
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# 4. Train linear regression model
# =====================
model = LinearRegression()
model.fit(X_train, y_train)

# =====================
# 5. Make predictions on test set
# =====================
y_pred = model.predict(X_test)

# =====================
# 6. Evaluate model
# =====================
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=== Model Results ===")
print("Intercept:", model.intercept_)
print("Coefficients:", dict(zip(features, model.coef_)))
print("RMSE:", rmse)
print("R²:", r2)
print("=====================\n")

# =====================
# 7. User input for prediction
# =====================
sqft = float(input("Enter square footage (GrLivArea): "))
bedrooms = int(input("Enter number of bedrooms: "))
baths = int(input("Enter number of full bathrooms: "))

# Prepare input
user_house = pd.DataFrame([[sqft, bedrooms, baths]], columns=features)

# Predict price
predicted_price = model.predict(user_house)[0]
print(f"\nPredicted House Price: ${predicted_price:,.2f}")


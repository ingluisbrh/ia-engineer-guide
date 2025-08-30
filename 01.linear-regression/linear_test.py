# 1. Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 2. Sample data (X = house size in m², y = price in thousands $)
X = np.array([[30], [50], [60], [80], [100], [120], [150], [200]])  # Feature
y = np.array([100, 150, 180, 240, 300, 360, 450, 500])          # Target

# 3. Create and train the model
model = LinearRegression()
model.fit(X, y)

# 4. Make predictions
predicted = model.predict(X)

# 5. Print slope (coef_) and intercept
print(f"Slope (m): {model.coef_[0]:.2f}")
print(f"Intercept (b): {model.intercept_:.2f}")

# 6. Predict price of a new house with 110 m²
new_house = np.array([[110]])
predicted_price = model.predict(new_house)[0]
print(f"Predicted price for 110m² house: ${predicted_price:.2f}k")

# 7. Plot the results
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, predicted, color="red", linewidth=2, label="Regression line")
plt.title("House Price vs. Size")
plt.xlabel("House size (m²)")
plt.ylabel("Price (thousands $)")
plt.legend()
plt.show()

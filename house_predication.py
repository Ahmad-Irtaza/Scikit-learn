# Step 1: Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Step 2: Training data (old data)
# Ghar ka area (input feature)
area = np.array([[1000], [1200], [1300], [1500], [1800]])
# Ghar ka price (target)
price = np.array([3000000, 3500000, 3700000, 4000000, 5000000])

# Step 3: Create and train the model
model = LinearRegression()
model.fit(area, price)

# Step 4: Make new prediction
new_area = np.array([[1600]])  # New ghar ka area
predicted_price = model.predict(new_area)

# Step 5: Show result
print(f"Agar ghar ka area {new_area[0][0]} sq ft ho, to estimated price: Rs {int(predicted_price[0])}")

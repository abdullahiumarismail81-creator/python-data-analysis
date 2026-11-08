from sklearn.linear_model import LinearRegression
import numpy as np

house_size = np.array([[100], [150], [200], [250]])
house_price = np.array([10, 15, 20, 25])

model = LinearRegression()

model.fit(house_size, house_price)

prediction = model.predict([[300]])

print("Predicted Price:", prediction[0])

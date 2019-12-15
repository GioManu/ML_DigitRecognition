import pandas as pd
# -- Read Data
train_data = pd.read_csv("data//train.csv")

# -- labels = Y, pixels = X
labels = train_data.pop("label")
pixels = train_data

# -- Dividing data for train/test data by 1/7
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(pixels, labels, test_size=1/7.0)

# -- Passing Data to RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -- Prediction on first test
y_pred = model.predict(X_test)
print(y_pred)
print(model.score(X_test, y_test))

# -- Printing accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_test))

# -- Predict on first test digit
y_predicted = model.predict(X_test.values[0].reshape(1,-1))

# -- Printing first digit plot
import numpy as np
import matplotlib.pyplot as plt

label = y_predicted
pixel = X_test.values[0]
pixel = np.array(pixel, dtype='uint8')
pixel = pixel.reshape((28,28))
plt.title('Label is {label}'.format(label=label))
plt.imshow(pixel, cmap='gray')
plt.show()


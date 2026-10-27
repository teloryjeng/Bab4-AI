from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import time

whole_data = load_breast_cancer()
X_data = whole_data.data
y_data = whole_data.target

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=7)

# Menggunakan 5 neuron pada dense layers
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=[X_train.shape[1]]))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

start = time.time()
model.fit(X_train, y_train, batch_size=50, validation_split=0.2, epochs=100, verbose=0)
training_time = time.time() - start

results = model.evaluate(X_test, y_test, verbose=0)
print(f"Training Time: {training_time:.2f} detik")
print(f"Loss: {results[0]}, MSE: {results[1]}")
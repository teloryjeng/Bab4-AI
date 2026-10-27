from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D

model = Sequential()
# Mengubah filters dari 6 ke 12, dan 16 ke 32
model.add(Conv2D(filters=12, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(AveragePooling2D())
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.summary()
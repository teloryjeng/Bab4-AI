# Example 4.8
from keras.datasets import mnist
import matplotlib.pyplot as plt

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plot 4 images as grayscale
plt.subplot(141)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(142)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(143)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(144)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
plt.show()
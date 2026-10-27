# Example 4.10
from tensorflow.keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet')
print(model.summary())
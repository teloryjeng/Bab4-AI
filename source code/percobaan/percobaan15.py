# Solusi Exercise 4.15 - Visualisasi Layer Konvolusi Kedua 
from keras.applications.vgg16 import VGG16
from matplotlib import pyplot 

# 1. Load model VGG16
model = VGG16()

# 2. n = 2 untuk memilih layer konvolusi kedua (block1_conv2)
n = 2 
filters, biases = model.layers[n].get_weights() 
s = filters.shape 

print("Color channels:", s[2]) 
print("Filter size:", s[0], s[1])
print("Total number of filters:", s[3])

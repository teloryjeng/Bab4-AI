# Solusi Exercise 4.16 - Menampilkan Seluruh Filter
from keras.applications.vgg16 import VGG16
from matplotlib import pyplot

# 1. Load model VGG16
model = VGG16()

# 2. Ambil bobot dari layer konvolusi pertama (n = 1)
n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape  # (tinggi, lebar, channel, jumlah_filter)

# 3. Normalisasi nilai filter ke rentang 0-1 untuk visualisasi
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters, ix = s[3], 1  # s[3] adalah total jumlah filter (64)

pyplot.figure(figsize=(8, 40))
for i in range(n_filters):
    # Ambil filter ke-i
    f = filters[:, :, :, i]
    
    # Plot setiap channel secara terpisah (RGB = 3 channel)
    for j in range(s[2]):
        ax = pyplot.subplot(n_filters, s[2], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[:, :, j], cmap='gray')
        ix += 1

pyplot.show()

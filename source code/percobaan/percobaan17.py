from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims

def plot_feature_maps(feature_maps, title="Feature Maps"):
    # Plot feature maps (up to 64 maps in an 8x8 grid)
    col = 8
    row = min(int(feature_maps.shape[3] / col), 8)
    ix = 1
    plt.figure(figsize=(12, 12))
    plt.suptitle(title)
    for _ in range(row):
        for _ in range(col):
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    plt.show()

# Load model VGG19
model = VGG19()

# Pilih layer yang ingin divisualisasikan dengan mengubah nilai n:
# n = 1 -> block1_conv1 (layer konvolusi ke-1)
# n = 2 -> block1_conv2 (layer konvolusi ke-2)
# n = 4 -> block2_conv1 (layer konvolusi ke-3)
# n = 5 -> block2_conv2 (layer konvolusi ke-4)
n = 2 # Mengubah n = 1 menjadi n = 2 untuk memilih layer konvolusi kedua

# Redefinisi model agar mengeluarkan output setelah hidden layer ke-n
model = Model(inputs=model.inputs, outputs=model.layers[n].output)
model.summary()

# Load dan pra-pemrosesan gambar
img = load_img('Elephant.jpg', target_size=(224, 224))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)

# Dapatkan feature maps untuk layer yang dipilih
feature_maps = model.predict(img)
print("Feature maps shape:", feature_maps.shape)

# Tampilkan seluruh feature maps
plot_feature_maps(feature_maps, title=f"Layer {n} ({model.layers[n].name})")
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST dataset and preprocess
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # Add channels dimension
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # Add channels dimension

# Build the CNN model directly using Sequential
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Visualize the filters (kernels) of the first convolutional layer
def visualize_filters(model, layer_index):
    layer = model.layers[layer_index]
    filters, biases = layer.get_weights()

    # Normalize filter values to [0, 1] for visualization
    filters = (filters - filters.min()) / (filters.max() - filters.min())

    num_filters = filters.shape[3]
    size = filters.shape[0]

    # Create a grid of subplots
    n_cols = 8
    n_rows = num_filters // n_cols
    fig, axarr = plt.subplots(n_rows, n_cols, figsize=(10, 10))

    for i in range(num_filters):
        ax = axarr[i // n_cols, i % n_cols]
        ax.imshow(filters[:, :, 0, i], cmap='gray')
        ax.axis('off')

    plt.show()

# Visualize filters for the first convolutional layer
visualize_filters(model, 0)

# Function to visualize activations for a given image
def visualize_activations(model, image):
    activations = []

    # Define a model to fetch the activations of all layers
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.MaxPooling2D):
            model_activations = tf.keras.models.Model(inputs=model.input, outputs=layer.output)
            activations.append(model_activations.predict(image[np.newaxis, ...]))

    # Plot activations for each layer
    layer_names = [layer.name for layer in model.layers]
    for i, activation in enumerate(activations):
        activation = activation[0]
        num_filters = activation.shape[-1]
        size = activation.shape[0]

        # Create a grid of subplots for each layer's feature maps
        n_cols = 8
        n_rows = num_filters // n_cols
        fig, axarr = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        for j in range(num_filters):
            ax = axarr[j // n_cols, j % n_cols]
            ax.imshow(activation[:, :, j], cmap='gray')
            ax.axis('off')

        plt.suptitle(f"Layer: {layer_names[i]}")
        plt.show()

# Visualize activations for a random image from the dataset
visualize_activations(model, x_test[0])

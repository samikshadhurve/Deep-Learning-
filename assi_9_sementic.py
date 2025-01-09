import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Define function to load and preprocess images and masks
def load_data(image_dir, mask_dir, image_size=(128, 128)):
    images = []
    masks = []

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Ensure you're loading only the right image types
            # Load the image
            image = load_img(os.path.join(image_dir, filename), target_size=image_size)
            image = img_to_array(image) / 255.0  # Normalize the image
            images.append(image)

            mask_filename = 'annotated_' + filename

            # Load the corresponding mask
            mask_path = os.path.join(mask_dir, mask_filename)
            try:
                mask = load_img(mask_path, target_size=image_size, color_mode='grayscale')
                mask = img_to_array(mask) / 255.0  # Normalize the mask
                masks.append(mask)
            except FileNotFoundError:
                print(f"Mask file not found: {mask_path}")  # Debugging: Output missing mask path

    return np.array(images), np.array(masks)


# Load training data
image_dir = r'C:\Users\samik\OneDrive\Documents\clg_assignments\semantic folder\Images'  # Replace with your image directory path
mask_dir = r'C:\Users\samik\OneDrive\Documents\clg_assignments\semantic folder\Labels'    # Replace with your mask directory path

X_train, y_train = load_data(image_dir, mask_dir)

# Check the shape of the data
print(X_train.shape, y_train.shape)
from tensorflow.keras import layers, models

def unet_model(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)

    # Contracting path (Encoder)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    # Bottleneck
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)

    # Expanding path (Decoder)
    up5 = layers.UpSampling2D((2, 2))(conv4)
    concat5 = layers.concatenate([conv3, up5], axis=-1)
    conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(concat5)
    conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

    up6 = layers.UpSampling2D((2, 2))(conv5)
    concat6 = layers.concatenate([conv2, up6], axis=-1)
    conv6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat6)
    conv6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    up7 = layers.UpSampling2D((2, 2))(conv6)
    concat7 = layers.concatenate([conv1, up7], axis=-1)
    conv7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat7)
    conv7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    # Output layer (for binary segmentation, use sigmoid)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv7)

    model = models.Model(inputs, outputs)

    return model

# Create U-Net model
model = unet_model(input_size=(128, 128, 3))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()
# Train the model
model.fit(X_train, y_train, epochs=3, batch_size=16, validation_split=0.1)
# Predict segmentation mask for a new image (e.g., from the test set)
sample_image = X_train[0]  # Take one sample from your training data
sample_image = np.expand_dims(sample_image, axis=0)  # Add batch dimension

# Predict the mask
predicted_mask = model.predict(sample_image)

# Plot the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(X_train[0])  # Original image
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(predicted_mask[0].squeeze(), cmap='gray')  # Predicted mask
plt.title("Predicted Mask")
plt.show()
# import os
#
# # Check if the directories exist
# print(os.path.exists(r'C:\Users\samik\OneDrive\Documents\clg_assignments\semantic folder\Images'))  # Should return True
# print(os.path.exists(r'C:\Users\samik\OneDrive\Documents\clg_assignments\semantic folder\Labels'))   # Should return True
#
# # List some files in both directories to verify
# print(os.listdir(r'C:\Users\samik\OneDrive\Documents\clg_assignments\semantic folder\Images'))  # Should list image files like dog.8789.jpg
# print(os.listdir(r'C:\Users\samik\OneDrive\Documents\clg_assignments\semantic folder\Labels'))   # Should list corresponding mask files like dog.8789_mask.png

import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            # The label is in the filename 'defect_1.jpg'
            if 'defect' in filename.lower():
                labels.append(1)  # 1 for defective
            else:
                labels.append(0)  # 0 for non-defective
    return np.array(images), np.array(labels)

# Load AITEX dataset from kaggle
aitex_folder = 'https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection'
aitex_images, aitex_labels = load_images_from_folder(aitex_folder)

# Load DAGM 2007 dataset from Kaggle
dagm_folder = 'https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection'
dagm_images, dagm_labels = load_images_from_folder(dagm_folder)

# Combining the datasets
images = np.concatenate((aitex_images, dagm_images), axis=0)
labels = np.concatenate((aitex_labels, dagm_labels), axis=0)

# Normalize images
images = images.astype('float32') / 255.0

input_img = Input(shape=(images.shape[1], images.shape[2], 1))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

# Select only non-defective images for training
non_defective_images = images[labels == 0]

# Reshape for CNN
non_defective_images = np.expand_dims(non_defective_images, axis=-1)

# Split into training and validation sets
X_train, X_val = train_test_split(non_defective_images, test_size=0.2, random_state=42)

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=128, validation_data=(X_val, X_val))

def detect_anomalies(autoencoder, images, threshold=0.1):
    anomalies = []
    for img in images:
        img = np.expand_dims(img, axis=(0, -1))
        reconstructed_img = autoencoder.predict(img)
        error = np.mean(np.abs(reconstructed_img - img))
        if error > threshold:
            anomalies.append(True)
        else:
            anomalies.append(False)
    return np.array(anomalies)

# Detect anomalies in all images
anomalies = detect_anomalies(autoencoder, images)

# Visualize some results
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title('Defect' if anomalies[i] else 'Normal')
    plt.axis('off')
plt.show()

# Calculate reconstruction errors
reconstruction_errors = []
for img in images:
    img = np.expand_dims(img, axis=(0, -1))
    reconstructed_img = autoencoder.predict(img)
    error = np.mean(np.abs(reconstructed_img - img))
    reconstruction_errors.append(error)

reconstruction_errors = np.array(reconstruction_errors).reshape(-1, 1)

# Fit KDE
kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(reconstruction_errors)

# Score samples (higher scores indicate more likely defects)
scores = kde.score_samples(reconstruction_errors)

# Set a threshold based on density estimation
threshold = np.percentile(scores, 95)  # For example, 95th percentile

# Detect anomalies
anomalies_kde = scores < threshold

# ground truth labels
accuracy = accuracy_score(labels, anomalies_kde)
precision = precision_score(labels, anomalies_kde)
recall = recall_score(labels, anomalies_kde)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

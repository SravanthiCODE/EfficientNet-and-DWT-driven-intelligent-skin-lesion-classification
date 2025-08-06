# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import pywt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

#Load metadata
metadata = pd.read_csv('/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')

# Map 7 classes into 2 classes: Benign and Malignant
benign_classes = ['nv', 'bkl', 'df', 'vasc']
malignant_classes = ['mel', 'bcc', 'akiec']

def map_binary(label):
    return 'benign' if label in benign_classes else 'malignant'

metadata['binary_label'] = metadata['dx'].apply(map_binary)

# Encode labels
label_encoder = LabelEncoder()
metadata['label_encoded'] = label_encoder.fit_transform(metadata['binary_label'])

# First split into train_val and test
from sklearn.model_selection import train_test_split
train_val_df, test_df = train_test_split(metadata, test_size=0.1, random_state=42, stratify=metadata['label_encoded'])

# Then split train_val into train and validation
train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['label_encoded'])

# Function: Apply DWT
def dwt2_transform(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    coeffs2 = pywt.dwt2(img_gray, 'haar')
    LL, (LH, HL, HH) = coeffs2 
    return LL

# Memory-efficient Data Generator
class DataGenerator(Sequence):
    def __init__(self, df, batch_size=32, img_size=(224,224), shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_df = self.df.iloc[batch_indexes]
        
        X = np.empty((len(batch_df), *self.img_size, 3), dtype=np.float32)
        y = np.empty((len(batch_df),), dtype=np.float32)
        for i, (_, row) in enumerate(batch_df.iterrows()):
            img_id = row['image_id']
            label = row['label_encoded']
            
            path1 = f'/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1/{img_id}.jpg'
            path2 = f'/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_2/{img_id}.jpg'
            
            img_path = path1 if os.path.exists(path1) else path2
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = cv2.GaussianBlur(img, (5,5), 0)        # Gaussian filtering
            
            img_dwt = dwt2_transform(img)
            img_dwt = cv2.resize(img_dwt, self.img_size)
            img_dwt = np.stack((img_dwt,)*3, axis=-1)  # 3-channel
            
            X[i,] = img_dwt / 255.0  # Normalize
            y[i] = label
            
        return X, y

# Initialize generators
batch_size = 32
train_gen = DataGenerator(train_df, batch_size=batch_size)
val_gen = DataGenerator(val_df, batch_size=batch_size, shuffle=False)
test_gen = DataGenerator(test_df, batch_size=batch_size)

# Example: load one image
img_id = test_df.iloc[0]['image_id']  # take first image from test_df

# Image paths
path1 = '/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1/' + img_id + '.jpg'
path2 = '/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_2/' + img_id + '.jpg'

if os.path.exists(path1):
    img_path = path1
else:
    img_path = path2

# 1. Load the image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Perform DWT
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
coeffs2 = pywt.dwt2(img_gray, 'haar')
LL, (LH, HL, HH) = coeffs2
img_dwt = LL  # Using only Approximation coefficients

# Resize DWT to original size for better comparison (optional)
img_dwt_resized = cv2.resize(img_dwt, (img.shape[1], img.shape[0]))

# 3. Plot side-by-side
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# After DWT
plt.subplot(1, 2, 2)
plt.imshow(img_dwt_resized, cmap='gray')
plt.title('After DWT (Approximation)')
plt.axis('off')

plt.show()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

augmenter = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

def train_with_augmentation(generator, augmenter):
    while True:
        X_batch, y_batch = generator.__getitem__(np.random.randint(0, len(generator)))
        aug_iter = augmenter.flow(X_batch, batch_size=X_batch.shape[0], shuffle=False)
        X_batch_augmented = next(aug_iter)
        yield X_batch_augmented, y_batch

train_augmented_gen = train_with_augmentation(train_gen, augmenter)

# Build EfficientNetB3 model
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(1, activation='sigmoid')(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=output)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_augmented_gen,
    steps_per_epoch=len(train_gen),
    epochs=15,
    validation_data=val_gen,
    verbose=1
)

# Unfreeze the base model
base_model.trainable = True

# Optionally, freeze earlier layers (like first 80%) and unfreeze last few layers
for layer in base_model.layers[:-30]:  # Keep first layers frozen
    layer.trainable = False

# Re-compile with a low learning rate
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=1e-5), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Fine-tune
fine_tune_history = model.fit(
    train_augmented_gen,
    steps_per_epoch=len(train_gen),
    epochs=10,  # smaller number of epochs here
    validation_data=val_gen,
    verbose=1
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate metrics
y_true = val_df['label_encoded'].values
y_pred_probs = model.predict(val_gen)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")



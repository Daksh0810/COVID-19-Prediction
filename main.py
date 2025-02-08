import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# ✅ Set image size (consistent for training and testing)
IMG_SIZE = (150, 150)  # Keep the same for both training & testing
BATCH_SIZE = 16  # Smaller batch size for GPU efficiency

# ✅ Dataset directories
train_dir = r"COVID_19_Detection\Dataset\train"
test_dir = r"COVID_19_Detection\Dataset\test"

# ✅ Data Augmentation (Balanced for better generalization)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)

# ✅ Load training and testing images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# ✅ Load Pretrained MobileNetV2 Model (Lighter & Faster)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(150, 150, 3))

# ✅ Unfreeze some layers for fine-tuning
fine_tune_at = 50  # Unfreeze layers after 50
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# ✅ Create the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary Classification (COVID / Non-COVID)
])

# ✅ Compile the model
model.compile(optimizer=Adam(lr=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Callbacks for better training
checkpoint = ModelCheckpoint("covid_model.h5", save_best_only=True, save_weights_only=False)
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# ✅ Train the model
EPOCHS = 20  # More epochs, but early stopping prevents overfitting
model.fit(train_generator, validation_data=test_generator, epochs=EPOCHS, callbacks=[checkpoint, lr_scheduler, early_stopping])

print("Model Training Complete (Optimized)")

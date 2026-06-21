import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

IMG_SIZE = 128

# -------------------------
# Data Augmentation
# -------------------------

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# -------------------------
# Training Data
# -------------------------

train_data = train_gen.flow_from_directory(
    "train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

# -------------------------
# Validation Data
# -------------------------

val_data = train_gen.flow_from_directory(
    "train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# -------------------------
# Class Labels
# -------------------------

print("\nClass Labels:")
print(train_data.class_indices)

# -------------------------
# CNN Model
# -------------------------

model = Sequential([

    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    Conv2D(32, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(256, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(512, activation="relu"),

    Dropout(0.5),

    Dense(
        train_data.num_classes,
        activation="softmax"
    )
])

# -------------------------
# Compile
# -------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------
# Callbacks
# -------------------------

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1
)

# -------------------------
# Summary
# -------------------------

model.summary()

# -------------------------
# Training
# -------------------------

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)

# -------------------------
# Final Accuracy
# -------------------------

print(
    "\nFinal Validation Accuracy:",
    max(history.history["val_accuracy"])
)

# -------------------------
# Save Model
# -------------------------

model.save(
    "image_classifier.keras"
)

print("\nModel Saved Successfully")

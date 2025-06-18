import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib
import numpy as np

# Configuration
DATA_DIR = "./processed_data"  
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 40
SEED = 42  # For reproducibility

# 1. Create dataset from directory structure (LOAD AS GRAYSCALE)
raw_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=None,  # Get individual images
    shuffle=True,
    seed=SEED,
    label_mode='int',
    color_mode='grayscale'  # CHANGED TO GRAYSCALE
)

# Get total number of images
total_images = raw_ds.cardinality().numpy()
print(f"Total images: {total_images}")

# Calculate split sizes
train_size = int(0.7 * total_images)
val_size = int(0.15 * total_images)
test_size = total_images - train_size - val_size

print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

# Split the dataset
train_ds = raw_ds.take(train_size)
val_ds = raw_ds.skip(train_size).take(val_size)
test_ds = raw_ds.skip(train_size + val_size).take(test_size)

# Batch datasets and configure performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(AUTOTUNE)

test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(AUTOTUNE)

# 2. Create the model with data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# UPDATED INPUT SHAPE FOR GRAYSCALE (1 CHANNEL)
model = models.Sequential([
    layers.Input(shape=(*IMAGE_SIZE, 1)),  # CHANGED TO 1 CHANNEL
    data_augmentation,
    layers.Rescaling(1./255),  # Normalize pixel values
    
    # Convolutional base
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Classifier head
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# 3. Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# 5. Evaluate on test set
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# 6. Save the model
model.save('model.keras')

# (Optional) Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('training_history.png')
plt.show()

# 7. Make prediction on a single image (UPDATED FOR GRAYSCALE)
def predict_image(image_path):
    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode='grayscale'  # CHANGED TO GRAYSCALE
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension

    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions[0]).numpy()
    confidence = tf.reduce_max(predictions[0]).numpy()
    
    return predicted_class, confidence

# Example usage:
predicted_class, confidence = predict_image('processed_data/3/0c0504c4-72cf-4de9-bd62-80e80355ca6d.png')
print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
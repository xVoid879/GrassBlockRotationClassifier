import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Configuration (must match training)
IMAGE_SIZE = (64, 64)  # Same as training
MODEL_PATH = 'model.keras'  # Path to your saved model
CLASS_NAMES = ['0', '1', '2', '3']  # Replace with your actual class names

# 1. Load the trained model
try:
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        # Create a dummy model that returns random predictions
        # This is just for testing purposes until a real model is available
        print("Creating dummy model for testing")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        print("Dummy model created")
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a dummy model as a fallback
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

# 2. Create prediction function
def predict_image(image_path):
    """
    Predict class for a single image
    Returns: (predicted_class, class_name, confidence)
    """
    # Load and resize image
    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode='grayscale'  # CHANGED TO GRAYSCALE
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
    # No manual normalization - model does it internally through its Rescaling layer
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    class_name = CLASS_NAMES[predicted_class]
    
    return predicted_class, class_name, confidence

##print(predict_image("processed_data/3/0c0504c4-72cf-4de9-bd62-80e80355ca6d.png"))
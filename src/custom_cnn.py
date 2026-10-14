import tensorflow as tf

from config import IMAGE_SIZE, NUM_CLASSES


def build_custom_cnn():

    model = tf.keras.Sequential([
        
        # Input layer
        tf.keras.layers.Input(
            shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        ),

        # Block 1
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Block 2
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Block 3
        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Convert feature maps into a vector
        tf.keras.layers.Flatten(),

        # Fully connected layer
        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        # Prevent overfitting
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    return model
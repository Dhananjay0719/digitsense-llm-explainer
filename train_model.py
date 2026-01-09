# import tensorflow as tf
# # from tensorflow.keras.datasets import mnist
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import (
# #     Conv2D, MaxPooling2D, Dense, Flatten, Dropout
# # )
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# # Load MNIST
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Normalize
# x_train = x_train / 255.0
# x_test = x_test / 255.0

# # Reshape for CNN
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)

# # CNN Model
# model = Sequential([
#     Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
#     MaxPooling2D((2,2)),

#     Conv2D(64, (3,3), activation="relu"),
#     MaxPooling2D((2,2)),

#     Flatten(),
#     Dense(128, activation="relu"),
#     Dropout(0.4),
#     Dense(10, activation="softmax")
# ])

# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.fit(
#     x_train, y_train,
#     epochs=15,           
#     validation_split=0.1,
#     batch_size=128
# )

# loss, acc = model.evaluate(x_test, y_test)
# print("Test accuracy:", acc)

# model.save("digits.keras")

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape for CNN input
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Define CNN model (Keras 3 safe)
model = Sequential([
    Input(shape=(28, 28, 1)),     # IMPORTANT for stable loading
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(10, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model (no verbose output)
model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    verbose=0        # 🔥 prevents accuracy/loss spam
)

# Save model
# model.save("digits.keras")
model.save_weights("digits.weights.h5")
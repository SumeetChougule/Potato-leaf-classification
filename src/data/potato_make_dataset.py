from cgi import test
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config import IMAGE_SIZE, BATCH_SIZE, CHANNELS, EPOCHS
from random import shuffle
from tkinter.tix import IMAGE
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import split_dataset
from CNN import model

from preprocessing import *


print(sys.path)
dir = "../../data/raw/potato"


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dir, shuffle=True, image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE
)

class_names = dataset.class_names
len(dataset)

# plt.figure(figsize=(10, 10))
# for images_batch, label_batch in dataset.take(1):
#     for i in range(12):
#         ax = plt.subplot(3, 4, i + 1)
#         plt.imshow(images_batch[i].numpy().astype("uint8"))
#         plt.title(class_names[label_batch[i]])
#         plt.axis("off")


# Split
train_ds, val_ds, test_ds = split_dataset(
    dataset, train_size=0.8, val_size=0.1, test_size=0.1, seed=12
)

# Print the number of samples in each split
print(
    "Number of samples in training set:",
    tf.data.experimental.cardinality(train_ds).numpy(),
)
print(
    "Number of samples in validation set:",
    tf.data.experimental.cardinality(val_ds).numpy(),
)
print(
    "Number of samples in test set:", tf.data.experimental.cardinality(test_ds).numpy()
)

train_ds.cache().shuffle(1000)

train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
    ]
)

data_augmentation = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ]
)


def export_dataset_to_tfrecord(dataset, filename):
    flat_dataset = dataset.flat_map(
        lambda x, y: tf.data.Dataset.from_tensor_slices((x, y))
    )

    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset)


def preprocess_and_export_datasets(train_ds, val_ds, test_ds, export_path):
    # Cache, shuffle, and prefetch for training dataset
    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)

    # Cache, shuffle, and prefetch for validation dataset
    val_ds = val_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)

    # Cache, shuffle, and prefetch for test dataset
    test_ds = test_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)

    # Image resizing and rescaling
    resize_and_rescale = tf.keras.Sequential(
        [
            layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
            layers.experimental.preprocessing.Rescaling(1.0 / 255),
        ]
    )

    # Data augmentation
    data_augmentation = tf.keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            layers.experimental.preprocessing.RandomRotation(0.2),
        ]
    )

    # Apply preprocessing to datasets
    train_ds = train_ds.map(lambda x, y: (resize_and_rescale(x), y))
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    # val_ds = val_ds.map(lambda x, y: (resize_and_rescale(x), y))

    # test_ds = test_ds.map(lambda x, y: (resize_and_rescale(x), y))

    return train_ds, val_ds, test_ds


# Example usage:
train_ds, val_ds, test_ds = preprocess_and_export_datasets(
    train_ds, val_ds, test_ds, "../../data/interim/potato"
)


input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = models.Sequential(
    [
        resize_and_rescale,
        # data_augmentation,
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ]
)

model.build(input_shape)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(
        learning_rate=0.001
    ),  # You can adjust the learning rate as needed
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_ds, batch_size=BATCH_SIZE, validation_data=val_ds, epochs=6, verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_ds)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(6, 6))
plt.subplot(1, 2, 1)
plt.plot(range(6), acc, label="Training Accuracy")
plt.plot(range(6), val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")

plt.subplot(1, 2, 1)
plt.plot(range(6), loss, label="Training loss")
plt.plot(range(6), val_loss, label="Validation loss")
plt.legend(loc="upper right")


for images_batch, labels_batch in test_ds.take(1):
    first_image = images_batch[0].numpy().astype("uint8")
    first_label = labels_batch[0].numpy()

    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:", class_names[first_label])

    batch_prediction = model.predict(images_batch)
    print("predicted label", class_names[np.argmax(batch_prediction)[0]])


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]]

        plt.title(
            f"Actual: {actual_class},\n Predicted: {predicted_class},\n Confidence {confidence}%"
        )
        plt.axis("off")


model_version = 1

model.save(f"../models/hi.h5")

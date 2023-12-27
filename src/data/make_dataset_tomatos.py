from cgi import test
import imp
import sys
import os


from random import shuffle
from tkinter.tix import IMAGE
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
from config import *
from preprocessing import DataPrep

dir = "../../data/raw/tomato"


dataset_t = tf.keras.preprocessing.image_dataset_from_directory(
    dir, shuffle=True, image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE
)

class_names = dataset_t.class_names
len(dataset_t)


for images_batch, label_batch in dataset_t.take(1):
    print(images_batch.shape)
    print(label_batch.numpy())

plt.figure(figsize=(15, 10))
for images_batch, label_batch in dataset_t.take(1):
    for i in range(12):
        ax = plt.subplot(4, 3, i + 1)
        plt.imshow(images_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")

splitter = DataPrep(dataset_t, train_size=0.8, val_size=0.1, test_size=0.1, seed=42)
train_set, val_set, test_set = splitter.split_dataset()

# Print the number of samples in each split
print(
    "Number of samples in training set:",
    tf.data.experimental.cardinality(train_set).numpy(),
)
print(
    "Number of samples in validation set:",
    tf.data.experimental.cardinality(val_set).numpy(),
)
print(
    "Number of samples in test set:", tf.data.experimental.cardinality(test_set).numpy()
)

# Preprocess and export datasets
(
    train_set_processed,
    val_set_processed,
    test_set_processed,
) = splitter.preprocess_and_export_datasets(
    train_set, val_set, test_set, export_path="your_export_path"
)

for images_batch, label_batch in train_set_processed.take(1):
    print(images_batch[0].numpy())
    print(label_batch.numpy())

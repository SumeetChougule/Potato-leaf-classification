import tensorflow as tf
from tensorflow.keras import layers
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config import *


class DataPrep:
    def __init__(self, dataset, train_size=0.8, val_size=0.1, test_size=0.1, seed=None):
        self.dataset = dataset
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed

    def split_dataset(self):
        num_samples = tf.data.experimental.cardinality(self.dataset).numpy()
        num_train = int(self.train_size * num_samples)
        num_val = int(self.val_size * num_samples)
        num_test = int(self.test_size * num_samples)

        if self.seed is not None:
            tf.random.set_seed(self.seed)

        shuffled_dataset = self.dataset.shuffle(buffer_size=num_samples, seed=self.seed)
        train_dataset = shuffled_dataset.take(num_train)
        remaining_dataset = shuffled_dataset.skip(num_train)
        val_dataset = remaining_dataset.take(num_val)
        test_dataset = remaining_dataset.skip(num_val)

        return train_dataset, val_dataset, test_dataset

    def preprocess_and_export_datasets(self, train_ds, val_ds, test_ds, export_path):
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
        # train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

        # val_ds = val_ds.map(lambda x, y: (resize_and_rescale(x), y))

        # test_ds = test_ds.map(lambda x, y: (resize_and_rescale(x), y))

        return train_ds, val_ds, test_ds


# def split_dataset(dataset, train_size=0.8, val_size=0.1, test_size=0.1, seed=None):
#     """
#     Split a TensorFlow dataset into training, validation, and test sets.

#     Parameters:
#     - dataset: TensorFlow dataset to be split.
#     - train_size: The proportion of the dataset to include in the training split.
#     - val_size: The proportion of the dataset to include in the validation split.
#     - test_size: The proportion of the dataset to include in the test split.
#     - seed: Seed for reproducibility.

#     Returns:
#     - train_dataset: Training dataset.
#     - val_dataset: Validation dataset.
#     - test_dataset: Test dataset.
#     """
#     # Get the total number of samples in the dataset
#     num_samples = tf.data.experimental.cardinality(dataset).numpy()

#     # Calculate the number of samples for each split
#     num_train = int(train_size * num_samples)
#     num_val = int(val_size * num_samples)
#     num_test = int(test_size * num_samples)

#     # Set seed for reproducibility
#     if seed is not None:
#         tf.random.set_seed(seed)

#     # Shuffle the dataset and split it into training, validation, and test sets
#     shuffled_dataset = dataset.shuffle(buffer_size=num_samples, seed=seed)
#     train_dataset = shuffled_dataset.take(num_train)
#     remaining_dataset = shuffled_dataset.skip(num_train)
#     val_dataset = remaining_dataset.take(num_val)
#     test_dataset = remaining_dataset.skip(num_val)

#     return train_dataset, val_dataset, test_dataset

import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config import IMAGE_SIZE, BATCH_SIZE, CHANNELS, EPOCHS
import tensorflow as tf
from tensorflow.keras import models, layers
from make_dataset_tomatos import *

input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 10

model = models.Sequential(
    [
        # resize_and_rescale,
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
    train_set_processed,
    batch_size=BATCH_SIZE,
    validation_data=val_set_processed,
    epochs=6,
    verbose=1,
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_set_processed)

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


for images_batch, labels_batch in test_set_processed.take(1):
    first_image = images_batch[0].numpy().astype("uint8")
    first_label = labels_batch[0].numpy()

    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:", class_names[first_label])

    batch_prediction = model.predict(images_batch)
    print("predicted label", class_names[np.argmax(batch_prediction)])


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions)]
    confidence = round(100 * (np.max(predictions)), 2)
    return predicted_class, confidence


plt.figure(figsize=(15, 15))
for images, labels in test_set_processed.take(1):
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

model.save(f"../models/tomatos{model_version}.h5")

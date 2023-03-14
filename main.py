import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Creation du dataset à partir d'images des patterns figurants sur les pdf
img_height = 28
img_width = 28
batch_size = 2

model = keras.Sequential([
    layers.Input((28, 28, 1)),
    layers.Conv2D(16, 3, padding='same'),
    layers.Conv2D(32, 3, padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10),
])

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/',
    labels = 'inferred',
    label_mode = 'int',
    color_mode = 'grayscale',
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle = True,
    seed = 123,
    validation_split = 0.1,
    subset = 'training',
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/',
    labels = 'inferred',
    label_mode = 'int',
    color_mode = 'grayscale',
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle = True,
    seed = 123,
    validation_split = 0.1,
    subset = 'validation',
)

def augment(x, y):
    image = tf.image.random_brightness(x, max_delta=0.05)
    return image, y

ds_train = ds_train.map(augment)

# Custom Loops
for epochs in range(10):
    for x, y in ds_train:
        #train here
        pass

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[
        keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    ],
    metrics=["accuracy"],
)

model.fit(ds_train, epochs=10, verbose=2)
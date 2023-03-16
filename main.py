import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Creation du dataset à partir d'images des patterns figurants sur les pdf
img_height = 32
img_width = 72
batch_size = 25

def train_the_model(h, w, b):
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

    print(ds_train)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(h, w)))
    model.add(tf.keras.layers.Dense(units=1024, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=1024, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=b, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(ds_train, epochs=55)

    loss, accuracy = model.evaluate(ds_train)
    print(accuracy)
    print(loss) 

    model.save('ground.model')

train_the_model(img_height, img_width, batch_size)

model = tf.keras.models.load_model('ground.model')

for x in range(1, 4):
    img = cv.imread(f'{x}.jpg')[:,:,0]
    img = np.invert(np.array([img]))

    prediction = model.predict(img)
    # print(prediction)
    print(f'Surement : {np.argmax(prediction)}')

    plt.imshow(img[0], cmap= plt.cm.binary)
    plt.show()
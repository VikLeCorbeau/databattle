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
img_width = 66
batch_size = 22

# ds_train = tf.keras.preprocessing.image_dataset_from_directory(
#     'dataset/',
#     labels = 'inferred',
#     label_mode = 'int',
#     color_mode = 'grayscale',
#     batch_size = batch_size,
#     image_size = (img_height, img_width),
#     shuffle = True,
#     seed = 123,
#     validation_split = 0.1,
#     subset = 'training',
# )

# ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
#     'dataset/',
#     labels = 'inferred',
#     label_mode = 'int',
#     color_mode = 'grayscale',
#     batch_size = batch_size,
#     image_size = (img_height, img_width),
#     shuffle = True,
#     seed = 123,
#     validation_split = 0.1,
#     subset = 'validation',
# )

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(img_height, img_width)))
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=batch_size, activation=tf.nn.softmax))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(ds_train, epochs=1000)

# loss, accuracy = model.evaluate(ds_train)
# print(accuracy)
# print(loss) 

# model.save('ground.model')

model = tf.keras.models.load_model('ground.model')

for x in range(1, 6):
    img = cv.imread(f'{x}.jpg')[:,:,0]
    img = np.invert(np.array([img]))

    prediction = model.predict(img)
    # print(prediction)
    print(f'Surement : {np.argmax(prediction)}')

    plt.imshow(img[0], cmap= plt.cm.binary)
    plt.show()
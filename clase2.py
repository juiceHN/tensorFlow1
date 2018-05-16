import math
import random
import numpy as np
import tensorflow as tf


def is_even(n):
    return n % 2 == 0


def generate_data(n, maximun):
    data = []
    labels = []
    bit_count = int(math.ceil(math.log2(maximun)))
    for i in range(n):
        number = random.randint(0, maximun)
        label = 1 if is_even(number) else 0
        data.append([int(i) for i in '{0:b}'.format(number).zfill(bit_count)])
        labels.append(label)

    return np.array(data), np.array(labels)


print('building data')
training_data, training_labels = generate_data(200000, 1000000)
testing_data, testing_labels = generate_data(50000, 1000000)

# keras

# 20 bits
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(20, input_shape=(20, )))
model.add(tf.keras.layers.Dense(20))
model.add(tf.keras.layers.Dense(2))
model.add(tf.keras.layers.Flatten())

print('compiling model')
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(), metrics=['accuracy'])

print('training model')
model.fit(training_data, training_labels, epochs=3, verbose=1)

(loss, accuracy) = model.evaluate(testing_data, testing_labels)
print('Test loss: ', loss)
print('Test accuracy: ', accuracy)

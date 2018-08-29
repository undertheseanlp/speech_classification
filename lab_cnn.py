import librosa
import numpy as np
from preprocess import extract_features
import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

import glob
from os.path import dirname, basename
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class CategoryEncoder:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder()

    def fit_transform(self, y):
        tmp = y
        tmp = self.label_encoder.fit_transform(tmp)
        tmp = tmp.reshape(len(tmp), 1)
        tmp = self.one_hot_encoder.fit_transform(tmp)
        return tmp

    def reverse_transform(y):
        return

files = glob.glob("data/train_100/*/*")
labels = []
is_first = False
n_files = len(files)
for i, file in enumerate(files):
    print(f"{i}/{n_files}: {file}")
    label = basename(dirname(file))
    features = extract_features(file)
    n = features.shape[0]
    labels.extend(n * [label])
    if not is_first:
        x_train = features
        is_first = True
    else:
        x_train = np.concatenate((x_train, features), axis=0)

encoder = CategoryEncoder()
y_train = encoder.fit_transform(labels)

model = Sequential()
input_shape = (60, 41, 2)
num_classes = 2
model.add(Conv2D(32, kernel_size=(5, 2), strides=(1, 1),
                 activation='relu',
                 input_shape=(60, 41, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

x_test, y_test = x_train, y_train

batch_size = 32
epochs = 100
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
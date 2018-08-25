import numpy as np
from numpy.random import seed

seed(2018)
from tensorflow import set_random_seed

set_random_seed(2018)
import pandas as pd

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam, Adadelta, Nadam, Adamax, RMSprop
from sklearn.model_selection import train_test_split
from keras import backend as K

K.tensorflow_backend._get_available_gpus()


def get_model(timeseries, nfeatures, nclass):
    model = Sequential()
    model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True,
                   input_shape=(timeseries, nfeatures)))
    model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    model.add(Dense(units=nclass, activation='softmax'))

    return model


data = np.load('./tmp/voice_zaloai/train.npz')
X, gender, accent = data['X'], data['gender'], data['region']
X_train, X_test, gender_train, gender_test, accent_train, accent_test = train_test_split(X, gender, accent,
                                                                                         test_size=0.2,
                                                                                         random_state=2018)

publictest = np.load('./tmp/voice_zaloai/publictest.npz')
X_publictest, fname = publictest['X'], publictest['name']
print('train test: ', X_train.shape, X_test.shape)
print('public test: ', X_publictest.shape)


batch_size = 1024
nb_epochs = 1000

# Train Gender
opt = RMSprop()
model = get_model(X.shape[1], X.shape[2], 2)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()
print("Train Gender")
model.fit(X_train, gender_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_test, gender_test),
          verbose=2)
predicts = model.predict(X_publictest, batch_size=batch_size)
gender_predicts = np.argmax(predicts, axis=1)
gender_dict = {0: 'female', 1: 'male'}
for i in range(32):
    print(fname[i], '-->', gender_dict[gender_predicts[i]])

# Train Accent
print("Train Accent")
opt = RMSprop()
model = get_model(X.shape[1], X.shape[2], 3)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()
model.fit(X_train, accent_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_test, accent_test),
          verbose=2)
predicts = model.predict(X_publictest, batch_size=batch_size)
accent_predicts = np.argmax(predicts, axis=1)
accent_dict = {0: 'north', 1: 'central', 2: 'south'}
for i in range(32):
    print(fname[i], '-->', accent_dict[accent_predicts[i]])

# Results
df = pd.DataFrame({
    "id": fname,
    "gender": gender_predicts,
    "accent": accent_predicts
})
df = df.sort_values("id", ascending=True)
df.to_csv('tmp/submit.csv', index=False)

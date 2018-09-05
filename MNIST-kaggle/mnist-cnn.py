import pandas as pd
import itertools
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPool2D, Flatten
from keras.optimizers import RMSprop, Adadelta, Adam
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.examples.tutorials.mnist import input_data as mnistoriginal
from keras.utils import multi_gpu_model

# download the dataset from https://www.kaggle.com/c/digit-recognizer/data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

x_train = train.drop('label',axis=1)
y_train = train['label']

x_train = x_train/255
test = test/255

x_train = x_train.values.reshape(-1, 28,28,1)
test = test.values.reshape(-1, 28,28,1)
y_train = to_categorical(y_train)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=30)

model = Sequential()

model.add(Conv2D(filters= 32, kernel_size=(5,5), strides=(1, 1), padding='same', activation='relu', input_shape = (28, 28, 1)))
model.add(Conv2D(filters= 32, kernel_size=(5,5), strides=(1, 1), padding='same', activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters= 64, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(filters= 64, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(1536, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='softmax'))

parallel_model = multi_gpu_model(model, gpus=4)
parallel_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

epochs = 50
batch_size = 64

idatagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)

idatagen.fit(x_train)

history = parallel_model.fit_generator(idatagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(x_val, y_val), verbose=2, steps_per_epoch=x_train.shape[0]//batch_size, callbacks=[lr_reducer])

y_pred = model.predict(x_val)
y_pred_class = np.argmax(y_pred, axis=1)
y_true_class = np.argmax(y_val, axis=1)
cmat = confusion_matrix(y_true_class, y_pred_class)
plot_confusion_matrix(cmat, classes=range(10))

errors = y_pred_class - y_true_class != 0

x_val_error = x_val[errors]
y_true_error = y_true_class[errors]
y_pred_error = y_pred_class[errors]
y_pred_prob_error = np.max(y_pred[errors], axis=1)

df = pd.DataFrame({'true' : y_true_error, 'pred' : y_pred_error, 'prob': y_pred_prob_error})

df_sorted = df.sort_values(by='prob', ascending=False)

idx = 10
print( "true", y_true_error[idx])
print( "predicted ", y_pred_error[idx])

result = model.predict(test)
result = np.argmax(result, axis=1)
imageid = range(1,len(result)+1)

submission = pd.DataFrame({'ImageId':imageid, 'Label':result})

submission.to_csv("submission.csv", index=False)

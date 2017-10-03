import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.cross_validation import train_test_split
from keras.models import model_from_json
import os

class LeNet5Classifier:
    def __init__(self):
        self.model = Sequential()
    def train(self, X, y, num_classes, input_shape = (128, 128)):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        self.batch_size = 128
        self.epochs = 25
        self.num_classes = num_classes
        self.img_rows, self.img_cols = input_shape[0], input_shape[1]
        if K.image_data_format() == 'channels_first':
            X_train = X_train.reshape(X_train.shape[0], 1, self.img_rows, self.img_cols)
            X_valid = X_valid.reshape(X_valid.shape[0], 1, self.img_rows, self.img_cols)
            input_shape = (1, self.img_rows, self.img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], self.img_rows, self.img_cols, 1)
            X_valid = X_valid.reshape(X_valid.shape[0], self.img_rows, self.img_cols, 1)
            input_shape = (self.img_rows, self.img_cols, 1)


        self.input_shape = input_shape
        
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_data=(X_valid, y_valid))

    def save(self, path):
        try:
            self.model.save_weights(path + 'model.h5')
            model_json = self.model.to_json()
            with open(path + 'model.json', 'w') as json_file:
                json_file.write(model_json)
            print("Model Saved to Disk at " + path)
        except:
            print("There was an error. Make sure you trained the model.")

    def load(self, path):
        try:
            json_file = open(path + 'model.json', 'r')
            model_json = json_file.read()
            json_file.close()

            self.model = model_from_json(model_json)
            self.model.load_weights(path + 'model.h5')

            self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

            print("Loaded model from disk")
        except:
            print("Something went wrong. Make sure you have previously saved the model in the excat path.")
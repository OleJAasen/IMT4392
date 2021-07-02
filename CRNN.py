import tensorflow as tf
from keras import models
from keras.models import load_model
from keras.layers import Dense, LSTM, Flatten, TimeDistributed, Conv2D, Dropout
from keras.utils import np_utils
from keras import optimizers
from keras.engine import input_layer
from keras.models import model_from_json
from keras import backend as Kb
from keras.utils import print_summary
from keras import Sequential
import numpy as np
import os, sys, random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import get_source_inputs
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras import optimizers


# rows = 224
# cols = 224
# channels = 3
# frames = 6
def create_model():
    model = models.Sequential()

    # load pretrained model
    json_file = open('model.json', 'r')
    loaded_model = json_file.read()
    json_file.close()
    vgg = model_from_json(loaded_model)
    vgg.load_weights("weights.h5")

#    for layer in vgg.layers:
#        layer.trainable = True
    
    # time distributed input
    # frames = input_layer.Input(shape=(6,224,224,3))
    #model = load_model('final.tf', compile = False)

    model.add(TimeDistributed(vgg, input_shape=(6, 224, 224, 3)))
    model.add(TimeDistributed(Flatten()))  # layer to convert to one dimensional input for the LSTM
    model.add(LSTM(256, activation='relu', return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(8, activation='softmax'))
    return model


def recover_3darrays(emotions_dir, neutral_instances):
    """Generates a single X, y arrays using all Numpy binary file.
      Args:
        emotions_dir: String path to folder with emotion folders.
        neutral_instances: Number of neutral instances to add to the X, y
            arrays. Given the high number of neutral instances that are
            generated, even with a low class weight in the training phase
            the model will have a poor performance. A good choice can be
            a number between 30 - 50.
      Returns:
        An array X with all 3D images on the dataset.
        An array y with the labels of all 3D images on the dataset.
    """

    labels = sorted(os.listdir(emotions_dir))

    for index1, label in enumerate(labels):
        if index1 == 0:
            print("Recovering arrays for label", label)
            for index2, npy in enumerate(
                    os.listdir(emotions_dir + label)[:neutral_instances]):
                im = np.load(emotions_dir + label + '/' + npy)
                if index1 == 0 and index2 == 0:
                    X = np.zeros((0, im.shape[0], im.shape[1], im.shape[2],
                                  im.shape[3]))
                    y = np.zeros((0, len(labels)))
                X = np.append(X, [im], axis=0)

                y_temp = [0] * len(labels)
                for index, lab in enumerate(labels):
                    if int(label) == int(lab):
                        y_temp[index] = 1.0
                        break
                y = np.append(y, [y_temp], axis=0)
        else:
            print("Recovering arrays for label", label)
            for index2, npy in enumerate(os.listdir(emotions_dir + label)):
                im = np.load(emotions_dir + label + '/' + npy)
                X = np.append(X, [im], axis=0)

                y_temp = [0] * len(labels)
                for index, lab in enumerate(labels):
                    if int(label) == int(lab):
                        y_temp[index] = 1.0
                        break
                y = np.append(y, [y_temp], axis=0)

    print("\nShape of X array:", X.shape)
    print("Shape of y array:", y.shape)
    return X, y


def train_test_valid_split(X, y, test_size, valid_size):
    """Generates the train, test and validation datasets.
      Args:
        X: Numpy array with all input images.
        y: Numpy array with all labels.
        test_size: Float percentage in the range (0, 1) of images
            used in test set.
        valid_size: Float percentage in the range (0, 1) of images
            used in validation set.

      Returns:
        Arrays of images and labels for each data partition.
    """

    total_size = len(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    train_size = len(y_train)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=total_size * valid_size / train_size)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def main(argv):
    emotions_dir = str(argv[0])
    neutral_instances = int(argv[1])
    valid_split = float(argv[2])
    test_split = float(argv[3])
    batch_size = int(argv[4])
    epochs = int(argv[5])

    try:
        assert (valid_split + test_split < .9)
    except AssertionError as e:
        print('Please check the validation and test set sizes.')
        raise
    
    X, y = recover_3darrays(emotions_dir, neutral_instances=neutral_instances)
    y_counts = np.sum(y, axis=0, dtype=np.int32)
    keys = range(len(y_counts))
    distrib = dict(zip(keys, y_counts))
    print("Class distribution:", distrib)

    X_train, y_train, X_valid, y_valid, X_test, y_test = \
        train_test_valid_split(X, y, test_split, valid_split)

    print("\n  Training set = ", str(X_train.shape))
    print("Validation set = ", str(X_valid.shape))
    print("      Test set = ", str(X_test.shape) + "\n")

    y_sum = np.append(y_train, y_valid, axis=0)
    y_count = np.sum(y_sum, axis=0, dtype=np.int32)
    y_cnt = np.round(np.max(y_count) / y_count, 4)
    keys = range(len(y_cnt))
    class_weights = dict(zip(keys, y_cnt))

    

    model = create_model()

    # Different optimizers to test
    # optimizer = optimizers.SGD(lr=0.01)
    optimizer = optimizers.SGD(momentum=0.001, nesterov=True)
    # optimizer = optimizers.adagrad
    # optimizer = optimizers.adadelta
    # optimizer = optimizers.rmsprop
    # optimizer = optimizers.adamax
    # optimizer = optimizers.nadam
    # optimizer = optimizers.adam(amsgrad=True)


    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer,loss,metrics)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_acc', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
    
    save_best_weights = ModelCheckpoint('full_model_weights2.h5',
        save_best_only=True,
        monitor='val_acc',
        mode='max',
        save_weights_only=True)

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(X_valid, y_valid),
        class_weight=class_weights,
        callbacks=[reduce_lr, save_best_weights]
    )

    scores = model.evaluate(X_test, y_test)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    print("\nDisplaying accuracy curves...")

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.savefig('SGD_Nesterov2_return.png')
    plt.show()

    print("\nSaving model and weights...")

    # serialize model to JSON
    model_json = model.to_json()
    with open("full_model2.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    # model.save_weights("full_model_weigths.h5")
    print("\nModel and weights saved to disk.\n")


if __name__ == "__main__":
    main(sys.argv[1:])
import tensorflow as tf
import keras_vggface
from keras import models
from keras import layers
from keras.utils import np_utils
from keras import optimizers
import pickle
from keras import backend as Kb
from keras.models import model_from_json
dataset_path = 'final.pickle'


with open(dataset_path, 'rb') as pickled_dataset:
    data_obj = pickle.load(pickled_dataset)

(training_data, validation_data, test_data) = data_obj['training_data'], data_obj['validation_data'], data_obj['test_data']
(X_train, y_train), (X_test, y_test) = (training_data[0],training_data[1]), (test_data[0],test_data[1])

batch_size = 30
nb_epochs = 30
nb_classes = 8
img_rows, img_cols = data_obj['img_dim']['width'], data_obj['img_dim']['height']

if Kb.image_data_format() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Load the VGG model
vgg_conv = keras_vggface.VGGFace(weights='vggface', include_top=False, input_shape=(img_rows, img_cols, 3))
# Freeze the layers except the last 4. The last 4 will be trained towards our goal, emotion-detection.
# json_file = open('model.json', 'r')
# loaded_model = json_file.read()
# json_file.close()
# vgg_conv = model_from_json(loaded_model)
# vgg_conv.load_weights("weights.h5")
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Create the model
model = models.Sequential()

# Add the vggface model
model.add(vgg_conv)

# Add new layers to adapt for CK+
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(nb_classes, activation='softmax'))

opt=optimizers.SGD(lr=0.01)
model.compile(loss='mean_squared_logarithmic_error',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Freeze layers to avoid further problems with nested model
for layer in model.layers:
    layer.trainable = False
# Save model and weights
model_json = model.to_json()
with open("model.json", "w") as jsonFile:
    jsonFile.write(model_json)

model.save_weights('weights.h5')
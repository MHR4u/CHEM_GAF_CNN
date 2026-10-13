# FROM: https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8

# https://keras.io/api/applications/
# https://keras.rstudio.com/articles/applications.html
# https://github.com/heuritech/convnets-keras#get-the-weights-of-the-pre-trained-networks


import os
import glob
import argparse
from time import time
from keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Activation, MaxPooling2D, \
  Conv2D, ZeroPadding2D
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
#from tf.keras.applications.EfficientNetB0 import EfficientNetB0, preprocess_input

# Case of fragment manually generated
#train_data_dir = "/home/mario/Scrivania/keras_transferlearning/data_avila/train"
#validation_data_dir = "/home/mario/Scrivania/keras_transferlearning/data_avila/test"
#from keras_applications.nasnet import NASNetLarge
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from datetime import datetime
#from tensorflow import keras

# IMPORTANT: example of transfer learning with last layer with new classifier
# https://gogul09.github.io/software/flower-recognition-deep-learning
from tensorflow.python.keras.regularizers import l2



def simple_cnn(img_shape=(224, 224, 3), n_classes=2):
	cnn = tf.keras.models.Sequential()
	cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=img_shape))
	cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
	cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'))
	cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
	cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'))
	cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
	cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'))
	cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
	cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'))
	cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
	cnn.add(tf.keras.layers.Conv2D(filters=8 , kernel_size=3, activation='relu'))
	cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
	cnn.add(tf.keras.layers.Flatten())
	cnn.add(tf.keras.layers.Dense(128, activation='relu'))
	cnn.add(Dense(64, activation='relu'))
	cnn.add(Dense(n_classes))
	cnn.add(BatchNormalization())
	cnn.add(Activation('softmax'))

	# finally compile and train the cnn
	cnn.compile(optimizer = SGD(learning_rate=0.001, momentum=0.95), loss='categorical_crossentropy', metrics=['accuracy'])

	return cnn


def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt



def plot_training(history):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r-', color='blue')
  plt.xlabel("epochs")
  plt.ylabel("accuracy")
  plt.title('Training and validation accuracy')

  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-', color='blue')
  plt.xlabel("epochs")
  plt.ylabel("loss")
  plt.title('Training and validation loss')
  plt.show()

def evalAccuracy(confusionMatrix):
    accuracy = 0
    accuracyByRow = []
    tot = 0
    for i in range(len(confusionMatrix)):
        totRow = 0
        for j in range(len(confusionMatrix)):
           tot = tot + confusionMatrix[i][j]
           totRow = totRow + confusionMatrix[i][j]
        accuracy = accuracy + confusionMatrix[i][i]
        accuracyByRow.append((confusionMatrix[i][i] * 100.0)/totRow)



    accuracy = (100.0 * accuracy)/tot

    return accuracy, accuracyByRow

def train(args, img_width, img_height):
    # Initiate the train and test generators with data Augumentation
    # PARAMETERS FROM COMMAND LINE
    base_dir = args.base_dir
    nb_epochs = int(args.nb_epoch)
    batch_size = int(args.batch_size)
    patience = int(args.patience)


    print("INPUT PARAMETERS: ")
    print("base_dir: " + base_dir)
    print("nb_epochs: " + str(nb_epochs))
    print("batch_size: " + str(batch_size))
    print("patience: " + str(patience))
    print("******************************************************")
    

    train_data_dir = base_dir + "/Train"
    validation_data_dir = base_dir + "/Val"
    nb_classes = len(glob.glob(train_data_dir + "/*"))

    learningModelName = "CNN_" + str(nb_classes) + "classes_" + "_epo_" + str(nb_epochs) + "_pat_" + str(patience) + "_batch_" + str(batch_size) + "_tl.keras"


    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,        
	      vertical_flip=True,
        fill_mode="nearest",

    )

    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
	      vertical_flip=True,
        fill_mode="nearest",
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical")

    validation_generator = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        class_mode="categorical")

    model = simple_cnn(n_classes=nb_classes)

    print("*********************************************************************************************")
    print("*********************************************************************************************")
    print("                                LEARNING - " + learningModelName)
    print("*********************************************************************************************")
    print("*********************************************************************************************")

    checkpoint = ModelCheckpoint(learningModelName, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', save_freq='epoch')
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')
    tensorboard = TensorBoard(log_dir="logs_transfer/{}".format(time()))

    print(model.summary())



    history_tl = model.fit(
        train_generator,
        epochs=nb_epochs,
        validation_data=validation_generator,
        class_weight=None,
        callbacks=[checkpoint, early])

    print("*********************************************************************************************")
    print("*********************************************************************************************")
    print("                          LEARNING COMPLETED - " + learningModelName)
    print("*********************************************************************************************")
    print("*********************************************************************************************")



    if args.plot:
        plot_training(history_tl)



    return model, learningModelName, img_width, img_height


def test(args, model, learningModelName, img_width, img_height):

    print("*********************************************************************************************")
    print("*********************************************************************************************")
    print("                                TESTING WITH - " + learningModelName)
    print("*********************************************************************************************")
    print("*********************************************************************************************")

    target_names = ["Benzochinone", "Catecolo", "Ferro-Ferri", "Idrochinone", "Idrochinone-Benzochinone"] # Tor Vergata

    base_dir = args.base_dir
    test_data_dir = base_dir + "/Test"
    

    nb_test_samples = get_nb_files(test_data_dir)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        color_mode="rgb",
        shuffle=False,
        class_mode='categorical',
        batch_size=1)
    model.load_weights(learningModelName)

    print("Prediction of " + str(nb_test_samples) + " rows.")
    probabilities = model.predict(test_generator, nb_test_samples)
    print(probabilities)
    print(test_generator.classes)
    probabilities = np.argmax(probabilities, axis=1)



    print('Confusion Matrix')
    cm = confusion_matrix(test_generator.classes, probabilities)
    print(cm)

    print('(On the row true label, on the column predicted label)')
    print('(Please, pay attention to the order of the label, that should be the alphabetic order)')

    accuracy, accuracyByRow = evalAccuracy(cm)
    print("Accuracy: " + str(accuracy))
    print("Accuracy by row:")

    for i in range(0, len(accuracyByRow)):
        print("\t" + target_names[i] + " => " + str(accuracyByRow[i]))
    print('Classification Report')

    print(classification_report(test_generator.classes, probabilities, target_names=target_names))

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--base_dir", default=".")
    a.add_argument("--nb_epoch", default=1000)
    a.add_argument("--batch_size", default=32)
    a.add_argument("--patience", default=50)
    a.add_argument("--plot", action="store_true")
    args = a.parse_args()

    img_height, img_widt = 224, 224


    model, learningModelName, img_width, img_height = train(args, img_height, img_widt)

    test(args, model, learningModelName, img_width, img_height)


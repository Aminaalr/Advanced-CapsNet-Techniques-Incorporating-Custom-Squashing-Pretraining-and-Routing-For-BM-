import time
start_time = time.time()
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Reshape, Flatten, Lambda, Multiply
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras import callbacks
from capsUtils import plotLog

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import time
import datetime
from capsUtils import plotLog
def conditional_prelu(inputs):
  features, predictions = inputs  # Split input into features and predictions
  pos = K.relu(features)
  neg = predictions * K.relu(-features)
  return pos + neg

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

fix_gpu()

# Image Directory Location
pathImg = 'images'

# Image Size
image_size = 75



# Cost-Sensitive Learning for Imbalanced Classification
def OverSample(imgArr, labelArr):
    strategy = {0: 5000, 1: 20000, 2: 5000, 3: 12000, 4: 28000, 5: 6000, 6: 5000, 7: 5000, 8: 5000, 9: 5000, 10: 27000,
                11: 5000, 12: 5000, 13: 8000, 14: 10000, 15: 30000, 16: 5000, 17: 5000, 18: 5000, 19: 8000, 20: 12000}
    oversample = SMOTE(sampling_strategy=strategy)
    x1 = imgArr.shape[1]
    x2 = imgArr.shape[2]
    x3 = imgArr.shape[3]
    # Reshape
    imgArr = (imgArr.reshape(imgArr.shape[0], x1 * x2 * x3))
    imgArr, labelArr = oversample.fit_resample(imgArr, labelArr)
    # Reshape
    imgArr = (imgArr.reshape(imgArr.shape[0], x1, x2, x3))
    return imgArr, labelArr


# Normalization of Data
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Loading the Dataset
def loadDataset():
    # the data, shuffled and split between train and test sets
    imgArr = []
    image_label = []
    class_names = []
    dirList = [f for f in listdir(pathImg) if not isfile(join(pathImg, f))]
    print(dirList)

    for i in range(len(dirList)):
        fileList = list()
        for (dirpath, dirnames, filenames) in os.walk(pathImg + '/' + dirList[i]):
            fileList += [os.path.join(dirpath, file) for file in filenames]
        print(dirList[i], len(fileList))
        for filename in fileList:
            if (filename.endswith('.jpg')):
                try:
                    imgLoad = Image.open(filename)
                    resImg = imgLoad.resize((image_size, image_size), Image.BICUBIC)
                    numImg = (np.array(resImg)).astype('float64')
                    normImg = NormalizeData(numImg) * ((i + 1) / len(dirList))
                    imgArr.append(normImg)
                    image_label.append(i)
                    class_names.append(dirList[i])
                except:
                    print('Problem in File : ', filename)

    print(len(imgArr))
    imgArr = np.array(imgArr)
    classNames = sorted(set(class_names), key=class_names.index)
    labelArr = to_categorical(np.array(image_label).astype('float32'))

    # SMOTE Over Sample
    imgArr, labelArr = OverSample(imgArr, labelArr)
    labelArr = np.array(labelArr).astype('float32')

    # Split the data into training, validation, and test sets
    x_train, x_temp, y_train, y_temp = train_test_split(imgArr, labelArr, test_size=0.3, random_state=2,
                                                        stratify=labelArr)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=2 / 3, random_state=2, stratify=y_temp)

    print('Read complete')
    print(len(x_train))
    print(len(x_test))
    return (x_train, y_train), (x_test, y_test), classNames


# Performance Matrics
def performance_metrics(cnf_matrix, class_names):
    # Confusion Matrix Plot
    cmd = ConfusionMatrixDisplay(cnf_matrix, display_labels=class_names)
    cmd.plot(cmap='Greens')
    cmd.ax_.set(xlabel='Predicted', ylabel='Actual')
    # Find All Parameters
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # Overall accuracy for each class
    ACC = 2 * (PPV * TPR) / (PPV + TPR)
    print('\n\nClassName\tTP\tFP\tFN\tTN\tPrecision\tSensitivity\tSpecificity\tAccuracy')
    for i in range(len(class_names)):
        print(class_names[i] + "\t\t{0:.0f}".format(TP[i]) + "\t{0:.0f}".format(FP[i]) + "\t{0:.0f}".format(
            FN[i]) + "\t{0:.0f}".format(TN[i]) + "\t{0:.4f}".format(PPV[i]) + "\t\t{0:.4f}".format(
            TPR[i]) + "\t\t{0:.4f}".format(TNR[i]) + "\t\t{0:.4f}".format(ACC[i]))


def squash(vectors, axis=-1):
    # Non-linear activation function (squashing)
    s_squared_norm = tf.reduce_sum(tf.square(vectors / 5), axis, keepdims=True)
    # s_squared_norm2= tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = (0.5 * s_squared_norm) / (1 + 0.5 * s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    # Layer normalization
    epsilon = 1e-6
    mean, variance = tf.nn.moments(vectors, axes=-1, keepdims=True)
    vectors = (vectors - mean) / tf.sqrt(variance + epsilon)

    return scale * vectors


# Build Model for GoogLeNet with CapsuleNet
def Gnet_capsNet_model(n_routings=3):
    # Initialize the GoogLeNet Model
    GNnet = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet",
                                              input_shape=(image_size, image_size, 3))
    # CapsuleNet as primary Capsule Convoluttions
    xCpas = Conv2D(filters=8 * 32, kernel_size=1, strides=2, padding='valid', name='primarycap_conv2')(GNnet.output)
    # Resahpe the Primary Capsule
    xCpas = Reshape(target_shape=[-1, 8], name='primarycap_reshape')(xCpas)
    # Squash for Primary Capsule
    xCpas = Lambda(squash, name='primarycap_squash')(xCpas)
    # Normzalization of Momentum as 0.8
    xCpas = BatchNormalization(momentum=0.8)(xCpas)
    # Flattern Layer
    xCpas = Flatten()(xCpas)
    # Intialization of Dense
    GNnet_Caps = Dense(160, kernel_initializer='he_normal', bias_initializer='zeros', name='uhat_digitcaps')(xCpas)
    # Repeat Dense for Number of Routings
    for i in range(n_routings):
        # Activation of Softmax
        cSoft = Activation('softmax', name='softmax_digitcaps' + str(i))(GNnet_Caps)
        # Assign Dense as 160
        cSoft = Dense(160)(cSoft)
        cSoft = Lambda(squash, name='squash2')(cSoft)
        # Multiply
        GNnetCaps = Multiply()([GNnet_Caps, cSoft])
        # conditional_prelu
        sJ = Lambda(conditional_prelu)([ GNnetCaps, cSoft])
    # Output Dense as 32
    GNnetCaps = Dense(32, activation='relu')(sJ)
    # Predict Model
    pred = Dense(len(classNames), activation='softmax')(GNnetCaps)
    return Model(GNnet.input, pred)


# load data
(x_train, y_train), (x_test, y_test), classNames = loadDataset()

GNnetCapsNet_Model = Gnet_capsNet_model()
GNnetCapsNet_Model.summary()
GNnetCapsNet_Model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
# Calling tf.config.run_functions_eagerly(True) will make all invocations of tf.function run eagerly instead of running as a traced graph function.
tf.config.run_functions_eagerly(True)

# callbacks
lr = 0.001
lr_decay = 0.9
log = callbacks.CSVLogger('./result' + '/log.csv')
checkpoint = callbacks.ModelCheckpoint('./result' + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                       save_best_only=True, save_weights_only=True, verbose=1)
lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (lr_decay ** epoch))

tensorboard = TensorBoard(log_dir='logs')
checkpoint = ModelCheckpoint("effnet.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)
history = GNnetCapsNet_Model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, verbose=1,
                                 batch_size=100, callbacks=[log, checkpoint, reduce_lr])

y_pred = GNnetCapsNet_Model.predict([x_test], batch_size=100)

# Confusion matrix
cm = confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1))
# Overall Performance
performance_metrics(cm, classNames)
plotLog('./result' + '/log.csv', showPlot=True)
print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")

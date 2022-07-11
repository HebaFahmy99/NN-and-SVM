# Import required libraries
import os
import cv2
import pandas as pd
import numpy as np
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

Dataset_dir = "E:/FCAI/semester 5/Neural Networks and Learning Machines/Assignment 2/Dataset"


def load_process_imgs(thresh, IMG_SIZE=100):  # Function for loading and processing the images
    Categories = ["Negative", "Positive"]
    features = []
    target = []
    for category in Categories:
        path = os.path.join(Dataset_dir, category)  # create path to "Negative","Positive"
        target_val = Categories.index(category)  # get the classification  (0 or a 1). 0=Negative , 1=Positive
        for img in os.listdir(path):  # iterate over each image per negative and postive
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            rez_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
            im_bin = (rez_img > thresh) * 255  # Binarize the input dataset with a suitable threshold value
            features.append(im_bin.flatten())
            target.append(target_val)
    return features, target


Features, Target = load_process_imgs(200, 150)

features = np.array(Features)
target = np.array(Target)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=0)


def SVM_Model(kernel, X_train, X_test, y_train, y_test):
    clf_SVM = SVC(C=10, kernel=kernel)
    clf_SVM.fit(X_train, y_train)
    y_pred = clf_SVM.predict(X_test)
    print(f"The SVM model with {kernel} kernel is {accuracy_score(y_pred, y_test) * 100}% accurate")


def NN_model_MLPClassifier(X_train, X_test, y_train, y_test, numOfHidden_neurons):
    mlp = MLPClassifier(hidden_layer_sizes=(numOfHidden_neurons), max_iter=1000)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    print(f"The MLPClassifier model is {accuracy_score(y_pred, y_test) * 100}% accurate")


def NN_Keras_Model(X_train, X_test, y_train, y_test, numOfHidden_neurons):
    # one hot encode outputs
    y_train = to_categorical(y_train)
    model = Sequential()
    model.add(Dense(500, activation='relu', input_dim=22500))
    model.add(Dense(numOfHidden_neurons, activation='relu'))
    #     model.add(Dense(numOfHidden_neurons, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # build the model
    model.fit(X_train, y_train, epochs=20)
    y_pred = model.predict_classes(X_test)
    print(f"The NN keras model is {accuracy_score(y_pred, y_test) * 100}% accurate")


# test case 1 for SVM Model with different kernels (rbf,poly,linear,sigmoid)
SVM_Model('rbf', X_train, X_test, y_train, y_test)
SVM_Model('poly', X_train, X_test, y_train, y_test)
SVM_Model('linear', X_train, X_test, y_train, y_test)
SVM_Model('sigmoid', X_train, X_test, y_train, y_test)

# test case 2 for NN model from MLPClassifier with different number Of hidden neurons
NN_model_MLPClassifier(X_train, X_test, y_train, y_test, 12)
NN_model_MLPClassifier(X_train, X_test, y_train, y_test, 44)
NN_model_MLPClassifier(X_train, X_test, y_train, y_test, 86)

# test case 3 for NN model from keras with different number of hidden neurons and hidden layer
NN_Keras_Model(X_train, X_test, y_train, y_test, numOfHidden_neurons=50)
NN_Keras_Model(X_train, X_test, y_train, y_test, numOfHidden_neurons=90)
NN_Keras_Model(X_train, X_test, y_train, y_test, numOfHidden_neurons=128)
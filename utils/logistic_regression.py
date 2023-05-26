#!/usr/bin/env python

"""
Logistic regression script
"""

## ---- DEPENDENCIES ---- ##
import os
import sys
sys.path.append(os.path.join(".."))
import tensorflow as tf
from tensorflow import keras
from utils.model_utils import *

## --- LOGISTIC REGRESSION CLASS --- ##

class LogisticRegression():
    def __init__(self, optimizer, learning_rate, batch_size, epochs):

        # define hyperparameters
        self.optimizer=optimizer
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.epochs=epochs

        # define optimization algorithm based on input
        if optimizer == "adam":
            self.opt_lr = tf.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "sgd":
            self.opt_lr = tf.optimizers.SGD(learning_rate=learning_rate)

        # build model
        self.model = tf.keras.Sequential(
            [keras.layers.Dense(units=1, 
            activation='sigmoid')])
        
        # compile model
        self.model.compile(
            optimizer=self.opt_lr,
            loss='binary_crossentropy',
            metrics=['binary_accuracy'])
        
    def train(self, X_train, y_train, X_val, y_val, verbose=0):
        """
        Train model using hyperparameters, and validate on validation data.
        """
        # train model
        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=verbose,
            validation_data=(X_val, y_val))
        
        # get train and val loss after last epoch
        self.train_loss = self.history.history["loss"][-1]
        self.val_loss = self.history.history["val_loss"][-1]

        return self.history

    def get_weights(self):
        weights = []
        for layer in self.model.layers: 
            weights.append(layer.get_weights()) # a dense layer returns both the kernel matrix and the bias vector
        return weights

    def evaluate(self, X_test, y_test):
        """
        Evaluates model on a test data
        """
        # compute test loss
        self.test_loss = self.model.evaluate(X_test, 
                                             y_test, 
                                             verbose=0)
        
        print(f"[INFO] Test loss: {self.test_loss}")

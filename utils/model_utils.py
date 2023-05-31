#!/usr/bin/env python

"""
Utility functions for all models: baseline (logistic regression models), XGBoost models, and Transformer models.
"""

# ---- DEPENDENCIES ---- #
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from datetime import datetime
import numpy as np
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoModel, AutoConfig, Trainer
import itertools
from datasets import Dataset, DatasetDict
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

# ---- UTILITY FUNCTIONS ---- #
def split_equal_test_size_task(X, y, test_size):
    """
    Returns a test dataset that contains an equal number of each class 0 and 1 and equal number of chronic vs. first episode depressed.

    Args:
        - X (df): data
        - y (df): labels
        - test_size (float): size of test split

    Returns:
        - X_train (df): training split
        - X_test (df): test split
        - y_train (df): training labels
        - y_test (df): test labels
    """
    # define number of samples
    samples_n = round(len(X)*test_size/2)
    first_ep_samples = round(samples_n/2)
    first_ep_count = 0
    
    # create lists to be appended to
    indicesClass1 = []
    indicesClass2 = []
    
    # append to indices lists based on samples_n
    for i in range(0, len(y)):
        if y[i] == 0 and len(indicesClass1) < samples_n:
            indicesClass1.append(i)
        elif y[i] == 1 and len(indicesClass2) < samples_n:
            if X["Diagnosis"][i]=="1st_episode" and first_ep_count < first_ep_samples:
                indicesClass2.append(i)
                first_ep_count = first_ep_count + 1
            elif X["Diagnosis"][i]=="chronic":
                indicesClass2.append(i)
            else:
                pass
        if len(indicesClass1) == samples_n and len(indicesClass2) == samples_n:
            break
  
    # define class1 and class2
    X_test_class1 = X.iloc[indicesClass1]
    X_test_class2 = X.iloc[indicesClass2]
    
    # concatenate classes
    X_test = pd.concat([X_test_class1, X_test_class2])
    
    # remove x_test from X
    X_train = X.drop(indicesClass1 + indicesClass2)
    
    # define labels for classes
    Y_test_class1 = y[indicesClass1]
    Y_test_class2 = y[indicesClass2]
    
    # concatenate labels
    y_test = np.concatenate((Y_test_class1, Y_test_class2), axis=0)
    
    # remove y_test from y
    df_y = pd.DataFrame(y)
    y_train = df_y.drop(indicesClass1 + indicesClass2)
    
    return X_train, X_test, y_train, y_test

def split_equal_test_size_patient(X, y, test_size):
    """
    Returns a test dataset that contains an equal number of each class 0 and 1 and equal number of chronic vs. first episode depressed.

    Args:
        - X (df): data
        - y (df): labels
        - test_size (float): size of test split

    Returns:
        - X_train (df): training split
        - X_test (df): test split
        - y_train (df): training labels
        - y_test (df): test labels
    """
    # define number of samples
    samples_n = round(len(X)*test_size/2)
    triangle_samples = round(samples_n/2)
    triangle_count = 0
    triangle_count_control = 0
    autobiographical_count_control = 0
    autobiographical_count_depressed = 0

    # create lists to be appended to
    indicesClass1 = []
    indicesClass2 = []
    
    # append to indices lists based on samples_n
    for i in range(0, len(y)):
        if y[i] == 0 and len(indicesClass1) < samples_n:
            if X["data_type"][i]=="Triangle" and triangle_count_control < triangle_samples:
                indicesClass1.append(i)
                triangle_count_control = triangle_count_control + 1
            elif X["data_type"][i]=="Autobiographical" and autobiographical_count_control < (samples_n - triangle_samples):
                indicesClass1.append(i)
                autobiographical_count_control = autobiographical_count_control + 1  
        elif y[i] == 1 and len(indicesClass2) < samples_n:
            if X["data_type"][i]=="Triangle" and triangle_count < triangle_samples:
                indicesClass2.append(i)
                triangle_count = triangle_count + 1
            elif X["data_type"][i]=="Autobiographical" and autobiographical_count_depressed < (samples_n - triangle_samples):
                indicesClass2.append(i)
                autobiographical_count_depressed = autobiographical_count_depressed + 1
            else:
                pass
        if len(indicesClass1) == samples_n and len(indicesClass2) == samples_n:
            break
  
    # define class1 and class2
    X_test_class1 = X.iloc[indicesClass1]
    X_test_class2 = X.iloc[indicesClass2]
    
    # concatenate classes
    X_test = pd.concat([X_test_class1, X_test_class2])
    
    # remove x_test from X
    X_train = X.drop(indicesClass1 + indicesClass2)
    
    # define labels for classes
    Y_test_class1 = y[indicesClass1]
    Y_test_class2 = y[indicesClass2]
    
    # concatenate labels
    y_test = np.concatenate((Y_test_class1, Y_test_class2), axis=0)
    
    # remove y_test from y
    df_y = pd.DataFrame(y)
    y_train = df_y.drop(indicesClass1 + indicesClass2)
    
    return X_train, X_test, y_train, y_test

def split_equal_test_size_patient_transformer(X, y, test_size):
    """
    Returns a test dataset that contains an equal number of each class 0 and 1 and equal number of chronic vs. first episode depressed.

    Args:
        - X (df): data
        - y (df): labels
        - test_size (float): size of test split

    Returns:
        - X_train (df): training split
        - X_test (df): test split
        - y_train (df): training labels
        - y_test (df): test labels
    """
   # define number of samples
    samples_n = round(len(X)*test_size/2)
    triangle_samples = round(samples_n/2)
    triangle_count = 0
    triangle_count_control = 0
    autobiographical_count_control = 0
    autobiographical_count_depressed = 0

    # create lists to be appended to
    indicesClass1 = []
    indicesClass2 = []

    # append to indices lists based on samples_n
    for i in range(0, len(y)):
        if y[i] == 0 and len(indicesClass1) < samples_n:
            print(X["data_type"][i])
            if X["data_type"][i]=="Triangle" and triangle_count_control < triangle_samples:
                indicesClass1.append(i)
                triangle_count_control = triangle_count_control + 1
            elif X["data_type"][i]=="Autobiographical" and autobiographical_count_control < (samples_n - triangle_samples):
                indicesClass1.append(i)
                autobiographical_count_control = autobiographical_count_control + 1
        elif y[i] == 1 and len(indicesClass2) < samples_n:
            if X["data_type"][i]=="Triangle" and triangle_count < triangle_samples:
                indicesClass2.append(i)
                triangle_count = triangle_count + 1
            elif X["data_type"][i]=="Autobiographical" and autobiographical_count_depressed < (samples_n - triangle_samples):
                indicesClass2.append(i)
                autobiographical_count_depressed = autobiographical_count_depressed + 1
            else:
                pass
        if len(indicesClass1) == samples_n and len(indicesClass2) == samples_n:
            break

    # define class1 and class2
    X_test_class1 = X.iloc[indicesClass1]
    X_test_class2 = X.iloc[indicesClass2]

    # concatenate classes
    X_test = pd.concat([X_test_class1, X_test_class2])

    # remove x_test from X
    X_train = X.drop(indicesClass1 + indicesClass2)

    # define labels for classes
    Y_test_class1 = y[indicesClass1]
    Y_test_class2 = y[indicesClass2]

    # concatenate labels
    y_test = np.concatenate((Y_test_class1, Y_test_class2), axis=0)

    # remove y_test from y
    df_y = pd.DataFrame(y)
    y_train = df_y.drop(indicesClass1 + indicesClass2)
    
    return X_train, X_test, y_train, y_test

def split_equal_test_size_task_transformer(X, y, test_size):
    """
    Returns a test dataset that contains an equal number of each class 0 and 1 and equal number of chronic vs. first episode depressed.

    Args:
        - X (df): data
        - y (df): labels
        - test_size (float): size of test split

    Returns:
        - X_train (df): training split
        - X_test (df): test split
        - y_train (df): training labels
        - y_test (df): test labels
    """
    # define number of samples
    samples_n = round(len(X)*test_size/2)
    chronic_samples = round(samples_n/2)
    chronic_count = 0
    
    # create lists to be appended to
    indicesClass1 = []
    indicesClass2 = []
    
    # append to indices lists based on samples_n
    for i in range(0, len(y)):
        if y[i] == 0 and len(indicesClass1) < samples_n:
            indicesClass1.append(i)
        elif y[i] == 1 and len(indicesClass2) < samples_n:
            if X["Diagnosis"][i]=="chronic" and chronic_count < chronic_samples:
                indicesClass2.append(i)
                chronic_count = chronic_count + 1
            elif X["Diagnosis"][i]=="1st_episode":
                indicesClass2.append(i)
            else:
                pass
        if len(indicesClass1) == samples_n and len(indicesClass2) == samples_n:
            break
  
    # define class1 and class2
    X_test_class1 = X.iloc[indicesClass1]
    X_test_class2 = X.iloc[indicesClass2]
    
    # concatenate classes
    X_test = pd.concat([X_test_class1, X_test_class2])
    
    # remove x_test from X
    X_train = X.drop(indicesClass1 + indicesClass2)
    
    # define labels for classes
    Y_test_class1 = y[indicesClass1]
    Y_test_class2 = y[indicesClass2]
    
    # concatenate labels
    y_test = np.concatenate((Y_test_class1, Y_test_class2), axis=0)
    
    # remove y_test from y
    df_y = pd.DataFrame(y)
    y_train = df_y.drop(indicesClass1 + indicesClass2)
    
    return X_train, X_test, y_train, y_test

def split_equal_test_size(X, y, test_size):
    """
    Returns a test dataset that contains an equal number of each class 0 and 1.

    Args:
        - X (df): data
        - y (df): labels
        - test_size (float): size of test split

    Returns:
        - X_train (df): training split
        - X_test (df): test split
        - y_train (df): training labels
        - y_test (df): test labels
    """
    # define number of samples
    samples_n = round(len(X)*test_size/2)
    
    # create lists to be appended to
    indicesClass1 = []
    indicesClass2 = []
    
    # append to indices lists based on samples_n
    for i in range(0, len(y)):
        if y[i] == 0 and len(indicesClass1) < samples_n:
            indicesClass1.append(i)
        elif y[i] == 1 and len(indicesClass2) < samples_n:
            indicesClass2.append(i)
        if len(indicesClass1) == samples_n and len(indicesClass2) == samples_n:
            break
    
    # define class1 and class2
    X_test_class1 = X.iloc[indicesClass1]
    X_test_class2 = X.iloc[indicesClass2]
    
    # concatenate classes
    X_test = pd.concat([X_test_class1, X_test_class2])
    
    # remove x_test from X
    X_train = X.drop(indicesClass1 + indicesClass2)
    
    # define labels for classes
    Y_test_class1 = y[indicesClass1]
    Y_test_class2 = y[indicesClass2]
    
    # concatenate labels
    y_test = np.concatenate((Y_test_class1, Y_test_class2), axis=0)
    
    # remove y_test from y
    df_y = pd.DataFrame(y)
    y_train = df_y.drop(indicesClass1 + indicesClass2)
    
    return X_train, X_test, y_train, y_test

def make_train_test_split_xgboost(data, test_size, seed, task_angle=None, patient_angle=None):
    """
    Makes train and test splits for XGBoost models.

    Args:
        - data (df): dataframe containing all data
        - test_size (float): size of test split
        - seed (int): random_state for reproducinility

    Returns:
        - X_train (df): training split
        - X_test (df): test split
        - y_train (df): training labels
        - y_test (df): test labels
    """
    # shuffle data
    for col in data.columns:
        np.random.seed(seed)
        np.random.shuffle(data[col].values)

    # define X and y
    X = data.loc[:, data.columns != "binary_label"]
    y = data["binary_label"]
    
    if task_angle == True and patient_angle != True:
        # split into train and test using split_equal_size function to make test-set equally balanced, i.e. 50/50 depressed and controls
        X_train, X_test, y_train, y_test = split_equal_test_size_task(X, y, test_size)
        print("Task_angle")
       
    if (patient_angle is None or patient_angle == False) and task_angle != True:
        X_train, X_test, y_train, y_test = split_equal_test_size(X, y, test_size)
        
    elif patient_angle == True and task_angle != True:
        # split into train and test using split_equal_size function to make test-set equally balanced, i.e. 50/50 depressed and controls
        X_train, X_test, y_train, y_test = split_equal_test_size_patient(X, y, test_size)

    return X_train, X_test, y_train, y_test

def prepare_transformer_data(data, test_size, val_size, task_angle=None, patient_angle=None):
    """
    Prepares the train, val, and test split for the transformer models.

    Args:
        - data (df): dataframe containing all data and labels
        - test_size (float): size of test split
        - val_size (float): size of val split

    Returns:
        - data (DatasetDict): datasetobject containing data prepared for transformer model.
    """
    # define X and y
    X = data
    y = data["binary_label"]
    
    if (task_angle is None or task_angle == False) and patient_angle != True:
        X_train, X_test, y_train, y_test = split_equal_test_size(X, y, test_size)
        
    elif task_angle == True and patient_angle != True:
        # split into train and test using split_equal_size function to make test-set equally balanced, i.e. 50/50 depressed and controls
        X_train, X_test, y_train, y_test = split_equal_test_size_task_transformer(X, y, test_size)
        print("task_angle true")
    
    if patient_angle == True and task_angle != True:
        # split into train and test using split_equal_size function to make test-set equally balanced, i.e. 50/50 depressed and controls
        X_train, X_test, y_train, y_test = split_equal_test_size_patient_transformer(X, y, test_size)
        print("patient_angle true")
    
    # reset index on X_train and y_train 
    y_train.reset_index(inplace=True, drop=True)
    X_train.reset_index(inplace=True, drop=True)

    # convert y_train to array
    y_train = y_train["binary_label"]
    
    
    if (task_angle is None or task_angle == False) and patient_angle != True:
        # split into train and val (where val-set is equally balanced, i.e. 50/50 depressed and controls)
        X_train, X_val, y_train, y_val = split_equal_test_size(X_train, 
                                                           y_train, 
                                                           val_size)
        
    elif task_angle == True and patient_angle != True:
        # split into train and test using split_equal_size function to make test-set equally balanced, i.e. 50/50 depressed and controls
        X_train, X_val, y_train, y_val = split_equal_test_size_task_transformer(X_train, y_train, val_size)
        print("task_angle true for val")
    
    if (patient_angle is None or patient_angle == False) and task_angle != True:
        # split into train and val (where val-set is equally balanced, i.e. 50/50 depressed and controls)
        X_train, X_val, y_train, y_val = split_equal_test_size(X_train, 
                                                                y_train, 
                                                                val_size)
        
    elif patient_angle == True:
        # split into train and test using split_equal_size function to make test-set equally balanced, i.e. 50/50 depressed and controls
        X_train, X_val, y_train, y_val = split_equal_test_size_patient_transformer(X_train, y_train, val_size)
        print("patient_angle true for val")

    # convert all data (X_train, X_val, X_test) to dataset which is required for transformer input
    X_train = X_train[["Filename", "ID", "binary_label", "Transcript"]]
    X_train = X_train.rename(columns = {"binary_label": "label", "Transcript": "transcript"})
    X_train = X_train.reset_index(drop=True)
    X_train_dataset = Dataset.from_pandas(X_train)

    X_val = X_val[["Filename", "ID", "binary_label", "Transcript"]]
    X_val = X_val.rename(columns = {"binary_label": "label", "Transcript": "transcript"})
    X_val = X_val.reset_index(drop=True)
    X_val_dataset = Dataset.from_pandas(X_val)

    X_test = X_test[["Filename", "ID", "binary_label", "Transcript"]]
    X_test = X_test.rename(columns = {"binary_label": "label", "Transcript": "transcript"})
    X_test = X_test.reset_index(drop=True)
    X_test_dataset = Dataset.from_pandas(X_test)

    # gather train, val, and test in datasetDict object
    data = DatasetDict({
        'train': X_train_dataset,
        'test': X_test_dataset,
        'valid': X_val_dataset})

    return data

def make_data_loader_objects(tokenized_dataset, data_collator):
    """
    Creates the data loader objects for transformer models.

    Args:
        - tokenized_dataset (DatasetDict): the tokenized data
        - data_collator (DataCollator): datacollator object

    Returns:
        - train_dataloader: dataloader object for training
        - eval_dataloader: dataloader object for validation
        - test_dataloader: datalodaer object for test
    """
    # create train dataloader object
    train_dataloader = DataLoader(
        tokenized_dataset["train"], 
        shuffle=True, 
        batch_size=10, 
        collate_fn=data_collator)
    
    # create validation dataloader object
    eval_dataloader = DataLoader(
        tokenized_dataset["valid"], 
        batch_size=32, 
        collate_fn=data_collator)

    # create test dataloader object
    test_dataloader = DataLoader(
        tokenized_dataset["test"], 
        batch_size=32, 
        collate_fn=data_collator)

    return train_dataloader, eval_dataloader, test_dataloader

def split_equal_val_size_cv(X, y, val_size):
    """
    Returns a val dataset that contains an equal number of each class specifically for cross-validation used for XGBoost models.

    Args:
        - X (df): data
        - y (df): labels
        - val_size: size of validation split

    Returns:
        - X_train: training data 
        - y_train: training labels
        - X_val: validation data
        - y_val: validation labels
    """
    # define number of samples
    samples_n = round(len(y)*val_size/2)
    
    # create lists to be appended to
    indicesClass1 = []
    indicesClass2 = []
    index_training = []

    # append to indices lists
    for index, label in enumerate(y["binary_label"].values):
        if label == 0 and len(indicesClass1) < samples_n:
            indicesClass1.append(index)
        elif label == 1 and len(indicesClass2) < samples_n:
            indicesClass2.append(index)
        else:
            index_training.append(index)
    
    # define class1 and class2
    X_val_class1 = X[indicesClass1]
    X_val_class2 = X[indicesClass2]

    # concatenate classes
    X_val = np.concatenate([X_val_class1, X_val_class2])
        
    # remove x_val from X
    X_train = X[index_training]

    # define labels for classes
    Y_val_class1 = y.iloc[indicesClass1]
    Y_val_class2 = y.iloc[indicesClass2]
    
    # concatenate labels
    y_val = np.concatenate((Y_val_class1,Y_val_class2), axis=0)
    
    # remove y_test from y
    y_train = y.iloc[index_training]

    return X_train, y_train, X_val, y_val

def make_train_val_test_split(data, test_size, val_size, text_features, seed):
    """
    Creates a train-val-test split.
    
    Args: 
      - features_path (str):path to .csv file containing low-level features
      - test_size (int): defining size of test set
      - val_size (int): defining size of val set
      - text_features (str): that defines the text features you want to train a model on
      - seed (int): that defines random state for reproducibility

    Returns:
      - X_train: training data
      - X_val: validation data
      - X_test: test data
      - y_train: training labels
      - y_val: validation labels
      - y_test: test labels
    """
    # define X and y
    X = data[text_features]
    y = data["binary_label"]

    # reset index
    X.reset_index(inplace=True, drop=True)
    y.reset_index(inplace=True, drop=True)
    
    # split into train and test (where test-set is equally balanced, i.e. 50/50 depressed and controls)
    X_train, X_test, y_train, y_test = split_equal_test_size(X, y, test_size)

    # reset index
    X_train.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)

    # split into train and val (where val-set is equally balanced, i.e. 50/50 depressed and controls)
    X_train, X_val, y_train, y_val = split_equal_test_size(X_train, np.array(y_train), val_size)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def plot_history(history, out_history, out_history_filename):
    """
    Plots model training and validation loss over epochs.

    Args:
        - history: model training history
        - out_history (str): defining path to output folder
        - out_history_filename (str): defining filename of plot

    Returns:
        - Saved .png history plot to specified output folder
    """
    # clear plt
    plt.clf()

    # plot line for train and validation loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    # plot line for train and validation accuracy
    plt.plot(history.history["binary_accuracy"], label='Train Accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')

    # limit y axis scale
    plt.ylim([0, 1])

    # labels of axes
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy [Binary]')

    # plot title
    plt.title('Training and Validation Loss / Accuracy over Epochs')

    # create a legend
    plt.legend()
    plt.grid(True)

    # save the figure
    plt.savefig(os.path.join(out_history, out_history_filename))

def write_model_results(model, model_type, optimizer, lr, batch_size, epochs, weights, out_path, filename, cl_report):
    """
    Writes model results which includes hyperparameters, training loss, validiation loss, test loss and classification report.

    Args:
        - model: the trained model
        - model_type (str): defining type of the trained model
        - optimizer (str): optimization algorithm used during training
        - lr (float): learning rate used during training
        - batch_size (int): size of batches used during training
        - epochs (int): number of epochs the model has been trained for
        - out_path (str): defining path to output folder
        - filename (str): defining name of results file
        - cl_report: classification report

    Returns:
        - Saves model results as .txt to output folder
    """

    # create output file and save information
    with open(os.path.join(out_path, filename), 'a') as file:
        file.write(f"MODEL USED FOR PREDICTIONS: {model_type}\n\n")
        file.write(f"{datetime.now()}\n\n")
        file.writelines([f"optimizer: {optimizer}\n",
                         f"lr: {lr}\n", 
                         f"batch_size: {batch_size}\n",
                         f"epochs: {epochs}\n",
                         f"train_loss: {model.train_loss}\n",
                         f"val_loss: {model.val_loss}\n", 
                         f"test_loss: {model.test_loss[0]}\n\n", # maybe delete [0]
                         f"weights: {weights}\n\n",
                         f"\n",
                         f"CLASSIFICATION REPORT: \n",
                         f"{cl_report}"])

def predict_class(y_pred, thresh=0.5): 
    """
    Returns a tensor with 1 if y_pred > 0.5 and 0 otherwise

    Args:
        - y_pred: model predictions
        - thresh (float): threshold for predictions

    Returns:
        - Tensor with 1 or 0
    """
    return tf.cast(y_pred > thresh, tf.float32)

def get_hpcombinations():
    """
    Defines and retrieves the possible hyperparameter options.

    Args:
        - model_type (str): defining type of model

    Returns:
        - hp_combinations (list): list of possible hyperparameter combinations
    """
    # define dictinary of hyperparameter options
    hpdict = {"optimizer": ['adam', 'sgd'],
                "learning_rate": [0.001, 0.01, 0.1],
                "batch_size": [8, 16, 32, 64],
                "epochs": [10, 50, 100]}
        
    # compute possible parameter combinations 
    hp_combinations = itertools.product(*(hpdict[key] for key in hpdict.keys()))
    hp_combinations = list(hp_combinations)
    
    return hp_combinations

def prepare_feature_splits(text_features_path):
    """
    Takes path to text features and returns autobiographical and triangle data separately for chronic and 1st episode patients respectively.

    Args:
        - text_features_path (str): defining path to text_features.csv

    Returns:
        - triangle_chronic_data (df): animated triangle transcripts for chronic patients
        - triangle_1st_episode_data (df): animated trianglae transcripts for 1st episode patients
        - autobiographical_chronic_data (df): autobiographical transcripts for chronic patients
        - autobiographical_1st_episode_data (df): autobiographical transcripts for 1st episode patients
    """
    # load text features as df
    text_features_df = pd.read_csv(text_features_path)

    # --- Animated Triangles data --- #

    # prepare chronic data
    triangle_chronic_data = text_features_df.loc[(text_features_df['data_type'] == "Triangle") & (text_features_df['Diagnosis'] != "1st_episode")]

    # prepare 1st episode data
    triangle_1st_episode_data = text_features_df.loc[(text_features_df['data_type'] == "Triangle") & (text_features_df['Diagnosis'] != "chronic")]

    # --- Autobiographical data --- #

    # prepare chronic data
    autobiographical_chronic_data = text_features_df.loc[(text_features_df['data_type'] == "Autobiographical") & (text_features_df['Diagnosis'] != "1st_episode")]

    # prepare 1st episode data
    autobiographical_1st_episode_data = text_features_df.loc[(text_features_df['data_type'] == "Autobiographical") & (text_features_df['Diagnosis'] != "chronic")]

    return triangle_chronic_data, triangle_1st_episode_data, autobiographical_chronic_data, autobiographical_1st_episode_data

def create_confusion_matrix(y_test, y_pred, out_path, out_filename):
    """
    Creates a confusion matrix based on the true and predicted labels using the sci-kit learn package.

    Args:
        - y_test: actual labels for test set
        - y_pred: predicted labels for test set
        - output_path (str): defining path to output folder
        - out_filename (str): defining name of .png file displaying confusion matrix

    Returns:
        - Confusion matrix as .png file saved in oputout folder
    """
    # clear plt
    plt.clf()

    # create confusion matrix
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

    # plot confusion matrix with matplotlib
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)

    # save confusion matrix in specified output fodler
    plt.savefig(os.path.join(out_path, out_filename))

def prepare_data_xgboost_transformer(seed, text_features_path="../data/features/text_features.csv"):
    """
    Prepares the data for the XGBoost and transformer models.

    Args:
        - seed (int): random_state for reproducibility
        - text_features_path (str): path to text_features.csv file

    Returns:
        - triangle_data (df): data from animated triangles task
        - autobiographical_data (df): data from autobiographical task
        - chronic_data (df): data for chronic patients
        - first_episode_data (df): data for first episode patients
    """
    # load text_features.csv
    df = pd.read_csv(text_features_path)

    # prepare low-level feature data
    triangle_chronic_data, triangle_1st_episode_data, autobiographical_chronic_data, autobiographical_1st_episode_data = prepare_feature_splits(text_features_path=text_features_path)

    # sample the controls for triangle data to seperate them between subsets
    controls_1 = triangle_chronic_data[triangle_chronic_data["binary_label"]==0].sample(frac=0.5, random_state=seed)
    
    # remove these controls from triangle_chronic data
    triangle_chronic_data_filtered = triangle_chronic_data[~triangle_chronic_data.loc[:,'ID'].isin(controls_1['ID'])]
    
    # filter the correct controls in 
    triangle_1st_episode_data_filtered = triangle_1st_episode_data[triangle_1st_episode_data.loc[:,'ID'].isin(controls_1['ID']) | triangle_1st_episode_data['binary_label']==1]
    
    # sample the controls for autobiographical data to seperate them between subsets
    controls_2 = autobiographical_chronic_data[autobiographical_chronic_data["binary_label"]==0].sample(frac=0.5, random_state=seed)
    
    # remove these controls from auto chronic data
    auto_chronic_data_filtered = autobiographical_chronic_data[~autobiographical_chronic_data.loc[:,'ID'].isin(controls_2['ID'])]
    
    # filter the correct controls in 
    auto_1st_episode_data_filtered = autobiographical_1st_episode_data[autobiographical_1st_episode_data.loc[:,'ID'].isin(controls_2['ID']) | autobiographical_1st_episode_data['binary_label']==1]

    # combine features into four datasets
    triangle_data = pd.concat([triangle_chronic_data[triangle_chronic_data["binary_label"]==1], triangle_1st_episode_data])
    autobiographical_data = pd.concat([autobiographical_chronic_data[autobiographical_chronic_data["binary_label"]==1], autobiographical_1st_episode_data])
    chronic_data = pd.concat([triangle_chronic_data_filtered, auto_chronic_data_filtered])
    first_episode_data = pd.concat([auto_1st_episode_data_filtered, triangle_1st_episode_data_filtered])

    # prepare word and sentence embeddings data
    word_embeddings = np.load("../data/features/embeddings/all_word_embeddings.npy")
    sentence_embeddings = np.load("../data/features/embeddings/all_sentence_embeddings.npy")
    y_IDs = df["ID"].values.tolist()

    # create dataframe with embeddings and labels
    dataset = pd.DataFrame({'word_embeddings': word_embeddings.tolist(), 
                            'sentence_embeddings': sentence_embeddings.tolist(), 
                            'y_IDs': y_IDs}, 
                            columns=['word_embeddings', 
                                    'sentence_embeddings', 
                                    'y_IDs'])
    
    # rename ID column
    dataset = dataset.rename(columns={"y_IDs": "ID"})

    # update the four datasets with word and sentence embeddings
    triangle_data = pd.merge(triangle_data, dataset, on='ID')
    autobiographical_data = pd.merge(autobiographical_data, dataset, on='ID')
    chronic_data = pd.merge(chronic_data, dataset, on='ID')
    first_episode_data = pd.merge(first_episode_data, dataset, on='ID')

    return triangle_data, autobiographical_data, chronic_data, first_episode_data

def compute_metrics(eval_preds):
    """
    Computes the accuracy based on the true and predicted labels.

    Args:
        - eval_preds: predicted labels on validation data

    Returns:
        - acc: model accuracy on validation data
    """
    # extract label ids
    labels = eval_preds.label_ids

    # extract validation predictions
    preds = eval_preds.predictions[0].argmax(-1) # [0] are the predictions

    # compute accuracy
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc}

def initialize_trainer(model, training_args, train_dataloader, eval_dataloader):
    """
    Initilizes the trainer for a transformer model when given training arguments as well as dataloader objects.

    Args:
        - model: initialized model to be trained
        - training_args: training arguments for model training
        - train_dataloader: dataloader object for training data
        - eval_dataloader: dataloader object for validation data

    Returns:
        - trainer: model trainer object
    """
    # initialize trainer 
    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataloader.dataset,
        eval_dataset=eval_dataloader.dataset,
        compute_metrics=compute_metrics)

    return trainer

class CustomModel(nn.Module):
    """
    Loads pretrained transformer model from checkpoint and extracts only its body (not head layers) and adds custom layer.
    The function was adopted from: https://towardsdatascience.com/adding-custom-layers-on-top-of-a-hugging-face-model-f1ccdfc257bd
    """
    def __init__(self, checkpoint, num_labels): 
        super(CustomModel, self).__init__() 

        # define number of labels
        self.num_labels = num_labels 

        # load pretrained model with given checkpoint and extract its body
        self.model = model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint, 
                                                                                                    output_attentions=True,
                                                                                                    output_hidden_states=True))
        # define dropout layer
        self.dropout = nn.Dropout(0.1) 

        # load and initialize weights
        self.classifier = nn.Linear(1024, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):

        # extract outputs from the body
        outputs = self.model(input_ids=input_ids, 
                             attention_mask=attention_mask)

        # add custom layers
        sequence_output = self.dropout(outputs[0])

        # compute losses
        logits = self.classifier(sequence_output[:,0,:].view(-1, 1024))
        loss = None
        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
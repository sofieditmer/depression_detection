#!/usr/bin/env python

"""
Script for training all XGBoost models
"""

# ---- DEPENDENCIES ---- #

import os
import sys
import joblib
sys.path.append(os.path.join(".."))
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
import wandb
from wandb.xgboost import WandbCallback
from utils.model_utils import split_equal_val_size_cv
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

# ---- MAIN FUNCTION ---- #

def tune_hyperparameters(X_train, y_train, val_size, text_features, sweep_project_name, rand_seed_name):
    """
    Performs hyperparameter optimization using the Bayesian search method to determine the optimal values for the 
    learning rate, gamma, max depth, min_child_weight and number of early stopping rounds based on the minimum log loss.
    The optimal combination of these values are returned. The hyperparameter optimization process is logged in weights and biases.

    Args:
        - X_train (df): training data
        - y_train (array): training labels
        - val_size (float): size of validation split
        - text_features (list[str]): the features on which the models are trained
        - sweep_project_name (str): name of the sweep in weights and biases
    
    Returns:
        - best_parameters (dict): optimal combination of hyperparameters
    """
    # define X_train
    X_train = X_train[text_features]

    # convert embeddings to arrays
    if text_features == ["word_embeddings"] or text_features == ["sentence_embeddings"]:
        X_train = np.array(list(X_train[text_features[0]]), dtype=np.float64)
    else:
        X_train = np.array(X_train)

    # login to weights and biases to log hyperparameter search
    wandb.login()

    # define train function
    def train():

        with wandb.init() as run:
            bst_params = {
                'objective': 'binary:logistic', 
                'n_estimators': 60,
                'booster': "gbtree",
                'learning_rate': run.config.learning_rate,     
                'gamma': run.config.gamma,
                'max_depth': run.config.max_depth,
                'min_child_weight': run.config.min_child_weight,
                'early_stopping_rounds': run.config.early_stopping_rounds,
                'max_delta_step': run.config.max_delta_step,  
                'eval_metric': ['logloss'],
                'tree_method': 'exact'
            }

            # initialize the XGBoostClassifier
            xgbmodel = xgb.XGBClassifier(**bst_params)

            # make lists to be appended to
            val_accuracy = []
            val_loss = []

            # run cross-validation
            seed_list = [7, 23, 68, 99, 75]
            for seed in seed_list:

                # shuffle X_train and y_train
                X_train_shuffled = shuffle(X_train, random_state=seed)
                y_train_shuffled = y_train.sample(frac=1, random_state=seed).reset_index(drop=True)

                # make train-val split
                X_train_split, y_train_split, X_val_split, y_val_split = split_equal_val_size_cv(X=X_train_shuffled,
                                                                                                y=y_train_shuffled, 
                                                                                                val_size=val_size)

                # train model and evaluate on validation
                xgbmodel.fit(X_train_split, 
                            y_train_split, 
                            eval_set=[(X_val_split, 
                                    y_val_split)],
                            callbacks=[WandbCallback(log_model=True,
                                                    log_feature_importance=False,
                                                    define_metric=True)],
                            verbose=False)
                
                # extract preds on validation
                val_preds = xgbmodel.predict(X_val_split)

                # calculate accuracy and append to list
                val_acc = accuracy_score(y_val_split, val_preds)
                val_accuracy.append(val_acc)

                # calculate log loss
                val_log_loss = log_loss(y_val_split, val_preds)
                val_loss.append(val_log_loss)

            # calculate mean accuracy and accuracy range for validation set
            mean_val_accuracy = np.mean(val_accuracy)
            print("Mean accuracy for all folds: %f" % (mean_val_accuracy))
            val_accuracy_range = (max(val_accuracy) - min(val_accuracy))

            # calculate mean log loss and log loss range for validation set
            mean_val_loss = np.mean(val_loss)
            print("Mean log loss for all folds: %f" % (mean_val_loss))
            val_loss_range = (max(val_loss) - min(val_loss))

            # add to wandb log
            wandb.log({"accuracy_val_score": mean_val_accuracy,
                    "accuracy_val_range": val_accuracy_range,
                    "log_loss_val": mean_val_loss,
                    "log_loss_val_range": val_loss_range
                        })

    # define sweep config (which hyperparamters to tune)
    sweep_config = {
        "name" : sweep_project_name,
        "method" : "bayes",
        "metric": {
            'name': 'logloss',
            'goal': 'minimize'   
        },
        "parameters" : {
            "learning_rate": {
                "min": 0.001,
                "max": 1.0
            },
            "gamma": {
                "min": 0.0,
                "max": 1.0 # change this to up to 10?
            },
            "max_depth": {
                "min": 1,
                "max": 10 # changed from [4, 6, 8]
            },
            "min_child_weight": {
                "min": 1,
                "max": 10
            },
            "early_stopping_rounds": {
                "values" : [10, 20, 40]
            }, 
            "max_delta_step": {
                "min": 0,
                "max": 5
            }}}

    # extract sweep id
    sweep_id = wandb.sweep(sweep_config, 
                        project=sweep_project_name)

    # run sweep
    wandb.agent(sweep_id, 
                project=sweep_project_name, 
                function=train, 
                count=50)

    # extract sweep from wandb
    api = wandb.Api()
    sweep = api.sweep(f"thesis_kat_sofie/{sweep_project_name}/{sweep_id}")

    # get best run parameters
    best_run = sweep.best_run(order="accuracy_val_score")
    best_parameters = best_run.config

    # save best paramaters as .pkl file
    with open(f'../output/xgboost/best_params/{sweep_project_name}_seed_{rand_seed_name}.pkl', 'wb') as f:
        pickle.dump(best_parameters, f)

    return best_parameters

def train_and_test_best_XGB_model(sweep_project_name, X_train, y_train, X_test, y_test, val_size, text_features, model_name, out_path, out_filename_cr, out_filename_cf, rand_seed_name):
    """
    Trains model using the hyperparameters that yielded the best model performance and tests model on test data. 
    Classification report and confusion matrix are saved to output directory.

    Args:
        - sweep_project_name (str): name of project in weights and biases
        - X_train (df): training data
        - y_train (df): training labels
        - X_test (df): test data
        - y_test (df): test labels
        - val_size (float): size of validation split
        - text_features (list[str]): text features on which to train model
        - model_name (str): filename of model to save
        - out_path (str): path to output directory
        - out_filename_cr (str): filename of classification report
        - out_filename_cf (str): filename of confusion matrix

    Returns:
        - classification_report.txt saved to output directory
        - confusion_matrix.png saved to output directory
    """
    # save IDs for appending later
    IDs = X_test["ID"].reset_index()

    # define X_train and X_test
    X_train = X_train[text_features]
    X_test = X_test[text_features]

    # convert embeddings to arrays
    if text_features == ["word_embeddings"] or text_features == ["sentence_embeddings"]:
        X_train = np.array(list(X_train[text_features[0]]), dtype=np.float64)
        X_test = np.array(list(X_test[text_features[0]]), dtype=np.float64)
    else:
        X_train = np.array(X_train)
        X_test = np.array(X_test)

    # make train-val split
    X_train, y_train, X_val, y_val = split_equal_val_size_cv(X=X_train, 
                                                             y=y_train, 
                                                             val_size=val_size)

    # load best parameters as estimated by hyperparameter optimization
    best_parameters = joblib.load(f'../output/xgboost/best_params/{sweep_project_name}_seed_{rand_seed_name}.pkl')

    # initialize the XGBoostClassifier with best parameters
    xgbmodel = xgb.XGBClassifier(**best_parameters)

    # Alter the number of tree estimators to 100
    xgbmodel.n_estimators = 100

    # create df with model predictions to append to
    model_predictions = pd.DataFrame({"true_diagnosis": y_test})

    # train model and evaluate on validation
    xgbmodel.fit(X_train, 
                 y_train, 
                 eval_set=[(X_val, y_val)])
    
    # save model
    if not os.path.exists('../output/xgboost/best_models'):
        os.makedirs('../output/xgboost/best_models')
    joblib.dump(xgbmodel, f'../output/xgboost/best_models/{model_name}_seed_{rand_seed_name}.pkl')

    # predict on test data 
    y_test_pred = xgbmodel.predict(X_test)
    
    # add model predictions to df   
    model_predictions[f"model_predicted_diagnosis"] = y_test_pred

    # append ID to df
    model_predictions["ID"] = IDs["ID"]

    # check if prediction is same as true diagnosis and append to df 
    model_predictions["correct_prediction"] = list(model_predictions["true_diagnosis"] == model_predictions["model_predicted_diagnosis"])

    # create classification report 
    cr = round(pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True)), 2)
    
    # create onfusion matrix
    cf = confusion_matrix(y_test, y_test_pred)

    # save classification report and confusion matrix
    save_performance_metrics(cr, 
                             cf, 
                             out_path=out_path,
                             out_filename_cr=out_filename_cr,
                             out_filename_cf=out_filename_cf,
                             best_parameters=best_parameters, 
                             rand_seed=rand_seed_name)

def test_model_generalizability(model_name, text_features, X_test, y_test, out_path, out_filename_cf, out_filename_cr, rand_seed_name):
    """
    This function loads the best model and tests it on a different data split than it has been trained on in 
    order to assess generalizability of the model.

    Args:
        - model_name (str): filename of model
        - text_features (list[str]): text features on which the model is trained
        - X_test (df): test data
        - y_test (df): test labels
        - out_path (str): path to output directory
        - out_filename_cf (str): filename of confusion matrix
        - out_filename_cr (str): filename of classification report

    Returns:
        - classification_report.txt saved to output directory
        - confusion_matrix.png saved to output directory
    """
    # prepare test data
    X_test = X_test[text_features]

    # convert embeddings to arrays
    if text_features == ["word_embeddings"] or text_features == ["sentence_embeddings"]:
        X_test = np.array(list(X_test[text_features[0]]), dtype=np.float64)
    else:
        X_test = np.array(X_test)

    # load model
    model = joblib.load(f'../output/xgboost/best_models/{model_name}_seed_{rand_seed_name}.pkl')

    # predict on test data
    y_test_pred = model.predict(X_test)
    
    # create classification report 
    gen_classification_report = round(pd.DataFrame(classification_report(y_test, 
                                                                         y_test_pred, 
                                                                         output_dict=True)), 2)
    
    # create confusion matrix
    gen_confusion_matrix = confusion_matrix(y_test, 
                                            y_test_pred)

    # save classification report and confusion matrix
    save_performance_metrics(gen_classification_report, 
                             gen_confusion_matrix, 
                             out_path=out_path,
                             out_filename_cf=out_filename_cf,
                             out_filename_cr=out_filename_cr,
                             best_parameters=model.get_xgb_params(), 
                             rand_seed=rand_seed_name)

def save_performance_metrics(cr, cf, out_path, out_filename_cf, out_filename_cr, best_parameters, rand_seed):
    """
    Saves the optimal hyperparameters as identified through hyperparameter optimization together with the classification report
    in a .txt file. Additinally, a confusion matrix is saved as a .png file.

    Args:
        - cr: classification report
        - cf: confusion matrix
        - out_path (str): path to output folder
        - out_filename_cf (str): filename of confusion matrix to be saved
        - out_filename_cr (str): filename of classifcation report to be saved
        - best_parameters (dict): dictionary containing the optimized hyperparamters

    Returns:
        - classification_report.txt saved in output/ dir
        - confusion_matrix.png saved in output/ dir
    """
    # create output directory if it does not exist already
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # define best parameters
    gamma = best_parameters.get("gamma")
    lr = best_parameters.get("learning_rate")
    max_depth = best_parameters.get("max_depth")
    min_child_weight = best_parameters.get("min_child_weight")
    early_stopping_rounds = best_parameters.get("early_stopping_rounds")
    max_delta_step = best_parameters.get("max_delta_step")

    # save classification report
    with open(os.path.join(out_path, out_filename_cr), 'a') as file:
        file.write(f"{datetime.now()}\n\n")
        file.writelines([f"Seed: {rand_seed}\n"
                         f"gamma: {gamma}\n",
                         f"lr: {lr}\n", 
                         f"max_depth: {max_depth}\n",
                         f"min_child_weight: {min_child_weight}\n",
                         f"max_delta_step: {max_delta_step}\n",
                         f"early_stopping_rounds: {early_stopping_rounds}\n\n",
                         f"CLASSIFICATION REPORT: \n",
                         f"{cr}\n\n"])

    # save confusion matrix
    plt.clf()
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cf, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cf.shape[0]):
        for j in range(cf.shape[1]):
            ax.text(x=j, y=i,s=cf[i, j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(os.path.join(out_path, out_filename_cf))

def error_analysis(text_features, X_test, y_test, model_name, type):
    X_test_raw = X_test
    X_test = X_test[text_features]
    # convert embeddings to arrays
    if text_features == ["word_embeddings"] or text_features == ["sentence_embeddings"]:
        X_test = np.array(list(X_test[text_features[0]]), dtype=np.float64)
    else:
        X_test = np.array(X_test)
    # load model
    model = joblib.load(f'../output/xgboost/best_models/{model_name}.pkl')
    # predict on test data
    y_test_pred = model.predict(X_test)

    df = pd.DataFrame({"ID": X_test_raw["ID"], "Diagnosis": X_test_raw["Diagnosis"], "Data type": X_test_raw["data_type"], "True label": y_test, "Prediction": y_test_pred})

    df.to_csv(f"{model_name}_{type}_error_analysis.csv")
    return df




    




    

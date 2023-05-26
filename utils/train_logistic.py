#!/usr/bin/env python

"""
This script is the utility script for training a logistic model, which is used in the main script train_baseline_models.py located in the src/ dir.
"""

## ---- DEPENDENCIES ---- ##

import os
import sys
sys.path.append(os.path.join(".."))
from utils.model_utils import *
from utils.logistic_regression import LogisticRegression
import argparse
from sklearn.metrics import classification_report

## --- TRAIN LOGISTIC REGRESSION MODEL --- ##

def main(data, model_type, optimizer, learning_rate, batch_size, epochs, output_path, input_text_features, out_filename, out_history_filename, out_cm_filename, test_size, val_size):
    """
    Trains logistic regression model with specified optimization algorithm, learning rate, batch size, and number of epochs.
    Evaluates model performance on test set and saves results in output/ dir.

    Args:
        data (df): data on which the model should be trained
        model_type (str): type of model to be trained
        optimizer (str): optimization algorithm
        learning_rate (int): learning rate used during trainig of the model
        batch_size (int): size of batches used during training 
        epochs (int): number of epochs to train for
        output_path (str): path to output folder
        input_text_features (list[str]): features on which the model should be trained
        out_filename (str): name of .txt file containing results of model training
        out_history_filename (str): name of .png file containing plot of training history
        out_cm_filename (str): name of .png file containig confusion matrix
        test_size (float): size of test split
        val_size (float): size of val split

    Returns:
        - model_results (.txt): model results saved in output folder
        - training_history (.png): model traning history
        - confusion_matrix (.png): confusion matrix
    """

    # --- PREPARATIONS --- #

    print("[INFO] Preparing data...")

    # define output path
    output_path = os.path.join(output_path, "logistic_regression/")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # make train-val-test split
    X_train, X_val, X_test, y_train, y_val, y_test = make_train_val_test_split(data=data, 
                                                                            test_size=test_size, 
                                                                            val_size=val_size, 
                                                                            text_features=input_text_features,
                                                                            seed=1)

    # --- TRAIN BASELINE LOGISTIC MODEL --- #
    
    print("[INFO] Initializing training of baseline logistic models...")
    
    # initialize model
    lr_model = LogisticRegression(optimizer, learning_rate, batch_size, epochs)

    # train model
    history = lr_model.train(X_train, 
                             y_train, 
                             X_val, 
                             y_val)
    
    # evaluate on test set
    lr_model.evaluate(X_test, y_test)

    # --- SAVE BASELINE MODEL METRICS, WEIGHTS AND EVALUATION --- #
    
    # save model_weights
    weights = lr_model.get_weights()

    # extract predictions on test data
    y_preds_test = lr_model.model.predict(X_test)

    # convert predictions to binary values (1 or 0) with threshold of 0.5
    y_preds_test = predict_class(y_preds_test, thresh=0.5)

    # extract classification report on test data
    cl_report = classification_report(y_test, y_preds_test)

    # create confusion matrix
    create_confusion_matrix(y_test, y_preds_test, output_path, out_cm_filename)

    # plot and save model training history
    plot_history(history, output_path, out_history_filename)

    # write and save results
    write_model_results(lr_model, 
                        model_type, 
                        optimizer, 
                        learning_rate, 
                        batch_size, 
                        epochs,
                        weights,  
                        output_path, 
                        out_filename, 
                        cl_report)

    print(f"[INFO] Finished! Model results are saved to {output_path}")

if __name__=="__main__":

    # --- ARGUMENT PARSER --- #

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--model_type', type=str, required=True,
                       help="define model type") 
    
    parser.add_argument('--optimizer', type=str, required=True,
                       help="adam or sgd")
    
    parser.add_argument('--learning_rate', type=float, required=True,
                       help="float defining learning rate of model")
    
    parser.add_argument('--batch_size', type=int, required=True,
                       help="int defining batch size used in model training")
    
    parser.add_argument('--epochs', type=int, required=True,
                       help="int defining number of epochs for training")

    parser.add_argument('--text_features', type=str, required=True,
                       help="str defining text features")
    
    parser.add_argument('--out_filename', type=str, required=True,
                        help="str defining name of output file")
    
    parser.add_argument('--out_history_filename', type=str, required=True,
                        help="str defining name of training history file")
    
    parser.add_argument('--output_path', type=str, required=True, 
                        help='Path to output directory')
    
    args = parser.parse_args()
                        
    # -- RUN MAIN FUNCTION --- #

    main(model_type = args.model_type,
         optimizer=args.optimizer,
         learning_rate=args.learning_rate,
         batch_size=args.batch_size,
         epochs=args.epochs,
         input_text_features=args.text_features,
         out_filename=args.out_filename,
         out_history_filename=args.out_history_filename,
         output_path=args.output_path)
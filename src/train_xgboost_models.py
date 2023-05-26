#!/usr/bin/env python

"""
Script for performing hyperparameter optimization and training all 12 XGBoost models.

Model overview:
    -- Models trained on low-level text features --
    - Model 1: model trained on autobiographical data to predict depressed patients based on low-level text features, i.e. use of 1st person personal pronouns, past-tense verbs, and negative words
    - Model 2: model trained on animated triangles data to predict depressed patients based on low-level text features, i.e. use of 1st person personal pronouns, past-tense verbs, and negative words
    - Model 3: model trained on data from chronic MDD patients to predict depressed patients based on low-level text features, i.e. use of 1st person personal pronouns, past-tense verbs, and negative words
    - Model 4: model trained on data from 1st episode MDD patients to predict depressed patients based on low-level text features, i.e. use of 1st person personal pronouns, past-tense verbs, and negative words

    -- Models trained on word embeddings --
    - Model 1: model trained on autobiographical data to predict depressed patients based on word embeddings generated from the transcripts using FastText
    - Model 2: model trained on animated triangles data to predict depressed patients based on word embeddings generated from the transcripts using FastText
    - Model 3: model trained on data from chronic MDD patients to predict depressed patients based on word embeddings generated from the transcripts using FastText
    - Model 4: model trained on data from 1st episode MDD patients to predict depressed patients based on word embeddings generated from the transcripts using FastText

    -- Models trained on sentence embeddings --
    - Model 1: model trained on autobiographical data to predict depressed patients based on sentence embeddings generated from the transcripts using pre-trained Danish sentence transformer model
    - Model 2: model trained on animated triangles data to predict depressed patients based on sentence embeddings generated from the transcripts using pre-trained Danish sentence transformer model
    - Model 3: model trained on data from chronic MDD patients to predict depressed patients based on sentence embeddings generated from the transcripts using pre-trained Danish sentence transformer model
    - Model 4: model trained on data from 1st episode MDD patients to predict depressed patients based on sentence embeddings generated from the transcripts using pre-trained Danish sentence transformer model
"""

# ---- DEPENDENCIES ---- #

import os
import sys
sys.path.append(os.path.join(".."))
import argparse
from utils.model_utils import make_train_test_split_xgboost, prepare_data_xgboost_transformer
from utils.train_XGB import tune_hyperparameters, train_and_test_best_XGB_model, test_model_generalizability

# ---- MAIN FUNCTION ---- #

def main(out_path="../output/xgboost/results/", test_size=0.20, val_size=0.13, rand_seed=1):
    """
    Performs hyperparameter optimization implemented in Weights and Biases using the Bayesian search method and trains all 
    the XGBoost models using the optimal hyperparameters. The generalizability of the models are then assessed by testing the 
    models on a different data splits than what they have been trained on.

    Args:
        - out_path_performance_reports (str): path to output directory to save performance metrics. Defaults to "../output/xgboost/results/".
        - test_size (float): size of test split which defaults to 0.2.
        - val_size (float): size of validation split which defaults to 0.25.
    """

    # ---- DATA PREPARATION ---- #

    print(f"[INFO] Preparing data with random seed: ", rand_seed)

    # prepare the four data splits
    triangle_data, autobiographical_data, chronic_data, first_episode_data = prepare_data_xgboost_transformer(seed=rand_seed, text_features_path="../data/features/text_features.csv")

    # split datasets into train and test (in order to have a held-out test set)
    
    X_train_auto, X_test_auto, y_train_auto, y_test_auto = make_train_test_split_xgboost(data=autobiographical_data,
                                                                                         test_size=test_size, 
                                                                                         seed=rand_seed, 
                                                                                         task_angle=True, 
                                                                                         patient_angle=False)
    
    X_train_tri, X_test_tri, y_train_tri, y_test_tri = make_train_test_split_xgboost(data=triangle_data,
                                                                                         test_size=test_size, 
                                                                                         seed=rand_seed, 
                                                                                         task_angle=True, 
                                                                                         patient_angle=False)
    
    X_train_chronic, X_test_chronic, y_train_chronic, y_test_chronic = make_train_test_split_xgboost(data=chronic_data,
                                                                                         test_size=test_size, 
                                                                                         seed=rand_seed, 
                                                                                         patient_angle=True)
    
    X_train_first_episode, X_test_first_episode, y_train_first_episode, y_test_first_episode = make_train_test_split_xgboost(data=first_episode_data,
                                                                                         test_size=test_size, 
                                                                                         seed=rand_seed,
                                                                                         patient_angle=True)

    print(X_test_chronic["Filename"])
    print(X_test_first_episode["Filename"])

    # ---- HYPERPARAMETER TUNING USING WEIGHTS & BIASES RANDOM SEARCH ---- #

    print("[INFO] Performing hyperparameter optimization and logging in Weights and Biases...")

    # --- models trained on low-level features --- #
    
    tune_hyperparameters(X_train=X_train_auto,
                         y_train=y_train_auto,
                         val_size=val_size,
                         text_features=["Pronouns_all_pronouns", 
                                        "Past_tense_all_verbs", 
                                        "Negative_all_sentiment",
                                        "Positive_all_sentiment"],
                         sweep_project_name="autobiographical_low_level",
                         rand_seed_name = rand_seed)
    
    print("[INFO] Done testing - moving on...")
    
    tune_hyperparameters(X_train=X_train_tri,
                         y_train=y_train_tri,
                         val_size=val_size,
                         text_features=["Pronouns_all_pronouns", 
                                        "Past_tense_all_verbs", 
                                        "Negative_all_sentiment",
                                        "Positive_all_sentiment"],
                         sweep_project_name="triangle_low_level",
                         rand_seed_name = rand_seed)
    
    tune_hyperparameters(X_train=X_train_chronic,
                         y_train=y_train_chronic,
                         val_size=val_size,
                         text_features=["Pronouns_all_pronouns", 
                                        "Past_tense_all_verbs",  
                                        "Negative_all_sentiment",
                                        "Positive_all_sentiment"],
                         sweep_project_name="chronic_low_level", 
                         rand_seed_name = rand_seed)

    tune_hyperparameters(X_train=X_train_first_episode, 
                         y_train=y_train_first_episode,
                         val_size=val_size,
                         text_features=["Pronouns_all_pronouns", 
                                        "Past_tense_all_verbs", 
                                        "Negative_all_sentiment",
                                        "Positive_all_sentiment"], 
                         sweep_project_name="first_episode_low_level", 
                         rand_seed_name = rand_seed)
    
    # --- models trained on word embeddings --- #

    tune_hyperparameters(X_train=X_train_auto, 
                         y_train=y_train_auto,
                         val_size=val_size,
                         text_features=["word_embeddings"], 
                         sweep_project_name="autobiographical_word_embeddings", 
                         rand_seed_name = rand_seed)
    
    tune_hyperparameters(X_train=X_train_tri,
                         y_train=y_train_tri,
                         val_size=val_size,
                         text_features=["word_embeddings"], 
                         sweep_project_name="triangle_word_embeddings", 
                         rand_seed_name = rand_seed)
    
    tune_hyperparameters(X_train=X_train_chronic, 
                         y_train=y_train_chronic,
                         val_size=val_size,
                         text_features=["word_embeddings"], 
                         sweep_project_name="chronic_word_embeddings", 
                         rand_seed_name = rand_seed)
    
    tune_hyperparameters(X_train=X_train_first_episode,
                         y_train=y_train_first_episode,
                         val_size=val_size,
                         text_features=["word_embeddings"], 
                         sweep_project_name="first_episode_word_embeddings", 
                         rand_seed_name = rand_seed)
    
    # --- models trained on sentence embeddings --- #
    
    tune_hyperparameters(X_train=X_train_auto, 
                         y_train=y_train_auto,
                         val_size=val_size,
                         text_features=["sentence_embeddings"], 
                         sweep_project_name="autobiographical_sentence_embeddings", 
                         rand_seed_name = rand_seed)
    
    tune_hyperparameters(X_train=X_train_tri,
                         y_train=y_train_tri,
                         val_size=val_size,
                         text_features=["sentence_embeddings"], 
                         sweep_project_name="triangle_sentence_embeddings", 
                         rand_seed_name = rand_seed)
    
    tune_hyperparameters(X_train=X_train_chronic, 
                         y_train=y_train_chronic,
                         val_size=val_size,
                         text_features=["sentence_embeddings"], 
                         sweep_project_name="chronic_sentence_embeddings", 
                         rand_seed_name = rand_seed)
   
    tune_hyperparameters(X_train=X_train_first_episode, 
                         y_train=y_train_first_episode,
                         val_size=val_size,
                         text_features=["sentence_embeddings"], 
                         sweep_project_name="first_episode_sentence_embeddings", 
                         rand_seed_name = rand_seed)
    '''
    # ---- TRAIN MODEL WITH OPTIMAL HYPERPARAMTERS AND SAVE PERFORMANCE RESULTS ---- #

    print("[INFO] Training models with optimized hyperparameters, testing on test data and assessing generalizability of model...")

    # --- models trained on low-level features --- #
    '''
    # define text features
    text_features = ["Pronouns_all_pronouns", 
                     "Past_tense_all_verbs",
                     "Negative_all_sentiment",
                     "Positive_all_sentiment"]

    # define filenames
    model_name="auto_low_level"
    out_filename_cf="auto_low_level_confusion_matrix.png"
    out_filename_cr="auto_low_level_classification_report.txt"

    # For now moved up for testing
    # train and test best model
    train_and_test_best_XGB_model(sweep_project_name="autobiographical_low_level",
                                  X_train=X_train_auto,
                                  y_train=y_train_auto, 
                                  X_test=X_test_auto,
                                  y_test=y_test_auto,
                                  val_size=val_size,
                                  text_features=text_features,
                                  model_name=model_name,
                                  out_path=out_path,
                                  out_filename_cf=out_filename_cf,
                                  out_filename_cr=out_filename_cr, 
                                  rand_seed_name = rand_seed)
    
    # test generalizability of model
    test_model_generalizability(model_name=model_name, 
                                  text_features=text_features, 
                                  X_test=X_test_tri, 
                                  y_test=y_test_tri, 
                                  out_path=out_path, 
                                  out_filename_cf=f"generalizability/test_tri_{out_filename_cf}", 
                                  out_filename_cr=f"generalizability/test_tri_{out_filename_cr}", 
                                  rand_seed_name = rand_seed)
    
    # define filenames
    model_name="triangle_low_level"
    out_filename_cf="triangle_low_level_confusion_matrix.png"
    out_filename_cr="triangle_low_level_classification_report.txt"

    # train and test best model
    train_and_test_best_XGB_model(sweep_project_name=model_name,
                                  X_train=X_train_tri,
                                  y_train=y_train_tri, 
                                  X_test=X_test_tri,
                                  y_test=y_test_tri,
                                  val_size=val_size,
                                  text_features=text_features,
                                  model_name=model_name,
                                  out_path=out_path,
                                  out_filename_cf=out_filename_cf,
                                  out_filename_cr=out_filename_cr, 
                                  rand_seed_name = rand_seed)
    
    # test generalizability of model
    test_model_generalizability(model_name=model_name, 
                                  text_features=text_features, 
                                  X_test=X_test_auto, 
                                  y_test=y_test_auto, 
                                  out_path=out_path, 
                                  out_filename_cf=f"generalizability/test_auto_{out_filename_cf}", 
                                  out_filename_cr=f"generalizability/test_auto_{out_filename_cr}", 
                                  rand_seed_name = rand_seed)
    
    # define filenames
    model_name="chronic_low_level"
    out_filename_cf="chronic_low_level_confusion_matrix.png"
    out_filename_cr="chronic_low_level_classification_report.txt"

    # train and test best model
    train_and_test_best_XGB_model(sweep_project_name=model_name,
                                  X_train=X_train_chronic,
                                  y_train=y_train_chronic, 
                                  X_test=X_test_chronic,
                                  y_test=y_test_chronic,
                                  val_size=val_size,
                                  text_features=text_features,
                                  model_name=model_name,
                                  out_path=out_path,
                                  out_filename_cf=out_filename_cf,
                                  out_filename_cr=out_filename_cr, 
                                  rand_seed_name = rand_seed)
    
    # test generalizability of model
    test_model_generalizability(model_name=model_name, 
                                  text_features=text_features, 
                                  X_test=X_test_first_episode, 
                                  y_test=y_test_first_episode, 
                                  out_path=out_path, 
                                  out_filename_cf=f"generalizability/test_first_episode_{out_filename_cf}", 
                                  out_filename_cr=f"generalizability/test_first_episode_{out_filename_cr}", 
                                  rand_seed_name = rand_seed)

    # define filenames
    model_name="first_episode_low_level"
    out_filename_cf="first_episode_low_level_confusion_matrix.png"
    out_filename_cr="first_episode_low_level_classification_report.txt"

    # train and test best model
    train_and_test_best_XGB_model(sweep_project_name=model_name,
                                  X_train=X_train_first_episode,
                                  y_train=y_train_first_episode, 
                                  X_test=X_test_first_episode,
                                  y_test=y_test_first_episode,
                                  val_size=val_size,
                                  text_features=text_features,
                                  model_name=model_name,
                                  out_path=out_path, 
                                  out_filename_cf=out_filename_cf,
                                  out_filename_cr=out_filename_cr, 
                                  rand_seed_name = rand_seed)
    
    # test generalizability of model
    test_model_generalizability(model_name=model_name, 
                                  text_features=text_features, 
                                  X_test=X_test_chronic, 
                                  y_test=y_test_chronic, 
                                  out_path=out_path, 
                                  out_filename_cf=f"generalizability/test_chronic_{out_filename_cf}", 
                                  out_filename_cr=f"generalizability/test_chronic_{out_filename_cr}", 
                                  rand_seed_name = rand_seed)
    
    # --- models trained on word embeddings --- #

    # define features
    text_features=["word_embeddings"]
    
    # define filenames
    model_name="autobiographical_word_embeddings"
    out_filename_cf="autobiographical_word_embeddings_confusion_matrix.png"
    out_filename_cr="autobiographical_word_embeddings_classification_report.txt"

    # train and test best model
    train_and_test_best_XGB_model(sweep_project_name=model_name,
                                  X_train=X_train_auto,
                                  y_train=y_train_auto, 
                                  X_test=X_test_auto,
                                  y_test=y_test_auto,
                                  val_size=val_size,
                                  text_features=text_features,
                                  model_name=model_name,
                                  out_path=out_path,
                                  out_filename_cf=out_filename_cf,
                                  out_filename_cr=out_filename_cr, 
                                  rand_seed_name = rand_seed)
    
    # test generalizability of model
    test_model_generalizability(model_name=model_name, 
                                  text_features=text_features, 
                                  X_test=X_test_tri, 
                                  y_test=y_test_tri, 
                                  out_path=out_path, 
                                  out_filename_cf=f"generalizability/test_tri_{out_filename_cf}", 
                                  out_filename_cr=f"generalizability/test_tri_{out_filename_cr}", 
                                  rand_seed_name = rand_seed)

    # define filenames
    model_name="triangle_word_embeddings"
    out_filename_cf="triangle_word_embeddings_confusion_matrix.png"
    out_filename_cr="triangle_word_embeddings_classification_report.txt"

    # train and test best model
    train_and_test_best_XGB_model(sweep_project_name=model_name,
                                  X_train=X_train_tri,
                                  y_train=y_train_tri, 
                                  X_test=X_test_tri,
                                  y_test=y_test_tri,
                                  val_size=val_size,
                                  text_features=text_features,
                                  model_name=model_name,
                                  out_path=out_path,
                                  out_filename_cf=out_filename_cf,
                                  out_filename_cr=out_filename_cr, 
                                  rand_seed_name = rand_seed)
    
    # test generalizability of model
    test_model_generalizability(model_name=model_name, 
                                  text_features=text_features, 
                                  X_test=X_test_auto, 
                                  y_test=y_test_auto, 
                                  out_path=out_path, 
                                  out_filename_cf=f"generalizability/test_auto_{out_filename_cf}", 
                                  out_filename_cr=f"generalizability/test_auto_{out_filename_cr}", 
                                  rand_seed_name = rand_seed)
    
    # define filenames
    model_name="chronic_word_embeddings"
    out_filename_cf="chronic_word_embeddings_confusion_matrix.png"
    out_filename_cr="chronic_word_embeddings_classification_report.txt"

    # train and test best model
    train_and_test_best_XGB_model(sweep_project_name=model_name,
                                  X_train=X_train_chronic,
                                  y_train=y_train_chronic, 
                                  X_test=X_test_chronic,
                                  y_test=y_test_chronic,
                                  val_size=val_size,
                                  text_features=text_features,
                                  model_name=model_name,
                                  out_path=out_path,
                                  out_filename_cf=out_filename_cf,
                                  out_filename_cr=out_filename_cr, 
                                  rand_seed_name = rand_seed)
    
    # test generalizability of model
    test_model_generalizability(model_name=model_name, 
                                  text_features=text_features, 
                                  X_test=X_test_first_episode, 
                                  y_test=y_test_first_episode, 
                                  out_path=out_path, 
                                  out_filename_cf=f"generalizability/test_first_episode_{out_filename_cf}", 
                                  out_filename_cr=f"generalizability/test_first_episode_{out_filename_cr}", 
                                  rand_seed_name = rand_seed)

    # define filenames
    model_name="first_episode_word_embeddings"
    out_filename_cf="first_episode_word_embeddings_confusion_matrix.png"
    out_filename_cr="first_episode_word_embeddings_classification_report.txt"

    # train and test best model
    train_and_test_best_XGB_model(sweep_project_name=model_name,
                                  X_train=X_train_first_episode,
                                  y_train=y_train_first_episode, 
                                  X_test=X_test_first_episode,
                                  y_test=y_test_first_episode,
                                  val_size=val_size,
                                  text_features=text_features,
                                  model_name=model_name,
                                  out_path=out_path, 
                                  out_filename_cf=out_filename_cf,
                                  out_filename_cr=out_filename_cr, 
                                  rand_seed_name = rand_seed)
    
    # test generalizability of model
    test_model_generalizability(model_name=model_name, 
                                  text_features=text_features, 
                                  X_test=X_test_chronic, 
                                  y_test=y_test_chronic, 
                                  out_path=out_path, 
                                  out_filename_cf=f"generalizability/test_chronic_{out_filename_cf}", 
                                  out_filename_cr=f"generalizability/test_chronic_{out_filename_cr}", 
                                  rand_seed_name = rand_seed)
    
    # --- models trained on sentence embeddings --- #

    # define features
    text_features=["sentence_embeddings"]
    
    # define filenames
    model_name="auto_sentence_embeddings"
    out_filename_cf="autobiographical_sentence_embeddings_classification_report.png"
    out_filename_cr="autobiographical_sentence_embeddings_classification_report.txt"
    
    # train and test best model
    train_and_test_best_XGB_model(sweep_project_name="autobiographical_sentence_embeddings",
                                  X_train=X_train_auto,
                                  y_train=y_train_auto, 
                                  X_test=X_test_auto,
                                  y_test=y_test_auto,
                                  val_size=val_size,
                                  text_features=text_features, 
                                  model_name=model_name,
                                  out_path=out_path, 
                                  out_filename_cf=out_filename_cf,
                                  out_filename_cr=out_filename_cr, 
                                  rand_seed_name = rand_seed)
    
    # test generalizability of model
    test_model_generalizability(model_name=model_name, 
                                  text_features=text_features, 
                                  X_test=X_test_tri, 
                                  y_test=y_test_tri, 
                                  out_path=out_path, 
                                  out_filename_cf=f"generalizability/test_tri_{out_filename_cf}", 
                                  out_filename_cr=f"generalizability/test_tri_{out_filename_cr}", 
                                  rand_seed_name = rand_seed)
                    
    # define filenames
    model_name="triangle_sentence_embeddings"
    out_filename_cf="triangle_sentence_embeddings_classification_report.png"
    out_filename_cr="triangle_sentence_embeddings_classification_report.txt"

    # train and test best model
    train_and_test_best_XGB_model(sweep_project_name=model_name,
                                  X_train=X_train_tri,
                                  y_train=y_train_tri, 
                                  X_test=X_test_tri,
                                  y_test=y_test_tri,
                                  val_size=val_size,
                                  text_features=text_features,
                                  model_name=model_name,
                                  out_path=out_path,
                                  out_filename_cf=out_filename_cf,
                                  out_filename_cr=out_filename_cr, 
                                  rand_seed_name = rand_seed)
    
    # test generalizability of model
    test_model_generalizability(model_name=model_name, 
                                  text_features=text_features, 
                                  X_test=X_test_auto, 
                                  y_test=y_test_auto, 
                                  out_path=out_path, 
                                  out_filename_cf=f"generalizability/test_auto_{out_filename_cf}", 
                                  out_filename_cr=f"generalizability/test_auto_{out_filename_cr}", 
                                  rand_seed_name = rand_seed)
    
    # define filenames
    model_name="chronic_sentence_embeddings"
    out_filename_cf="chronic_sentence_embeddings_classification_report.png"
    out_filename_cr="chronic_sentence_embeddings_classification_report.txt"
    
    # train and test best model
    train_and_test_best_XGB_model(sweep_project_name=model_name,
                                  X_train=X_train_chronic,
                                  y_train=y_train_chronic, 
                                  X_test=X_test_chronic,
                                  y_test=y_test_chronic,
                                  val_size=val_size,
                                  text_features=text_features,
                                  model_name=model_name,
                                  out_path=out_path,
                                  out_filename_cf=out_filename_cf,
                                  out_filename_cr=out_filename_cr, 
                                  rand_seed_name = rand_seed)
    
    # test generalizability of model
    test_model_generalizability(model_name=model_name, 
                                  text_features=text_features, 
                                  X_test=X_test_first_episode, 
                                  y_test=y_test_first_episode, 
                                  out_path=out_path, 
                                  out_filename_cf=f"generalizability/test_first_episode_{out_filename_cf}", 
                                  out_filename_cr=f"generalizability/test_first_episode_{out_filename_cr}", 
                                  rand_seed_name = rand_seed)
    
    # define filenames
    model_name="first_episode_sentence_embeddings"
    out_filename_cf="first_episode_sentence_embeddings_classification_report.png"
    out_filename_cr="first_episode_sentence_embeddings_classification_report.txt"

    # train and test best model
    train_and_test_best_XGB_model(sweep_project_name=model_name,
                                  X_train=X_train_first_episode,
                                  y_train=y_train_first_episode, 
                                  X_test=X_test_first_episode,
                                  y_test=y_test_first_episode,
                                  val_size=val_size,
                                  text_features=text_features,
                                  model_name=model_name,
                                  out_path=out_path,
                                  out_filename_cf=out_filename_cf,
                                  out_filename_cr=out_filename_cr, 
                                  rand_seed_name = rand_seed)

    # test generalizability of model
    test_model_generalizability(model_name=model_name, 
                                  text_features=text_features, 
                                  X_test=X_test_chronic, 
                                  y_test=y_test_chronic, 
                                  out_path=out_path, 
                                  out_filename_cf=f"generalizability/test_chronic_{out_filename_cf}", 
                                  out_filename_cr=f"generalizability/test_chronic_{out_filename_cr}", 
                                  rand_seed_name = rand_seed)

    print(f"[INFO] Done! Performance results are saved in {out_path}...")

if __name__=="__main__":

    # --- ARGUMENT PARSER --- #

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--out_path', type=str, required=False, default="../output/xgboost/results/",
                        help='define path to output dir')
    
    parser.add_argument('--test_size', type=float, required=False, default=0.20,
                        help='define size of test split as float, e.g. 0.20')

    parser.add_argument('--val_size', type=float, required=False, default=0.13,
                        help='define size of val split as float, e.g. 0.25')
    
    parser.add_argument('--rand_seed', type=int, required=False, default=1,
                        help='define a random seed for the data splits, e.g. 1')
    
    args = parser.parse_args()

    # -- RUN MAIN FUNCTION --- #

    main(out_path = args.out_path,
         test_size=args.test_size,
         val_size=args.val_size,
         rand_seed=args.rand_seed)
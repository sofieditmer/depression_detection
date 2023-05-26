#!/usr/bin/env python

"""
Script for performing grid search to optimize optimal hyperparamters and training all 12 baseline logistic regression models.

Model overview:
    -- Animated triangles task --
    - Model 1: trained on animated triangles task data to predict chronic MDD patients based on 1st person singular pronoun use.
    - Model 2: trained on animated triangles task data to predict chronic MDD patients based on past-tense verb use.
    - Model 3: trained on animated triangles task data to predict chronic MDD patients based on negative word use. 
    - Model 4: trained on animated triangles task data to predict 1st episode MDD patients based on 1st person singular pronoun use. 
    - Model 5: trained on animated triangles task data to predict 1st episode MDD patients based on past-tense verb use.
    - Model 6: trained on animated triangles task data to predict 1st episode MDD patients based on negative word use. 

    -- Autobiographical task --
    - Model 1: trained on autobiographical data to predict chronic MDD patients based on 1st person singular pronoun use.
    - Model 2: trained on autobiographical data to predict chronic MDD patients based on past-tense verb use.
    - Model 3: trained on autobiographical data to predict chronic MDD patients based on negative word use.
    - Model 4: trained on autobiographical data to predict 1st episode MDD patients based on 1st person singular pronoun use. 
    - Model 5: trained on autobiographical data to predict 1st episode MDD patients based on past-tense verb use.
    - Model 6: trained on autobiographical data to predict 1st episode MDD patients based on negative word use.
"""

# ---- DEPENDENCIES ---- #

import os
import sys
sys.path.append(os.path.join(".."))
from utils.train_logistic import main as train_model
from utils.grid_search import main as grid_search
from utils.model_utils import prepare_feature_splits
import argparse

# ---- MAIN FUNCTION ---- #

def main(out_path="../output/baseline", test_size=0.20, val_size=0.13):
    """
    Trains all logistic regression models used as baseline models.

    Args:
        - out_path (str): path to output folder that defaults to "../output/baseline".
        - test_size (float): size of test split that defaults to 0.20.
        - val_size (float): size of validation split that defaults to 0.25.

    Returns:
        - grid search results for all models saved in output/baseline/gridsearch folder
        - model performance results for all models saved in output/baseline/logistic_regression folder
    """

    # --- PREPARE DATA --- #

    print("[INFO] Preparing data splits...")
    
    triangle_chronic_data, triangle_1st_episode_data, autobiographical_chronic_data, autobiographical_1st_episode_data = prepare_feature_splits(text_features_path="../data/features/text_features.csv")

    # Shuffle data
    triangle_chronic_data = triangle_chronic_data.sample(frac = 1).reset_index(drop=True)
    triangle_1st_episode_data = triangle_1st_episode_data.sample(frac = 1).reset_index(drop=True)
    autobiographical_chronic_data = autobiographical_chronic_data.sample(frac = 1).reset_index(drop=True)
    autobiographical_1st_episode_data = autobiographical_1st_episode_data.sample(frac = 1).reset_index(drop=True)

    # --- GRID SEARCH --- #

    print("[INFO] Performing grid search for all models...")

    # --- TRIANGLE/CHRONIC MODELS --- #
 
    best_combination_triangle_model1 = grid_search(data=triangle_chronic_data,
                                                   model_type="logistic_regression", 
                                                   filename="gridsearch_results_model1_triangle_chronic_pronouns_all_pronouns.txt", 
                                                   input_text_features=["Pronouns_all_pronouns"],
                                                   output_path=out_path,
                                                   test_size=test_size,
                                                   val_size=val_size)
    
    best_combination_triangle_model2 = grid_search(data=triangle_chronic_data,
                                                   model_type="logistic_regression", 
                                                   filename="gridsearch_results_model2_triangle_chronic_past_tense_all_verbs.txt", 
                                                   input_text_features=["Past_tense_all_verbs"],
                                                   output_path=out_path,
                                                   test_size=test_size,
                                                   val_size=val_size)
    
    best_combination_triangle_model3 = grid_search(data=triangle_chronic_data,
                                                   model_type="logistic_regression", 
                                                   filename="gridsearch_results_model3_triangle_chronic_negative_all_sentiment.txt", 
                                                   input_text_features=["Negative_all_sentiment"],
                                                   output_path=out_path,
                                                   test_size=test_size,
                                                   val_size=val_size)

    best_combination_triangle_model3a = grid_search(data=triangle_chronic_data,
                                                   model_type="logistic_regression", 
                                                   filename="gridsearch_results_model3a_triangle_chronic_positive_all_sentiment.txt", 
                                                   input_text_features=["Positive_all_sentiment"],
                                                   output_path=out_path,
                                                   test_size=test_size,
                                                   val_size=val_size)                                     
    
    # --- TRIANGLE/1ST EPISODE MODELS --- #
    
    best_combination_triangle_model4 = grid_search(data=triangle_1st_episode_data,
                                                   model_type="logistic_regression", 
                                                   filename="gridsearch_results_model4_triangle_1st_episode_pronouns_all_pronouns.txt", 
                                                   input_text_features=["Pronouns_all_pronouns"],
                                                   output_path=out_path,
                                                   test_size=test_size,
                                                   val_size=val_size)
    
    best_combination_triangle_model5 = grid_search(data=triangle_1st_episode_data,
                                                   model_type="logistic_regression", 
                                                   filename="gridsearch_results_model5_triangle_1st_episode_past_tense_all_verbs.txt", 
                                                   input_text_features=["Past_tense_all_verbs"],
                                                   output_path=out_path,
                                                   test_size=test_size,
                                                   val_size=val_size)

    best_combination_triangle_model6 = grid_search(data=triangle_1st_episode_data,
                                                   model_type="logistic_regression", 
                                                   filename="gridsearch_results_model5_triangle_1st_episode_negative_all_sentiment.txt", 
                                                   input_text_features=["Negative_all_sentiment"],
                                                   output_path=out_path,
                                                   test_size=test_size,
                                                   val_size=val_size)
                                            
    best_combination_triangle_model6a = grid_search(data=triangle_1st_episode_data,
                                                   model_type="logistic_regression", 
                                                   filename="gridsearch_results_model5a_triangle_1st_episode_positive_all_sentiment.txt", 
                                                   input_text_features=["Positive_all_sentiment"],
                                                   output_path=out_path,
                                                   test_size=test_size,
                                                   val_size=val_size)
    
    # --- AUTOBIOGRAPHICAL/CHRONIC MODELS --- #

    best_combination_autobiographical_model1 = grid_search(data=autobiographical_chronic_data,
                                                   model_type="logistic_regression", 
                                                   filename="gridsearch_results_model1_autobiographical_pronouns_all_pronouns.txt", 
                                                   input_text_features=["Pronouns_all_pronouns"],
                                                   output_path=out_path,
                                                   test_size=test_size,
                                                   val_size=val_size)
    
    best_combination_autobiographical_model2 = grid_search(data=autobiographical_chronic_data,
                                                   model_type="logistic_regression", 
                                                   filename="gridsearch_results_model2_autobiographical_past_tense_all_verbs.txt", 
                                                   input_text_features=["Past_tense_all_verbs"],
                                                   output_path=out_path,
                                                   test_size=test_size,
                                                   val_size=val_size)
    
    best_combination_autobiographical_model3 = grid_search(data=autobiographical_chronic_data,
                                                   model_type="logistic_regression", 
                                                   filename="gridsearch_results_model3_autobiographical_negative_all_sentiment.txt", 
                                                   input_text_features=["Negative_all_sentiment"],
                                                   output_path=out_path,
                                                   test_size=test_size,
                                                   val_size=val_size)

    best_combination_autobiographical_model3a = grid_search(data=autobiographical_chronic_data,
                                                   model_type="logistic_regression", 
                                                   filename="gridsearch_results_model3a_autobiographical_positive_all_sentiment.txt", 
                                                   input_text_features=["Positive_all_sentiment"],
                                                   output_path=out_path,
                                                   test_size=test_size,
                                                   val_size=val_size)
    
    # --- AUTOBIOGRAPHICAL/1ST EPISODE MODELS --- #
    
    best_combination_autobiographical_model4 = grid_search(data=autobiographical_1st_episode_data,
                                                   model_type="logistic_regression", 
                                                   filename="gridsearch_results_model4_autobiographical_pronouns_all_pronouns.txt", 
                                                   input_text_features=["Pronouns_all_pronouns"],
                                                   output_path=out_path,
                                                   test_size=test_size,
                                                   val_size=val_size)
    
    best_combination_autobiographical_model5 = grid_search(data=autobiographical_1st_episode_data,
                                                   model_type="logistic_regression", 
                                                   filename="gridsearch_results_model5_autobiographical_past_tense_all_verbs.txt", 
                                                   input_text_features=["Past_tense_all_verbs"],
                                                   output_path=out_path,
                                                   test_size=test_size,
                                                   val_size=val_size)

    best_combination_autobiographical_model6 = grid_search(data=autobiographical_1st_episode_data,
                                                   model_type="logistic_regression", 
                                                   filename="gridsearch_results_model6_autobiographical_negative_all_sentiment.txt", 
                                                   input_text_features=["Negative_all_sentiment"],
                                                   output_path=out_path,
                                                   test_size=test_size,
                                                   val_size=val_size)
    
    best_combination_autobiographical_model6a = grid_search(data=autobiographical_1st_episode_data,
                                                   model_type="logistic_regression", 
                                                   filename="gridsearch_results_model6a_autobiographical_positive_all_sentiment.txt", 
                                                   input_text_features=["Positive_all_sentiment"],
                                                   output_path=out_path,
                                                   test_size=test_size,
                                                   val_size=val_size)
    
    # --- TRAIN MODELS --- #

    print("[INFO] Traning models with parameters optimized by grid search...")

    # --- TRAIN TRIANGLE/CHRONIC MODELS --- #
    
    # model 1
    train_model(data=triangle_chronic_data, 
                model_type="logistic_regression", 
                optimizer=best_combination_triangle_model1[0], 
                learning_rate=best_combination_triangle_model1[1], 
                batch_size=best_combination_triangle_model1[2], 
                epochs=best_combination_triangle_model1[3], 
                output_path=out_path, 
                input_text_features=["Pronouns_all_pronouns"], 
                out_filename="lr_model1_triangle_chronic_pronouns_all_pronouns_results.txt", 
                out_history_filename="lr_model1_triangle_chronic_pronouns_all_pronouns_history.png",
                out_cm_filename="lr_model1_triangle_chronic_pronouns_all_pronouns_cm.png",
                test_size=test_size,
                val_size=val_size)

    # model 2
    train_model(data=triangle_chronic_data, 
                model_type="logistic_regression", 
                optimizer=best_combination_triangle_model2[0], 
                learning_rate=best_combination_triangle_model2[1], 
                batch_size=best_combination_triangle_model2[2], 
                epochs=best_combination_triangle_model2[3], 
                output_path=out_path, 
                input_text_features=["Past_tense_all_verbs"], 
                out_filename="lr_model2_triangle_chronic_past_tense_all_verbs_results.txt", 
                out_history_filename="lr_model2_triangle_chronic_past_tense_all_verbs_history.png",
                out_cm_filename="lr_model2_triangle_chronic_past_tense_all_verbs_cm.png",
                test_size=test_size,
                val_size=val_size)

    # model 3
    train_model(data=triangle_chronic_data, 
                model_type="logistic_regression", 
                optimizer=best_combination_triangle_model3[0], 
                learning_rate=best_combination_triangle_model3[1], 
                batch_size=best_combination_triangle_model3[2], 
                epochs=best_combination_triangle_model3[3], 
                output_path=out_path, 
                input_text_features=["Negative_all_sentiment"], 
                out_filename="lr_model2_triangle_chronic_negative_all_sentiment_results.txt", 
                out_history_filename="lr_model2_triangle_chronic_negative_all_sentiment_history.png",
                out_cm_filename="lr_model2_triangle_chronic_negative_all_sentiment_cm.png",
                test_size=test_size,
                val_size=val_size)
    
    # model 3a
    train_model(data=triangle_chronic_data, 
                model_type="logistic_regression", 
                optimizer=best_combination_triangle_model3a[0], 
                learning_rate=best_combination_triangle_model3a[1], 
                batch_size=best_combination_triangle_model3a[2], 
                epochs=best_combination_triangle_model3a[3], 
                output_path=out_path, 
                input_text_features=["Positive_all_sentiment"], 
                out_filename="lr_model2_triangle_chronic_positive_all_sentiment_results.txt", 
                out_history_filename="lr_model2_triangle_chronic_positive_all_sentiment_history.png",
                out_cm_filename="lr_model2_triangle_chronic_positive_all_sentiment_cm.png",
                test_size=test_size,
                val_size=val_size)
    
    # --- TRAIN TRIANGLE/1ST EPISODE MODELS --- #
   
    # model 4
    train_model(data=triangle_1st_episode_data, 
                model_type="logistic_regression", 
                optimizer=best_combination_triangle_model4[0], 
                learning_rate=best_combination_triangle_model4[1], 
                batch_size=best_combination_triangle_model4[2], 
                epochs=best_combination_triangle_model4[3], 
                output_path=out_path, 
                input_text_features=["Pronouns_all_pronouns"], 
                out_filename="lr_model2_triangle_1st_episode_pronouns_all_pronouns_results.txt", 
                out_history_filename="lr_model2_triangle_1st_episode_pronouns_all_pronouns_history.png",
                out_cm_filename="lr_model2_triangle_1st_episode_pronouns_all_pronouns_cm.png",
                test_size=test_size,
                val_size=val_size)

    # model 5
    train_model(data=triangle_1st_episode_data, 
                model_type="logistic_regression", 
                optimizer=best_combination_triangle_model5[0], 
                learning_rate=best_combination_triangle_model5[1], 
                batch_size=best_combination_triangle_model5[2], 
                epochs=best_combination_triangle_model5[3], 
                output_path=out_path, 
                input_text_features=["Past_tense_all_verbs"], 
                out_filename="lr_model2_triangle_1st_episode_past_tense_all_verbs_results.txt", 
                out_history_filename="lr_model2_triangle_1st_episode_past_tense_all_verbs_history.png",
                out_cm_filename="lr_model2_triangle_1st_episode_past_tense_all_verbs_cm.png",
                test_size=test_size,
                val_size=val_size)

    # model 6
    train_model(data=triangle_1st_episode_data, 
                model_type="logistic_regression", 
                optimizer=best_combination_triangle_model6[0], 
                learning_rate=best_combination_triangle_model6[1], 
                batch_size=best_combination_triangle_model6[2], 
                epochs=best_combination_triangle_model6[3], 
                output_path=out_path, 
                input_text_features=["Negative_all_sentiment"], 
                out_filename="lr_model2_triangle_1st_episode_negative_all_sentiment_results.txt", 
                out_history_filename="lr_model2_triangle_1st_episode_negative_all_sentiment_history.png",
                out_cm_filename="lr_model2_triangle_1st_episode_negative_all_sentiment_cm.png",
                test_size=test_size,
                val_size=val_size)
    
    # model 6a
    train_model(data=triangle_1st_episode_data, 
                model_type="logistic_regression", 
                optimizer=best_combination_triangle_model6a[0], 
                learning_rate=best_combination_triangle_model6a[1], 
                batch_size=best_combination_triangle_model6a[2], 
                epochs=best_combination_triangle_model6a[3], 
                output_path=out_path, 
                input_text_features=["Positive_all_sentiment"], 
                out_filename="lr_model2_triangle_1st_episode_positive_all_sentiment_results.txt", 
                out_history_filename="lr_model2_triangle_1st_episode_positive_all_sentiment_history.png",
                out_cm_filename="lr_model2_triangle_1st_episode_positive_all_sentiment_cm.png",
                test_size=test_size,
                val_size=val_size)

    # --- TRAIN AUTOBIOGRAPHICAL/CHRONIC MODELS --- #
   
    # model 1
    train_model(data=autobiographical_chronic_data, 
                model_type="logistic_regression", 
                optimizer=best_combination_autobiographical_model1[0], 
                learning_rate=best_combination_autobiographical_model1[1], 
                batch_size=best_combination_autobiographical_model1[2], 
                epochs=best_combination_triangle_model1[3], 
                output_path=out_path, 
                input_text_features=["Pronouns_all_pronouns"], 
                out_filename="lr_model1_autobiographical_chronic_pronouns_all_pronouns_results.txt", 
                out_history_filename="lr_model1_autobiographical_chronic_pronouns_all_pronouns_history.png",
                out_cm_filename="lr_model1_autobiographical_chronic_pronouns_all_pronouns_cm.png",
                test_size=test_size,
                val_size=val_size)

    # model 2
    train_model(data=autobiographical_chronic_data, 
                model_type="logistic_regression", 
                optimizer=best_combination_autobiographical_model2[0], 
                learning_rate=best_combination_autobiographical_model2[1], 
                batch_size=best_combination_autobiographical_model2[2], 
                epochs=best_combination_autobiographical_model2[3], 
                output_path=out_path, 
                input_text_features=["Past_tense_all_verbs"], 
                out_filename="lr_model2_autobiographical_chronic_past_tense_all_verbs_results.txt", 
                out_history_filename="lr_model2_autobiographical_chronic_past_tense_all_verbs_history.png",
                out_cm_filename="lr_model2_autobiographical_chronic_past_tense_all_verbs_cm.png",
                test_size=test_size,
                val_size=val_size)

    # model 3
    train_model(data=autobiographical_chronic_data, 
                model_type="logistic_regression", 
                optimizer=best_combination_autobiographical_model3[0], 
                learning_rate=best_combination_autobiographical_model3[1], 
                batch_size=best_combination_autobiographical_model3[2], 
                epochs=best_combination_autobiographical_model3[3], 
                output_path=out_path, 
                input_text_features=["Negative_all_sentiment"], 
                out_filename="lr_model2_autobiographical_chronic_negative_all_sentiment_results.txt", 
                out_history_filename="lr_model2_autobiographical_chronic_negative_all_sentiment_history.png",
                out_cm_filename="lr_model2_autobiographical_chronic_negative_all_sentiment_cm.png",
                test_size=test_size,
                val_size=val_size)
  
    # model 3a
    train_model(data=autobiographical_chronic_data, 
                model_type="logistic_regression", 
                optimizer=best_combination_autobiographical_model3a[0], 
                learning_rate=best_combination_autobiographical_model3a[1], 
                batch_size=best_combination_autobiographical_model3a[2], 
                epochs=best_combination_autobiographical_model3a[3], 
                output_path=out_path, 
                input_text_features=["Positive_all_sentiment"], 
                out_filename="lr_model2_autobiographical_chronic_positive_all_sentiment_results.txt", 
                out_history_filename="lr_model2_autobiographical_chronic_positive_all_sentiment_history.png",
                out_cm_filename="lr_model2_autobiographical_chronic_positive_all_sentiment_cm.png",
                test_size=test_size,
                val_size=val_size)
    
    # --- TRAIN AUTOBIOGRAPHICAL/1ST EPISODE MODELS --- #

    # model 4
    train_model(data=autobiographical_1st_episode_data, 
                model_type="logistic_regression", 
                optimizer=best_combination_autobiographical_model4[0], 
                learning_rate=best_combination_autobiographical_model4[1], 
                batch_size=best_combination_autobiographical_model4[2], 
                epochs=best_combination_autobiographical_model4[3], 
                output_path=out_path, 
                input_text_features=["Pronouns_all_pronouns"], 
                out_filename="lr_model2_autobiographical_1st_episode_pronouns_all_pronouns_results.txt", 
                out_history_filename="lr_model2_autobiographical_1st_episode_pronouns_all_pronouns_history.png",
                out_cm_filename="lr_model2_autobiographical_1st_episode_pronouns_all_pronouns_cm.png",
                test_size=test_size,
                val_size=val_size)

    # model 5
    train_model(data=autobiographical_1st_episode_data, 
                model_type="logistic_regression", 
                optimizer=best_combination_autobiographical_model5[0], 
                learning_rate=best_combination_autobiographical_model5[1], 
                batch_size=best_combination_autobiographical_model5[2], 
                epochs=best_combination_autobiographical_model5[3], 
                output_path=out_path, 
                input_text_features=["Past_tense_all_verbs"], 
                out_filename="lr_model2_autobiographical_1st_episode_past_tense_all_verbs_results.txt", 
                out_history_filename="lr_model2_autobiographical_1st_episode_past_tense_all_verbs_history.png",
                out_cm_filename="lr_model2_autobiographical_1st_episode_past_tense_all_verbs_cm.png",
                test_size=test_size,
                val_size=val_size)

    # model 6
    train_model(data=autobiographical_1st_episode_data, 
                model_type="logistic_regression", 
                optimizer=best_combination_autobiographical_model6[0], 
                learning_rate=best_combination_autobiographical_model6[1], 
                batch_size=best_combination_autobiographical_model6[2], 
                epochs=best_combination_autobiographical_model6[3], 
                output_path=out_path, 
                input_text_features=["Negative_all_sentiment"], 
                out_filename="lr_model2_autobiographical_1st_episode_negative_all_sentiment_results.txt", 
                out_history_filename="lr_model2_autobiographical_1st_episode_negative_all_sentiment_history.png",
                out_cm_filename="lr_model2_autobiographical_1st_episode_negative_all_sentiment_cm.png",
                test_size=test_size,
                val_size=val_size)

    # model 6a
    train_model(data=autobiographical_1st_episode_data, 
                model_type="logistic_regression", 
                optimizer=best_combination_autobiographical_model6a[0], 
                learning_rate=best_combination_autobiographical_model6a[1], 
                batch_size=best_combination_autobiographical_model6a[2], 
                epochs=best_combination_autobiographical_model6a[3], 
                output_path=out_path, 
                input_text_features=["Positive_all_sentiment"], 
                out_filename="lr_model2_autobiographical_1st_episode_positive_all_sentiment_results.txt", 
                out_history_filename="lr_model2_autobiographical_1st_episode_positive_all_sentiment_history.png",
                out_cm_filename="lr_model2_autobiographical_1st_episode_positive_all_sentiment_cm.png",
                test_size=test_size,
                val_size=val_size)

if __name__=="__main__":

    # --- ARGUMENT PARSER --- #

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--out_path', type=str, required=False, default="../output/baseline",
                        help='define path to output dir')
    
    parser.add_argument('--test_size', type=float, required=False, default=0.20,
                        help='define size of test split as float, e.g. 0.20')

    parser.add_argument('--val_size', type=float, required=False, default=0.13,
                        help='define size of val split as float, e.g. 0.25')
    
    args = parser.parse_args()

    # -- RUN MAIN FUNCTION --- #

    main(out_path = args.out_path,
         test_size=args.test_size,
         val_size=args.val_size)
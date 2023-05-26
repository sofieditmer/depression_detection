"""
SCRIPT TO PERFORM GRID SEARCH
"""

# --- DEPENDENCIES --- #

import sys, os
sys.path.append(os.path.join(".."))
import argparse
from tqdm import tqdm
from datetime import datetime
from utils.model_utils import make_train_val_test_split, get_hpcombinations
from utils.logistic_regression import LogisticRegression

# --- MAIN FUNCTION --- #

def main(model_type, filename, input_text_features, output_path, data, test_size, val_size):
    """
    Performs grid search on specified models based on defined hyperparameter combinations. The best combination is based on the lowest validation loss.

    Args:
        - model_type (str): type of model
        - filename (str): name of output file containing results of grid search
        - input_text_features (list[str]): features on which the model should be trained
        - output_path (str): path to output folder
        - data (df): data on which the model should be trained

    Returns:
        - best_combination: the hyperparameter combination that returns the lowest validation loss.
    """

    # --- PREPARATIONS --- #
    
    # define output path
    output_path = os.path.join(output_path, "grid_search")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # write info to file
    output_file = os.path.join(output_path, filename)
    with open(output_file, 'a') as file:
        file.write(f"\nRun from {datetime.now()}\n")
        file.write(f"Results of Grid Search for {model_type}: \n")
        file.write(f"Order of Parameters: optimizer, learning_rate, batch_size, epochs \n\n")
    
    # prepare data
    X_train, X_val, _, y_train, y_val, _ = make_train_val_test_split(data=data, 
                                                                    test_size=test_size, 
                                                                    val_size=val_size, 
                                                                    text_features=input_text_features,
                                                                    seed=1)
    
    # prepare list to save losses
    val_losses = []
    
    # get hyperparameter combinations
    hpcombiantions = get_hpcombinations()
    
    # --- TRAIN MODELS --- #
    
    # loop through the possible hyperparameter combinatons:
    for combination in tqdm(hpcombiantions):
        
        # define logistic regression regression
        if model_type == "logistic_regression":
            optimizer, learning_rate, batch_size, epochs = combination[:4]
            model = LogisticRegression(optimizer, learning_rate, batch_size, epochs)
            
        # train the specified model
        model.train(X_train, y_train, X_val, y_val, verbose=0)
        
        # append values to validation losses to find the best one
        val_losses.append(model.val_loss)
        
        # save results to file
        with open(output_file, 'a') as file:
            file.writelines([f"{combination}\n", 
                             f"train_loss: {model.train_loss}, val_loss: {model.val_loss}\n\n"])
            
    # --- FIND BEST MODEL --- #
        
    # get the best lowest val loss
    min_value = min(val_losses)
    
    # find the corresponding combinaton of parameters
    best_idx = val_losses.index(min_value)
    best_combination = hpcombiantions[best_idx]
    
    # get the parameters of the best model for linear regression
    if model_type == "logistic_regression":
        optimizer, learning_rate, batch_size, epochs = best_combination[:4]

    # append best model to file
    with open(output_file, 'a') as file:
            file.write("\n---------\n")
            file.write("BEST MODEL\n")
            file.writelines([f"{best_combination}\n", f"val_loss: {min_value}"])
            file.write("\n---------\n")

    return best_combination

if __name__=="__main__":

    # --- ARGUMENT PARSER --- #
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_type', type=str, required=True,
                        help='logistic_regression')
    
    parser.add_argument('--filename', type=str, required=True,
                        help='define filename of grid search results as txt')

    parser.add_argument('--output_path', type=str, required=True,
                        help='define path to output directory')
    
    parser.add_argument('--test_size', type=float, required=True,
                        help='define size of test split, e.g. 0.20')
    
    parser.add_argument('--val_size', type=float, required=True,
                        help='define size of val split, e.g. 0.25')
    
    args = parser.parse_args()
    
    # --- RUN MAIN FUNCTION --- #

    main(model_type=args.model_type,
         output_path=args.output_path,
         filename=args.filename,
         test_size=args.test_size,
         val_size=args.val_size)
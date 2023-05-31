#!/usr/bin/env python

"""
Script for extracting low-level text markers as well as word and sentence embeddings from transcribed clinical interviews with chronic and 1st episode MDD patients as well as controls.

Overview of text markers:
    - First person singular pronoun use (the ratio between number of Danish first person singular pronouns and all Danish pronouns used)
    - Use of past tense verbs (the ratio between number of Danish past tense verbs and all Danish verbs used)
    - Use of negative and positive words (the ratio between number of negative words used and all affective words)

Overview of embeddings:
    - Word embeddings: extracted using pretrained Danish FastText model
    - Sentence embeddings: extracted using pretrained Danish sentence transformer model 
"""

# ---- DEPENDENCIES ---- #
import os
import sys
import argparse
import numpy as np
sys.path.append(os.path.join(".."))
from utils.preprocessing_utils import prepare_data, extract_text_features, extract_embeddings

# ---- MAIN FUNCTION ---- #

def main(sentiment_threshold=3, output_path="../data/features", output_filename="text_features.csv"):
    """
    Extracts low-level text markers as well as word and sentence embeddings from transcripts and saves features in .csv file in output/ folder 

    Args:
        sentiment_threshold (int): value indicating the threshold for determining negative and positive words on the 11-item scale (from -5 to +5) used by Danish sentiment dictionary SENTIDA
        output_path (str): path to output folder
        output_filename (str): name of output file containing text feature.

    Returns:
        - text_features (.csv): contains extracted features as well as subject information + transcript
        - X_all_word_embeddings (.npy): all extracted word embeddings 
        - X_all_sentence_embeddings (.npy): all extracted sentence embeddings
    """

    # --- PREPARE DATA AND DF --- #

    print("[INFO] Preparing data...")

    df = prepare_data(data_path = "../data/", remove_filler_words=False)

    # --- EXTRACT LOW-LEVEL FEATURES FROM TRANSCRIPTS --- #

    print("[INFO] Extracting hand-engineered text features...")
    
    df_features = extract_text_features(df, sentiment_threshold=sentiment_threshold)

    # --- EXTRACT EMBEDDINGS --- #

    print("[INFO] Extracting word and sentence embeddings...")

    all_word_embeddings, all_sentence_embeddings = extract_embeddings(df)

    # --- SAVE ALL EXTRACTED FEATURES --- #

    print(f"[INFO] Saving all extracted features in {output_path}...")

    # save low-level text features as .csv file
    df_features.to_csv(os.path.join(output_path, output_filename))

    # save embeddings as .npy files
    np.save(os.path.join(output_path, "embeddings/all_word_embeddings.npy"), all_word_embeddings)
    np.save(os.path.join(output_path, "embeddings/all_sentence_embeddings.npy"), all_sentence_embeddings)

    print(f"[INFO] Done! All extracted features are saved in {output_path}")

if __name__=="__main__":

    # -- ARGUMENT PARSER --- #

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--sentiment_threshold', type=int, default=3,
                       help="define threshold for positive and negative words") 
    
    parser.add_argument('--output_path', type=str, default="../data/features",
                       help="define path to output folder") 
    
    parser.add_argument('--output_filename', type=str, default="text_features.csv",
                       help="define name of output file") 
    
    args = parser.parse_args()
                        
    # --- RUN MAIN FUNCTION --- #

    main(sentiment_threshold = args.sentiment_threshold,
         output_path=args.output_path,
         output_filename=args.output_filename)


















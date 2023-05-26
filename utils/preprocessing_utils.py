#!/usr/bin/env python

"""
Preprocessing functions for scripts in src/ folder.
"""

# ---- DEPENDENCIES ---- #
import os
import dacy
nlp = dacy.load("small")
import numpy as np
from spacy.language import Language
import pandas as pd
import fasttext
from sentence_transformers import SentenceTransformer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from scipy.stats import iqr

# ---- PREPROCESSING FUNCTIONS ---- #

def prepare_data(data_path, remove_filler_words=False):
    """
    Takes path to folder containing transcripts and outputs IDs and filepath names.

    Args:
        - data_path (str): path to data folder

    Returns:
        - df (dataframe): dataframe containing prepared data
    """
    # make list of filenames, filepaths, and transcripts
    file_names = []
    file_paths = []
    list_transcripts = []

    for file in os.listdir(os.path.join(data_path, "animated_triangles")):
        if file.endswith(".txt"):
            file_names.append(file)
            file_paths.append(os.path.join(data_path, "animated_triangles", file))

    for file in os.listdir(os.path.join(data_path, "autobiographical")):
        if file.endswith(".txt"):
            file_names.append(file)
            file_paths.append(os.path.join(data_path, "autobiographical", file))

    # extract IDs from file names
    IDs = []
    for name in file_names:
        ID = name.split(".")[0]
        IDs.append(ID)

    # prepare transcripts
    list_filler_words = []

    for file in file_paths:

        # print filename
        print(f"Filename: {file}")

        # open file
        open_file = open(file, encoding="utf8")

        # load content
        transcript = open_file.read()

        # lowercase
        transcript = transcript.lower()

        # remove line breaks and tailing blanks
        transcript = transcript.replace('\n', ' ').strip()

        # count number of filler words for each transcript 
        filler_words = ["øh", "øhm", "øhmm", "mmh", "mh", "hm", "hmm"]
        n_fills = sum([transcript.count(i) for i in filler_words])

        # append to list
        list_filler_words.append(n_fills)

        # remove filler words such as "øhm" and "øh" etc.
        if remove_filler_words == True:
            transcript = transcript.replace("øhmm", "").replace("øhm", "").replace("øh", "").replace("mmh", "").replace("mh", "").replace("hmm", "").replace("hm", "")

        # append raw transcript to list 
        list_transcripts.append(transcript)

    # make empty df
    df = pd.DataFrame({"Filename": [],
                        "ID": [],
                        "Diagnosis": [],
                        "binary_label": [],
                        "data_type": [], 
                        "Pronouns_all_pronouns": [],
                        "Pronouns_all_words": [],
                        "Past_tense_all_words": [],
                        "Past_tense_all_verbs": [],
                        "Negative_all_words": [],
                        "Positive_all_words": [],
                        "Negative_all_sentiment": [],
                        "Positive_all_sentiment": [],
                        "Filler_words": [], 
                        "Transcript": []
                        })
    
    # append filenames, IDs and transcripts to df
    df["Filename"] = file_paths
    df["ID"] = IDs
    df["Transcript"] = list_transcripts

    # make diagnosis and binary label columns
    diagnoses = []
    binary_labels = []
    for ID in IDs:
        if ID[:3] == "dpc":
            diagnoses.append("chronic")
            binary_labels.append(1)
        elif ID[:2] == "dc":
            diagnoses.append("control")
            binary_labels.append(0)
        else:
            diagnoses.append("1st_episode")
            binary_labels.append(1)

    # make data_type column
    data_type = []
    for filepath in file_paths:
        first_split = filepath.split("\\")
        second_split = first_split[0].split("/")[2]
        if second_split == "animated_triangles":
            data_type.append("Triangle")
        else:
            data_type.append("Autobiographical")
        
    # append diagnoses and binary labels to df
    df["Diagnosis"] = diagnoses
    df["binary_label"] = binary_labels
    df["data_type"] = data_type
    df["Filler_words"] = list_filler_words

    return df

def prepare_triangles_to_txt(path_to_csv):
    """
    Takes a path to a csv file containing all triangle data and splits data into independent csv file
    saves files in folder data/triangle

    Args:
        - path_to_csv (str): path to .csv file containing animated triangle data

    Returns:
        - triangles (df): dataframe containing preprocessed triangle data
    """
    # load triangle data via csv
    triangles = pd.read_csv(path_to_csv, sep=';' , encoding='latin-1')

    # Remove time columns
    triangles = triangles.drop(columns = ['TimeStart', 'TimeStop'])

    # make binary label and diagnosis based on file columns
    labels = []
    diagnosis = []
    ids = []
    for sentence in triangles["File"]:
        unique_id = str(sentence.split('.')[0] + "_tri")
        ids.append(unique_id)
        if sentence[:3] == "dpc":
            labels.append(1)
            diagnosis.append("chronic")
        elif sentence[:2] == "dc":
            labels.append(0)
            diagnosis.append("control")
        else:
            diagnosis.append("1st_episode")
            labels.append(1)

    # Insert ids and labels in df        
    triangles['ids'] = ids
    triangles['labels'] = labels

    start = 0
    texts = []
    unique_ids = []
    for index, id in enumerate(triangles["ids"][1:len(triangles)]):
        if id == triangles["ids"][index] and  index != (len(triangles["ids"][1:])-1):
            pass
        else:
            if index == (len(triangles["ids"][1:])-1):
                index = index +1
            else:
                pass
            group = triangles["Sentence"][start:index+1]
            start = index+1
            texts.append(group)
            unique_ids.append(triangles["ids"][index])

            # save as txt file
            with open(str('../data/triangle/'+f'{triangles["ids"][index]}'+'.txt'), "w", newline="") as f:
                for item in group:
                    f.write("%s\n" % item)

    return triangles

def extract_text_features(df, sentiment_threshold=3):
    """
    Takes df and sentiment threshold (a value between -5 and 5 that defines the sentiment of the words) as input and extract text features.

    Args:
        - df (dataframe): dataframe containing all data
        - sentiment_threshold (int): value defining the treshold based on which sentiment words are extracted

    Returns:
        - df (dataframe): dataframe containing all low-level text features
    """
    # make empty lists to append to later
    list_ratio_pronouns = []
    list_ratio_words_pronouns = []
    list_ratio_verbs = []
    list_ratio_words_past_tense = []
    list_negative_ratio_sentiment_words = []
    list_positive_ratio_sentiment_words = []
    list_negative_ratio_all_words = []
    list_positive_ratio_all_words = []

    # loop through transcripts
    for index, transcript in enumerate(df["Transcript"]):

        # print filename
        filename = df["ID"][index]
        print(f"Filename: {filename}")

        # run transcript through dacy
        doc = nlp(transcript)

        # ---- 1st person pronoun use ---- #

        print("[INFO] Extracting 1st person pronoun use...")

        # define lists of first person Danish singular pronouns + list of all Danish pronouns
        first_person_pronouns_list = ["jeg", "mig", "min", "mit", "mine"]
        all_pronouns_list = ["jeg", "mig", "min", "mit", "mine", "du", "dig", "din", "dit", "dine", "han", "hun", "den", "det", "de", "ham", "hende", "dem", "sin", "hans", "hendes", "dens", "dets", "deres", "sit", "sine", "vi", "os", "jeres", "vores", "vor", "vort", "vore", "i"]
        all_pronouns_list_no_i = ["jeg", "mig", "min", "mit", "mine", "du", "dig", "din", "dit", "dine", "han", "hun", "den", "det", "de", "ham", "hende", "dem", "sin", "hans", "hendes", "dens", "dets", "deres", "sit", "sine", "vi", "os", "jeres", "vores", "vor", "vort", "vore"]
        
        # extract first person singular pronouns and all pronouns from transcript using DaCy
        all_pronouns = []
        all_pronouns_filtered = []
        first_person_pronouns = []

        # extract all pronouns and determiners
        for token in doc:
            if token.pos_ == "PRON":
                all_pronouns.append(token)
            elif token.pos_ == "DET" and token.text in all_pronouns_list_no_i:
                all_pronouns.append(token)

        # filter pronouns list and append to first person singular pronouns list
        for pronoun in all_pronouns:
            if pronoun.text in all_pronouns_list:
                all_pronouns_filtered.append(pronoun)
            if pronoun.text in first_person_pronouns_list:
                first_person_pronouns.append(pronoun)

        # count pronouns and calculate ratios
        n_first_persons_pronouns = len(first_person_pronouns)
        n_pronouns = len(all_pronouns_filtered)
        n_words = len(doc)
        ratio_pronouns = n_first_persons_pronouns/n_pronouns
        ratio_words_pronouns = n_first_persons_pronouns/n_words

        # ---- Use of past tense ---- #

        print("[INFO] Extracting use of past tense verbs...")

        # extract verbs and past-tense verbs
        all_verbs = []
        past_tense_verbs = []

        for token in doc:
            if token.pos_ == "VERB":
                all_verbs.append(token)
                if token.morph.get("Tense") == ['Past']:    
                    past_tense_verbs.append(token)

        # save verbs for sanity check
        with open(f"../output/all_verbs_{filename}.txt", "w") as output:
            output.write(str("All verbs:"))
            output.write(str(all_verbs))
            output.write(str("Past tense verbs:"))
            output.write(str(past_tense_verbs))

        # count verbs and calculate ratios
        n_past_tense_verbs = len(past_tense_verbs)
        n_verbs = len(all_verbs)
        ratio_verbs = n_past_tense_verbs/n_verbs
        ratio_words_past_tense = n_past_tense_verbs/n_words

        # ---- Positive and negative word use ---- #

        print("[INFO] Extracting use of positive and negative words...")

        # lemmatize transcript
        lemmas = []
        for token in doc:
            lemmas.append(token.lemma_)

        # get SENTIDA2 list of sentiment words
        sentiment_words = pd.read_csv("../utils/sentidav2_lemmas.csv")

        # extract only words that are beyond thresholds
        df_negative = sentiment_words.loc[sentiment_words['score'] <= -sentiment_threshold]
        negative_word_list = list(df_negative["word"])
        df_positive = sentiment_words.loc[sentiment_words['score'] >= sentiment_threshold]
        positive_word_list = list(df_positive["word"])

        # extract negative and positive words used in transcript
        negative_words_used = []
        for word in lemmas:
            if word in negative_word_list:
                negative_words_used.append(word)

        positive_words_used = []
        for word in lemmas:
            if word in positive_word_list:
                positive_words_used.append(word)

        # count number of negative and positive words and calculate ratios
        n_negative_words = len(negative_words_used)
        n_positive_words = len(positive_words_used)

        if n_negative_words != 0:
            negative_ratio_sentiment_words = n_negative_words/(n_negative_words+n_positive_words)
            negative_ratio_all_words = n_negative_words/n_words
            
        else:
            negative_ratio_sentiment_words = 0
            negative_ratio_all_words = 0

        if n_positive_words != 0: 
            positive_ratio_sentiment_words = n_positive_words/(n_negative_words+n_positive_words)
            positive_ratio_all_words = n_positive_words/n_words
        else:
            positive_ratio_sentiment_words = 0
            positive_ratio_all_words = 0

        # append to empty lists
        list_ratio_pronouns.append(ratio_pronouns)
        list_ratio_words_pronouns.append(ratio_words_pronouns)
        list_ratio_verbs.append(ratio_verbs)
        list_ratio_words_past_tense.append(ratio_words_past_tense)
        list_negative_ratio_sentiment_words.append(negative_ratio_sentiment_words)
        list_positive_ratio_sentiment_words.append(positive_ratio_sentiment_words)
        list_negative_ratio_all_words.append(negative_ratio_all_words)
        list_positive_ratio_all_words.append(positive_ratio_all_words)
        
    # append list of text features to df
    df["Pronouns_all_pronouns"] = list_ratio_pronouns
    df["Pronouns_all_words"] = list_ratio_words_pronouns
    df["Past_tense_all_words"] = list_ratio_words_past_tense
    df["Past_tense_all_verbs"] = list_ratio_verbs
    df["Negative_all_sentiment"] = list_negative_ratio_sentiment_words
    df["Positive_all_sentiment"] = list_positive_ratio_sentiment_words
    df["Negative_all_words"] = list_negative_ratio_all_words
    df["Positive_all_words"] = list_positive_ratio_all_words

    return df

def extract_embeddings(df):
    """
    Takes IDs and filepaths as input and extracts word and sentence embeddings from transcripts.

    Args:
        - IDs (list[str]): subject IDs
        - file_paths (list[str]): list of filepaths to all transcripts

    Returns:
        - all_word_embeddings (array): numpy array containing all extracted word embeddings for all transcripts
        - all_sentence_embeddings (array): numpy array containing all extracted sentence embeddings for all transcripts
    """
    # --- LOAD MODELS --- #

    print("[INFO] Loading pretrained models...")

    # load Danish fasttext model (to be able to extract static word embeddings)
    ft_model = fasttext.load_model("../data/features/embeddings/fastText_embeddings.bin")

    # load Danish sentence transformer model (to be able to extact dynamic text embeddings)
    print("[INFO] Loading Sentence Transformer model...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # --- CUSTOMIZE DACY PIPELINE --- #

    # customizing DaCy parser to yield shorter sentences
    @Language.component('custom_boundaries')
    def set_custom_boundaries(doc):
        for token in doc[:-1]:
            if token.text == "men" or token.text == "." or token.text == ",":
                doc[token.i].is_sent_start = True
        return doc
    
    # add function to nlp pipeline before inherent parser
    nlp.add_pipe("custom_boundaries", first=True)
    nlp.pipeline

    # --- EXTRACT EMBEDDINGS --- #

    # define overall lists to be appended to 
    all_word_embeddings = []
    all_sentence_embeddings = []
    all_iqr_word_embeddings = []
    all_iqr_300dim_word_embeddings = []
    all_iqr_sentence_embeddings = []
    all_iqr_300dim_sentence_embeddings = []

    print("[INFO] Extracting word and sentence embeddings for each transcript...")

    # iterate over transcripts
    for index, transcript in enumerate(df["Transcript"]):
        
        # print filename
        filename = df["ID"][index]
        print(f"Filename: {filename}")

        # --- EXTRACT WORD EMBEDDINGS --- #

        # make list to be appended to 
        word_embeddings_transcript = []

        # iterate over words in transcript
        for word in transcript.split():

            # extract embedding for word
            embed = ft_model.get_word_vector(f"{word}")

            # append to list of embeddings
            word_embeddings_transcript.append(embed)

        # calculate interquartile range per transcript/subject
        iqr_transcript_word_emb = iqr(word_embeddings_transcript)
        all_iqr_word_embeddings.append(iqr_transcript_word_emb)

        iqr_transcript_300dim_word_emb = iqr(word_embeddings_transcript, axis=0)
        all_iqr_300dim_word_embeddings.append(iqr_transcript_300dim_word_emb)

        # take element-wise average of word embeddings per transcript
        embeddings_transcript_mean = np.mean(word_embeddings_transcript, axis=0)

        # append word embeddings to overall lists
        all_word_embeddings.append(embeddings_transcript_mean)

        # --- EXTRACT SENTENCE EMBEDDINGS --- #

        # split transcript into sentences
        doc = nlp(transcript)
        sentences = []
        for sent in doc.sents:
            sentences.append(sent.text)

        # remove sentences that are fewer or equal to one word 
        for index, sent in enumerate(sentences):
            if len(sent.split()) <= 1:
                del sentences[index]

        # vectorize sentences with sentence transformer model
        sentence_embeddings = []
        for sentence in sentences:
            sentence_embedding = model.encode(sentence)
            sentence_embeddings.append(sentence_embedding)

        # calculate interquartile range per transcript/subject
        iqr_transcript_sent_emb = iqr(sentence_embeddings)
        all_iqr_sentence_embeddings.append(iqr_transcript_sent_emb)

        iqr_transcript_300dim_sent_emb = iqr(sentence_embeddings, axis=0)
        all_iqr_300dim_sentence_embeddings.append(iqr_transcript_300dim_sent_emb)

        # take element-wise average of sentence embeddings per transcript 
        sentence_embeddings_mean = np.mean(sentence_embeddings, axis=0)

        # append to overall list containing all sentence embeddings for all transcripts
        all_sentence_embeddings.append(sentence_embeddings_mean)

    # save interquartile ranges for word and sentence embeddings
    np.save("../data/features/embeddings/interquartile_ranges_transcript_word_embeddings.npy", all_iqr_word_embeddings)
    np.save("../data/features/embeddings/interquartile_ranges_300dim_transcript_word_embeddings.npy", all_iqr_300dim_word_embeddings)
    np.save("../data/features/embeddings/interquartile_ranges_transcript_sentence_embeddings.npy", all_iqr_sentence_embeddings)
    np.save("../data/features/embeddings/interquartile_ranges_300dim_transcript_sentence_embeddings.npy", all_iqr_300dim_sentence_embeddings)

    return all_word_embeddings, all_sentence_embeddings

if __name__=="__main__":
    pass
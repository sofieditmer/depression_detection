# Linguistic markers of depression: Assessing the generalizability of machine learning models

## Project Description
This repository contains the contents of a Master's degree in Cognitive Science thesis project concerned with assessing the generalizability of machine learning models for depression detection in transcribed clinical interviews with patients diagnosed with chronic and first-episode MDD as well as controls.

Two overall foci were examined by each of the repository contributors:

1. **Patient Heterogeneity**: Do models trained on chronically depressed patients generalize when tested on first-episode depressed patients and vice versa? This was examined by Sofie Ditmer.

2. **Task Heterogeneity**: Do models trained on the IPII task generalize when tested on the FHA task and vice versa? This was examined by Katrine Nymman.

## Data Description
The data employed for this project consisted of transcribed audio files originally collected in 2011 as part of a researh project on higher-order social cognition conducted by Nicolai Ladegaard in 2014 (Aarhus University Hospital). Data was obtained from MDD patients and healthy controls undergoing to interviews:

1. **The Indiana Psychiatric Illness Interview (IPII):** a clinical interview conducted by a psychiatrist in which the subject is asked to tell their life story. 

2. **The Frith-Happé Animations (FHA) task:** the subject is asked to describe a series of animations depicting non-verbal, computer-based triangles. 

An overview of the number of subjects in each data category is presented below:

|   | Indiana Psychiatric Illness Interview (IPII) | Frith-Happé Animations (FHA) |  
| ------------- | :-------------: | :-------------: |
| Chronic MDD patients  | 19  | 27 |
| First-episode MDD patients  | 41  | 40 |
| Healthy control subjects | 43 | 42

## Project Pipeline
 The project consisted of three overall analysis steps:

1. **Baseline: Examining the signal present in hand-engineered text features with logistic regression models**: Baseline logistic regression models were constructed to examine whether the use of particular text features by MDD patients can be used to predict the presence of depression. A systematic literature review was conducted and found that particularly the use of first-person singular pronouns, past-tense verbs, as well as negative and positive words have been shown to be significant markers of depression. These text markers were therefore used as features in baseline logistic regression models to assess whether the transcribed speech of depressed patients would depict a strong enough signal to be able to accurately predict depression across subject groups and task types.

2. **Using XGBoost models to predict depression from combined hand-engineered features as well as word and sentence embeddings**: The extreme gradient boosting framework, XGBoost, was employed examine whether higher-level language representations of the clinical transcripts would increase performance and generalize to a higher extent. For each subject, static FastText word embeddings and BERT sentence embeddings were extracted and used as predictive features in the XGBoost models to predict diagnosis. A model with all three hand-engineered text markes described in the previous section was likewise constructed to assess whether the XGBoost framework would make a difference in predictive performance.

3. **Using transformer-based models to predict depression:** To examine whether employing a transformer-based model would yield an increase in predictive performance compared to the XGBoost framework, the body of a pre-trained transformer-based model was fine-tuned on the current dataset.

## Repository Structure
```
|-- jupiter_notebooks/              # Directory containing the notebooks used for data exploration and visualization                                   

|-- output/                         # Directory containing the results output from the scripts

|-- src/                            # Directory containing the main project scripts
                                    # See README.md in directory for detailed information

|-- utils/                          # Directory containing utility functions used in the main scripts
                                    # See README.md in directory for detailed information
|-- .gitignore                                      
|-- install-requirements.sh         # Bash script for installing necessary dependencies
|-- requirements.txt                # Necessary dependencies to run scripts
|-- README.md                       # Main README file for repository
```

## Usage
**!** The scripts have been tested on MacOS and Windows using Python 3.9.10.
Due to ethical and legal concerns, the data of this project cannot be shared on this repository. Nevertheless, scripts and directions on how the scripts were used are provided in the README.md files of the subdirectories. 

## Contact
This code for this project was developed by Katrine Nymann (201807243@post.au.dk) & Sofie Ditmer (201805308@post.au.dk)
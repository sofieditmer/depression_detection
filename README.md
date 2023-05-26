# Text-based markers of depression: Assessing the generalizability of machine learning models

## Project Description
This repository contains the contents of a Master's degree in Cognitive Science thesis project concerned with assessing the generalizability of machine learning models for depression detection in transcribed clinical interviews with patients diagnosed with chronic and first-episode MDD as well as controls.

## Data Description
The data employed for this project consisted of transcribed audio files originally collected in 2011 as part of a researh project on higher-order social cognition conducted by Nicolai Ladegaard in 2014 (Aarhus University Hospital). MDD patients and healthy controls were asked to complete two separate tasks:

1. **Autobiographical free-speech task:** a clinical interview conducted by a psychiatrist in which the subject is asked to tell their life story. 

2. **The animated triangles task:** the subject is asked to describe a series of animations depicting non-verbal, computer-based triangles. 

An overview of the number of subjects in each data category is presented below:

|   | Autobiographical free-speech task | The animated triangles task |  
| ------------- | :-------------: | :-------------: |
| Chronic MDD patients  | 19  | 27 |
| First-episode MDD patients  | 41  | 40 |
| Healthy control subjects | 43 | 42

## Project Pipeline
 The project consisted of three overall analysis steps:

1. **Baseline: Examining the signal present in low-level text features with logistic regression models**: Baseline logistic regression models were constructed to examine whether the use of low-level text features by MDD patients can be used to predict the presence of depression. A systematic literature review was conducted and found that particularly the use of first-person singular pronouns, past-tense verbs, as well as negative words have been shown to be significant markers of depression. These low-level markers were therefore used as features in baseline logistic regression models to assess whether the transcribed speech of depressed patients would depict a strong enough signal to be able to accurately predict depression. 12 baseline logistic regression models were constructed to assess signal present in the three low-level features (use of first-person singular pronouns, use of past-tense verbs, and use of negative words) to predict diagnosis across the two tasks described in the previous section as well as across chronic and first-episode MDD patients. 

2. **Using XGBoost models to predict depression from word and sentence embeddings**: Given that the signal present in low-level text features did not prove significant enough to provide well-performing baseline models, the extreme gradient boosting framework, XGBoost, was employed to increase performance. For each subject, static FastText word embeddings and dynamic BERT sentence embeddings were created and used as predictive features in the XGBoost models to predict diagnosis. A model with all three low-level text markes described in the previous section was likewise constructed to assess whether the XGBoost framework would make a difference in predictive performance. In this way, 12 XGBoost models were constructed. 

3. **Using transformer-based models to predict depression from sentence embeddings:** To examine whether more complex transformer-based models would yield an even better predictive performance compared to the XGBoost framework lalalal...

## Results

## Repository Structure
```
|-- src/                            # Directory containing the main project scripts
                                    # See README.md in directory for detailed information

|-- utils/                          # Directory containing utility functions used in the main scripts
                                    # See README.md in directory for detailed information
                                      
|-- output/                         # Directory containing the output produced by the main scripts
|-- install-requirements.sh         # Bash script for installing necessary dependencies
|-- requirements.txt                # Necessary dependencies to run scripts
|-- README.md                       # Main README file for repository
```

## Usage
**!** The scripts have been tested on MacOS and Windows using Python 3.9.10.
Due to ethical and legal concerns, the data of this project cannot be shared on this repository. Nevertheless, scripts and directions on how the scripts were used are provided in the README.md files of the subdirectories. 

## Contact
This code for this project was developed by Katrine Nymann (201807243@post.au.dk) & Sofie Ditmer (201805308@post.au.dk)
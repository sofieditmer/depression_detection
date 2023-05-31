# Content Description of ```src/```

The ```src/``` folder contains the main scripts of this project which include the scripts for
- Initial exploration of the data 
- Baseline logistic regression models
- XGBoost models 
- Transformer models

## Directory Structure
```
|-- data_exploration.ipynb                # Notebook containing the initial exploration of the data
|-- feature_extraction.py                 # Script for extracting the low-level text markers as well as the word and sentence embeddings from the transcribed audio files
|-- train_baseline_models.py              # Script for training the 12 baseline logistic regression models          
|-- train_xgboost_models.py               # Script for training the 12 XGBoost models
|-- train_transformer_models.py           # Script for training the 4 transformer models
|-- run_seeds_xgboost.py                  # Script for training the 12 XGBoost models on five randomly chosen seeds. 
|-- README.md                             # README file providing information of the src/ folder structure
```

## Usage

Even though the data could not be provided, the following steps describe how to run the scripts within the ```src/``` folder. Note that in order to run the scripts, one need to first install the relevant packages listed in the ```requirements.txt``` file. Information on this can be found in the main ```README.md``` file on the main page.

**1. Perform Feature Extraction**

To extract the low-level text markers as well as the FastText word embeddings and BERT sentence embeddings the ```feature_extraction.py``` script can be run in the following manner:

```
cd src/
python3 feature_extraction.py
```

The features are saved as ```text_features.csv``` in the ```output/``` folder.


**2. Training Baseline Logistic Regression Models**

To train the 12 baseline logistic regression models which includes running grid search to define the optimal hyperparameters the ```train_baseline_models.py``` script can be run in the following way:

```
cd src/
python3 train_baseline_models.py 
```
The results of the grid search are saved in ```output/baseline/grid_search/``` while the model performances are saved in ```output/baseline/logistic_regression/``` folder.


**3. Training XGBoost Models**

To train the 12 XGBoost models which includes running a random search to define the optimal hyperparameters for each model the ```train_xgboost_models.py``` script can be run in the following way:

```
cd src/
python3 train_xgboost_models.py
```

The output of the XGBoost model training can be found in ```output/xgboost/``` folder.


**4. Training Transformer Models**

To train the transformer models, which requires a lot of RAM and should preferably by done on an external GPU, the following code can be executed:

```
src/
python3 train_transformer_models.py
```

The outputs of the transformer models are saved in ```output/transformer/```.


    

    

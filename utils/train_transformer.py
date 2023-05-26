#!/usr/bin/env python

"""
Utility script for transformer models
"""

# ---- DEPENDENCIES ---- #
import os
import sys
sys.path.append(os.path.join(".."))
from utils.model_utils import prepare_transformer_data, make_data_loader_objects, CustomModel, compute_metrics
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, AutoTokenizer, EarlyStoppingCallback
import torch
import wandb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import joblib
import pandas as pd

# ---- MAIN FUNCTION ---- #

def train_and_evaluate_transformer(train_dataloader, eval_dataloader, test_dataloader, gen_test_dataloader, out_path, out_filename_cr, out_filename_cf, sweep_project_name, n_sweeps, n_epochs, test_size, val_size):
    """
    ADD DESCRIPTION
    """

    # --- SETUP WEIGHTS AND BIASES --- #
    
    # set accuracy variable for later use
    accuracy = 0
    
    # login to wandb to log results
    wandb.login()

    # define sweep config for hyperparameter optimization
    sweep_config = {
        "name" : sweep_project_name,
        "method" : "bayes",
            "metric": {
                'name': 'eval_loss',
                'goal': 'minimize'}
    }

    # define hyperparameter space as dict
    parameters_dict = {
        "learning_rate": {
            "min": 1e-5,
            "max": 5e-5
        },
        "batch_size": {
            "values": [8, 16, 32, 64]
        },
        "weight_decay": {
            "min": 0.0,
            "max": 0.3
        }
    }

    # add parameters dict to sweep confict 
    sweep_config['parameters'] = parameters_dict
    
    # define sweep id
    sweep_id = wandb.sweep(sweep_config, project=sweep_project_name)
    
    # intialize custom transformer model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_model = CustomModel(checkpoint='chcaa/dfm-encoder-large-v1',
                                    num_labels=2).to(device)
    
    # --- DEFINE TRAIN FUNCTION --- #
    
    def train():

        # initialize project
        with wandb.init():
            
            # create internal variables
            accuracy_internal = accuracy
            model_name = sweep_project_name
            
            # define config
            config_internal = wandb.config
    
            # define training arguments
            training_args = TrainingArguments(
                output_dir='./results',
                report_to='wandb',
                evaluation_strategy="steps",
                save_strategy="steps",
                learning_rate=config_internal.learning_rate,
                num_train_epochs=10,
                per_device_train_batch_size=config_internal.batch_size,  
                per_device_eval_batch_size=config_internal.batch_size,   
                warmup_steps=500,                
                weight_decay=config_internal.weight_decay,              
                logging_dir='./logs',            
                logging_steps=1,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                load_best_model_at_end=True)
 
            # add early stopping to avoid overfitting on training data
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)]

            # initialize Trainer
            trainer = Trainer(
                model=transformer_model,                         
                args=training_args,                  
                train_dataset=train_dataloader.dataset,
                eval_dataset=eval_dataloader.dataset,
                compute_metrics=compute_metrics,
                callbacks=callbacks)

            # train model
            trainer.train()
            
            # extract evaluation metrics on evaluation dataset
            metrics = trainer.evaluate(eval_dataset=eval_dataloader.dataset)
            
            # if the evaluation accuracy exceeds 0
            if metrics.get("eval_accuracy") > accuracy_internal:
                
                # save model
                trainer.save_model(f"../output/transformers/best_models/{model_name}")
                
                # overwrite accuracy
                accuracy_internal = metrics.get("eval_accuracy")
                
    # extract sweep id
    sweep_id = wandb.sweep(sweep_config, 
                        project=sweep_project_name)

    # run hyperparameter optimization
    wandb.agent(sweep_id,
                project=sweep_project_name,
                function=train,
                count=n_sweeps)
    
    # --- SAVE PARAMETERS --- #
    
    # extract sweep from wandb
    api = wandb.Api()
    sweep = api.sweep(f"thesis_kat_sofie/{sweep_project_name}/{sweep_id}")

    # get best run parameters
    best_run = sweep.best_run(order="eval_accuracy")
    best_parameters = best_run.config

    # save best parameters as .pkl file
    with open(f'../output/transformers/best_params/{sweep_project_name}.pkl', 'wb') as f:
        pickle.dump(best_parameters, f)
     
    # --- EVALUATE MODEL ON TEST SET AND TEST GENERALIZABILITY --- #
    
    print("[INFO] Evaluating model on test data and testing model generalizability...")
        
    evaluate_transformer_test_set(sweep_project_name, 
                                  transformer_model, 
                                  train_dataloader, 
                                  eval_dataloader, 
                                  test_dataloader,
                                  gen_test_dataloader, 
                                  out_path, 
                                  out_filename_cf, 
                                  out_filename_cr,
                                  n_epochs)
    
def evaluate_transformer_test_set(sweep_project_name, transformer_model, train_dataloader, eval_dataloader, test_dataloader, gen_test_dataloader, out_path, out_filename_cf, out_filename_cr, n_epochs):
    """_summary_

    Args:
        best_parameters (_type_): _description_
        transformer_model (_type_): _description_
        train_dataloader (_type_): _description_
        eval_dataloader (_type_): _description_
        test_dataloader (_type_): _description_
        out_path (_type_): _description_
        out_filename_cf (_type_): _description_
        out_filename_cr (_type_): _description_
        n_epochs (_type_): _description_
    """
    # load best parameters as estimated by hyperparameter optimization
    best_parameters = joblib.load(f'../output/transformers/best_params/{sweep_project_name}.pkl')

    # define training arguments using best parameters
    training_args = TrainingArguments(
        output_dir='./results',
        report_to='wandb',
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=best_parameters.get("learning_rate"),
        num_train_epochs=n_epochs,              
        per_device_train_batch_size=best_parameters.get("batch_size"),  
        per_device_eval_batch_size=best_parameters.get("batch_size"),   
        warmup_steps=500,                
        weight_decay=best_parameters.get("weight_decay"),              
        logging_dir='./logs',            
        logging_steps=1,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        load_best_model_at_end=True)
    
    # add early stopping to avoid overfitting on training data
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)]

    # initialize Trainer
    trainer = Trainer(
        model=transformer_model,                         
        args=training_args,                  
        train_dataset=train_dataloader.dataset,
        eval_dataset=eval_dataloader.dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks)

    # train model
    trainer.train()

    # --- TEST ON TEST SET --- #

    # extract predictions on test data
    test_predictions = trainer.predict(test_dataloader.dataset)

    # extract true labels
    y_true = test_predictions[1]

    # extract predictions on test set
    y_preds = test_predictions[0][0]
    y_preds = y_preds.argmax(axis=-1)

    # make dataframe with predictions
    preds_id = test_dataloader.dataset["ID"]
    df_preds = pd.DataFrame({"ID": preds_id, "True label": y_true, "Prediction_internal_test": y_preds})
    df_preds.to_csv(f"../output/transformers/results/predictions/internal_{sweep_project_name}")
    df_preds.to_csv(f"../output/transformers/results/predictions/internal_{sweep_project_name}.csv")

    # create classifcation report
    cr = classification_report(y_true, y_preds)
    
    # create confusion matrix
    cf = confusion_matrix(y_true, y_preds)

    # extract evaluation metrics on test dataset
    test_metrics = trainer.evaluate(eval_dataset=test_dataloader.dataset)
    
    # save classification report and confusion matrix
    save_performance_metrics(test_metrics, cr, cf, out_filename_cf, out_filename_cr, best_parameters, out_path)

    # --- TEST GENERALIZABILITY -- #

    # extract predictions on gen test data
    gen_test_predictions = trainer.predict(gen_test_dataloader.dataset)

    # extract true labels
    gen_y_true = gen_test_predictions[1]

    # extract predictions on test set
    gen_y_preds = gen_test_predictions[0][0]
    gen_y_preds = gen_y_preds.argmax(axis=-1)

    # make dataframe with predictions
    gen_preds_id = gen_test_dataloader.dataset["ID"]
    df_gen_preds = pd.DataFrame({"ID": gen_preds_id, "True label": gen_y_true, "Prediction_external_test": gen_y_preds})
    df_gen_preds.to_csv(f"../output/transformers/results/predictions/external_{sweep_project_name}")
    df_gen_preds.to_csv(f"../output/transformers/results/predictions/external_{sweep_project_name}.csv")

    # create classifcation report
    gen_cr = classification_report(gen_y_true, gen_y_preds)
    
    # create confusion matrix
    gen_cf = confusion_matrix(gen_y_true, gen_y_preds)
    
    # save classification report and confusion matrix
    out_path_gen = os.path.join(out_path, "generalizability")

    # extract evaluation metrics on test dataset
    gen_test_metrics = trainer.evaluate(eval_dataset=gen_test_dataloader.dataset)

    save_performance_metrics(gen_test_metrics, gen_cr, gen_cf, out_filename_cf, out_filename_cr, best_parameters, out_path_gen)
    
def prepare_data(data, test_size, val_size, task_angle=None, patient_angle=None):
    """_summary_

    Returns:
        _type_: _description_
    """
    # split data into train, val and test and convert to datasetDict object
    data_dict = prepare_transformer_data(data, 
                                         test_size, 
                                         val_size, 
                                         task_angle,
                                         patient_angle)

    # load pretrained transformer model and tokenizer
    MODEL_NAME = 'chcaa/dfm-encoder-large-v1'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # define tokenizer function
    def tokenize(batch):
        return tokenizer(batch["transcript"], truncation=True, max_length=512, padding=True)

    # tokenize data
    tokenized_dataset = data_dict.map(tokenize, batched=True)

    # set format of tokenized data
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # initialize data collator object
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # make dataloader objects for train, val, and test 
    train_dataloader, eval_dataloader, test_dataloader = make_data_loader_objects(tokenized_dataset, data_collator)
    
    return train_dataloader, eval_dataloader, test_dataloader    
    
def save_performance_metrics(test_metrics, cr, cf, out_filename_cf, out_filename_cr, best_parameters, out_path):
    """
    ADD DESCRIPTION
    """
    # create output directory if it does not exist already
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # define best parameters
    lr = best_parameters.get("learning_rate")
    n_epochs = best_parameters.get("n_epochs")
    batch_size = best_parameters.get("batch_size")
    weight_decay = best_parameters.get("weight_decay")

    # save classification report
    with open(os.path.join(out_path, out_filename_cr), 'a') as file:
        file.write(f"{datetime.now()}\n\n")
        file.writelines(["Performance metrics on test data: \n",
                         f"{test_metrics} \n\n",
                         f"Hyperparameters: \n"
                         f"lr: {lr}\n",
                         f"n_epochs: {n_epochs}\n", 
                         f"batch_size: {batch_size}\n",
                         f"weight_decay: {weight_decay}\n\n",
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
# APHA CDC Data Science Demonstration Project 

This repository contains code and resources for our project "Reducing Barriers to Developing and Applying Natural Language Processing (NLP) Methods to the National Violent Death Reporting System (NVDRS). 

We aim to: 
* assess how much and what kind of training data do NLP applications to NVDRS need 
* provide code for compact LLM applications  

We use a compact LLM (distilBERT) for supervised text classification. To run models on NVDRS outcomes, provide .csvs with two columns: label and text. Label takes on a value of 1 if the NVDRS case is the positive class target outcome and 0 if not. Text may be concatenated or individual report narratives. Each .csv should consist of a train and test set. 

For example: 

```
run_model.py train.csv test.csv path/to/output/ pred.csv
```
   

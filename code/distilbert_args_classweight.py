from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import datasets
import pandas as pd 
import numpy as np
import sys
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

print(sys.argv[0]) # prints python_script.py
print(sys.argv[1]) # this is the train data path 
print(sys.argv[2]) # this is the test data path
print(sys.argv[3]) # this is the file name of the prediction file of what you are running 
print(sys.argv[4]) # this is the path of the prediction file of what you are running 

train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])

train.columns = ['text', 'label']
test.columns = ['text', 'label']

train = train[['text','label']]
test = test[['text','label']]

train['text'] = train['text'].astype(str)
test['text'] = test['text'].astype(str)

# Create dataset dictionary
train_dataset = Dataset.from_dict(train)
test_dataset = Dataset.from_dict(test)
df = datasets.DatasetDict({"train":train_dataset,"test":test_dataset})

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_temp = df.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")
roc_auc = evaluate.load("roc_auc")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision = precision_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(predictions=predictions, references=labels)
    acc = accuracy.compute(predictions=predictions, references=labels)
    roc_auc_value = roc_auc.compute(prediction_scores=predictions, references=labels)
    
    # Save predictions to CSV
    pd.DataFrame(predictions).to_csv(sys.argv[3], index=False)

    return {
        "accuracy": acc["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "roc_auc": roc_auc_value["roc_auc"]
    }



# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train['label']),
    y=train['label']
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Custom Trainer class to include class weights in the loss function
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir=sys.argv[4],
    learning_rate=5e-5,
    per_device_train_batch_size=60,
    per_device_eval_batch_size=60,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
   # metric_for_best_model="precision",  # Use precision to select the best model
   # greater_is_better=True,    
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_temp["train"],
    eval_dataset=tokenized_temp["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

print("Evaluation Results:")
print(f"Eval Loss: {eval_results['eval_loss']:.4f}")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"ROC AUC: {eval_results['eval_roc_auc']:.4f}")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall: {eval_results['eval_recall']:.4f}")

print("Training complete...")

from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch


print("Chargement du jeu de données IMDb...")
dataset = load_dataset("imdb")

print("Chargement du tokenizer BERT...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

print("Tokenisation des données...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

print("Chargement du modèle BERT pré-entraîné pour la classification...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

print("Définition des paramètres d'entraînement...")
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

print("Création du Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

print("Entraînement du modèle...")
trainer.train()

print("Évaluation du modèle...")
trainer.evaluate()

print("Test du modèle avec de nouveaux exemples...")
texts = ["I loved this movie!", "This film was awful."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
print("Prédictions :", predictions)

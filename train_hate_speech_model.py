import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json


df = pd.read_csv('labeled_data2.csv')  
# Fine-tune the pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # 3 classes: hate speech, offensive language, neither

# Split the dataset into training and testing sets
X = df['tweet']
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize and encode the text data
X_train_encodings = tokenizer(X_train.tolist(), padding=True, truncation=True, return_tensors="pt", max_length=128, return_attention_mask=True)
X_test_encodings = tokenizer(X_test.tolist(), padding=True, truncation=True, return_tensors="pt", max_length=128, return_attention_mask=True)

# Convert labels to tensors
y_train = torch.tensor(list(y_train))
y_test = torch.tensor(list(y_test))

# Fine-tune the BERT model on your hate speech dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import TrainingArguments, Trainer

train_dataset = TensorDataset(X_train_encodings.input_ids, X_train_encodings.attention_mask, y_train)
test_dataset = TensorDataset(X_test_encodings.input_ids, X_test_encodings.attention_mask, y_test)

# Define a custom data collator
class CustomDataCollator:
    def __call__(self, features):
        input_ids = torch.stack([f[0] for f in features])
        attention_mask = torch.stack([f[1] for f in features])
        labels = torch.tensor([f[2:5] for f in features])
        print("Labels:", labels)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# Create a custom data collator instance
data_collator = CustomDataCollator()

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    output_dir='./model',
    evaluation_strategy='steps',
    save_steps=500,
    num_train_epochs=2,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,  # Use the custom data collator
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: accuracy_score(p.label_ids, p.predictions.argmax(axis=1)),
)

# Rest of your training code
trainer.train()
trainer.save_model('./model')
# After fine-tuning, save the model files
model.save_pretrained('./model')

# Define your BERT model's configuration as a dictionary
config = model.config.to_dict()
'''config = {
    "vocab_size": 30522,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "initializer_range": 0.02
}
'''

# Save the configuration to a JSON file
with open('./model/config.json', 'w') as json_file:
    json.dump(config, json_file)
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('./model')  # Load your trained model

# Define a function for hate speech detection
def detect_hate_speech(comment):
    # Tokenize the input comment and convert it to a tensor
    inputs = tokenizer(comment, padding=True, truncation=True, return_tensors="pt", max_length=128)

    # Make a prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    # The output contains scores for each class; you can define a threshold for hate speech
    hate_speech_score = predictions[0][1].item()  # Assuming class 1 represents hate speech
    print(hate_speech_score)
    threshold = 1  # Adjust this threshold as needed

    # Determine if it's hate speech or not based on the threshold
    if hate_speech_score >= threshold:
        return 'Hate Speech'
    else:
        return 'Clean'


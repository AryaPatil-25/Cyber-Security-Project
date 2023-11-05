from flask import Flask, request, render_template, jsonify
from hate_speech_detection import detect_hate_speech  # Import the function from hate_speech_detection.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('./model')  # Load your trained model
model.eval()

hate_speech_comments = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    comment = request.form['comment']
    analysis_result = detect_hate_speech(comment)  # Use the imported function to detect hate speech
    if analysis_result == 'Hate Speech':
        hate_speech_comments.append(comment)
        return jsonify({'result': 'Comment has been deleted for violating hate speech guidelines.'})
    
    return render_template('result.html', result=comment)


if __name__ == '__main__':
    app.run(debug=True)

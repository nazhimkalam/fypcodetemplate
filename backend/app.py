from flask_cors import CORS
from flask_restful import Api
from flask import Flask, request, make_response
import requests
import transformers
import pandas as pd
import os
import csv
from jsonify import convert
import json
from cryptography.fernet import Fernet
import firebase_admin
from firebase_admin import firestore, credentials
from utils.model_retraining import hyperparameter_serach
from utils.types import DOMAIN_TYPES

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)
API = Api(app)

MODEL_NAME = 'bart-base_model'
TOKENIZER_NAME = 'bart-base_tokenizer'

generalized_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(GENERALIZED_MODEL_PATH)
generalized_tokenizer = transformers.AutoTokenizer.from_pretrained(GENERALIZED_TOKENIZER_PATH)


@app.route('/', methods=['GET'])
def hello_world():
    return {'message': 'Hello World'}, 200

@app.route('/api/gensum/general', methods=['POST'])
def getGeneralizedSummary():
    try:
        data = request.get_json()
        review = data['review']
        inputs = generalized_tokenizer.encode(review, return_tensors='pt', max_length=MAX_INPUT, truncation=True)
        outputs = generalized_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = generalized_tokenizer.decode(outputs[0], skip_special_tokens=True)

        sentimentAnalysisOutput = query({ "inputs": summary })
        sentiment, score = getOverallSentimentWithScore(sentimentAnalysisOutput)
        return {'summary': summary, 'sentiment': {
            'sentiment': sentiment,
            'score': score
        } }, 200
    except Exception as e:
        return {'message': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)

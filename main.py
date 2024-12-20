import torch
from scripts.infer import bert_inferencing_sentence
from flask import Flask, render_template, request, jsonify
from scripts.logging import create_logger
from scripts.config import parse_args
from waitress import serve
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cpu"
opts = parse_args()
logger, _ = create_logger(opts)
bert_sentence = bert_inferencing_sentence(device, logger)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sentiment_classify', methods=['POST'])
def upload():
    data = request.json
    
    text_data = data.get('query', '')
    if text_data:
        output, predicted, text_data = bert_sentence.process(text_data)
        results = list(predicted.values())[0] 
        response = f"This text is {results}" 

    return jsonify({"answer": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
    # serve(app, host='0.0.0.0', port=8000) works in ubuntu

from flask import Flask, render_template, request
import pandas as pd
from main import inference
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
 return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def predict():
    ecg_file = request.files['file']
    N, S, V, F, Q = 0, 0, 0, 0, 0

    N, S, V, F, Q = inference(ecg_file)
    
    return render_template('prediction.html', N=N, S=S, V=V, F=F, Q=Q)

import os
import json
import numpy as np
from keras.models import load_model
from azureml.core.model import Model


def init():
    global model
    
    try:
        model_name = "component-condition-check"
        print('Looking for model path for model: ', model_name)
        model_path = Model.get_model_path(model_name=model_name, version=1)
        print('Loading model from: ', model_path)
        model = load_model(model_path)
        print("Model loaded from disk.")
        print(model.summary())

    except Exception as e:
        print(e)
        
# note you can pass in multiple rows for scoring
def run(raw_data):
    import time
    try:
        print("Received input: ", raw_data)
        
        inputs = json.loads(raw_data)     
        inputs = np.array(inputs).reshape(-1, 100)
        results = model.predict(inputs).reshape(-1)

        inputs_dc.collect(inputs) #this call is saving our input data into Azure Blob
        prediction_dc.collect(results) #this call is saving our output data into Azure Blob

        print("Prediction created " + time.strftime("%H:%M:%S"))
        
        results = results.tolist()
        return json.dumps(results)
    except Exception as e:
        error = str(e)
        print("ERROR: " + error + " " + time.strftime("%H:%M:%S"))
        return error
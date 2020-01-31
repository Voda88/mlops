import json
import os
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


def init():
    global model
    global tokenizer

    model_dir = "/var/azureml-app/" + os.environ["AZUREML_MODEL_DIR"]

    with open(f"{model_dir}/model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    model = load_model(f"{model_dir}/model/model.h5")


def run(raw_data):

    inputs = json.loads(raw_data)
    sequences = tokenizer.texts_to_sequences(inputs["componentNotes"])
    data = pad_sequences(sequences, maxlen=100)

    results = model.predict(data)

    results = {
        "predictions": [
            "compliant" if int(m[0]) else "non-compliant" for m in results.tolist()
        ]
    }
    return results

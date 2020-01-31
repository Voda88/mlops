import score
import os
import json
import yaml

with open("conf.yaml", "r") as f:
    model_name = yaml.load(f, Loader=yaml.FullLoader)["metadata"]["model_name"]
    os.environ["AZUREML_MODEL_DIR"] = f"azureml-models/{model_name}/1"


def test_init():
    score.init()
    members = dir(score)
    assert "model" in members
    assert "tokenizer" in members


def test_run():
    sample_input = {
        "componentNotes": [
            "iron component manufactured in 1998 in good condition",
            "manufactured in 2017 made of steel in good condition"
        ]
    }

    res = score.run(json.dumps(sample_input))
    assert res == json.dumps({"predictions": ["compliant", "non-compliant"]})

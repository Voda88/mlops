import os
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

from azureml.core import Run, Model

import yaml
import pickle

datasets_folder = "./data"

embedding_dim = 100
training_samples = 90000
validation_samples = 5000
max_words = 10000


# Load component notes
car_components_df = pd.read_csv("./data/component-notes.csv")
components = car_components_df["text"].tolist()
labels = car_components_df["label"].tolist()


print("Tokenizing data...")

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(components)
sequences = tokenizer.texts_to_sequences(components)

word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))
data = pad_sequences(sequences, maxlen=embedding_dim)

labels = np.asarray(labels)
print("Shape of data tensor:", data.shape)
print("Shape of label tensor:", labels.shape)

indices = np.arange(data.shape[0])
np.random.seed(100)
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]

x_val = data[training_samples:training_samples + validation_samples]
y_val = labels[training_samples:training_samples + validation_samples]

x_test = data[training_samples + validation_samples:]
y_test = labels[training_samples + validation_samples:]

# apply the vectors provided by GloVe to create a word embedding matrix
print("Applying GloVe vectors...")

embeddings_index = {}
with open("./data/glove-vectors.txt", "r") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs


embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
print("Applying GloVe vectors completed.")

# use Keras to define the structure of the deep neural network
print("Creating model structure...")

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=embedding_dim))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# fix the weights for the first layer to those provided by the embedding matrix
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
print("Creating model structure completed.")
model.summary()


print("Training model...")
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(
    x_train, y_train, epochs=3, batch_size=32, validation_data=(x_val, y_val)
)
print("Training model completed.")

print("Saving model files...")
# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs("./outputs/model", exist_ok=True)
# save model
model.save("./outputs/model/model.h5")

with open("./outputs/model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("model saved in ./outputs/model folder")
print("Saving model files completed.")

# Magic code to register model if running inside
run = Run.get_context()
if run._run_id.startswith("OfflineRun"):
    print("This appears to be an offline run. I will not register the model")
else:
    with open("conf.yaml", "r") as f:
        metadata = yaml.load(f)["metadata"]

    Model.register(
        run.experiment.workspace,
        model_path="./outputs/model",
        model_name=metadata["model_name"],
        description=metadata["description"],
        tags=metadata["tags"],
    )

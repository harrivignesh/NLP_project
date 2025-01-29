import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, SimpleRNN, Bidirectional, LSTM, GlobalMaxPooling1D, Dense, Dropout, Layer, Multiply
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
df = pd.read_csv('train.csv')
tweets = df['text'].tolist()
labels = df['target'].tolist()
np_labels = np.array(labels)
ten_labels = np.asarray(labels).astype('float32').reshape((-1,1))

trans_labels = []
for label in labels:
  if label == 1:
    trans_labels.append('disaster')
  else:
    trans_labels.append('not disaster')

df_new = df.drop(['id', 'keyword', 'location'], axis=1)
print(df.head())
print(df.shape)
print(df.info)
df['target'].value_counts().plot(kind='pie', figsize=(6,6), title="Target Count")
df['keyword'].value_counts().plot(kind='bar', figsize=(16,9), xlabel="Keywords", title="Keywords in Dataset")
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(tweets)
sequences = tokenizer.texts_to_sequences(tweets)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=50)
# Building the model
model = Sequential([
    Embedding(10000, 128),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(data, np_labels, epochs=10, batch_size=32)

# Evaluating the model
test_loss, test_acc = model.evaluate(data, np_labels)
print('\nCNN Test accuracy:', test_acc)
# Building the model
model = Sequential([
      Embedding(10000, 128),
      SimpleRNN(64, return_sequences=True),
      SimpleRNN(64),
      Dense(128, activation="relu"),
      Dropout(0.2),
      Dense(1, activation="sigmoid")
])

model.compile("rmsprop", "binary_crossentropy", metrics=["accuracy"])

# Training the model
model.fit(data, np_labels, epochs=10)

# # Evaluating the model
test_loss, test_acc = model.evaluate(data, np_labels)
print('\nRNN Test accuracy:', test_acc)
# Building the model
model = Sequential([
    Embedding(10000, 128),
    Bidirectional(LSTM(128)),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

# Training the model
model.fit(data, np_labels, epochs=10)

# Evaluating the model
test_loss, test_acc = model.evaluate(data, np_labels)
print('\nBi-RNN Test accuracy:', test_acc)
tokenizer = AutoTokenizer.from_pretrained("JiaJiaCen/xlm-roberta-base-language-detection-disaster-twitter")
model = AutoModelForSequenceClassification.from_pretrained("JiaJiaCen/xlm-roberta-base-language-detection-disaster-twitter")
def calculate_accuracy(predicted_labels, true_labels):
    correct_predictions = 0
    for i in range(len(predicted_labels)):
      if predicted_labels[i] == true_labels[i]:
        correct_predictions += 1

    total_samples = len(true_labels)
    accuracy = (correct_predictions / total_samples) * 100
    return accuracy
n = 3000
predicted_labels = []
true_labels = trans_labels[:n]

for tweet in tweets[:n]:
  inputs = tokenizer(tweet, return_tensors="pt")

  with torch.no_grad():
    logits = model(**inputs).logits

  predicted_class_id = logits.argmax().item()
  predicted_labels.append(model.config.id2label[predicted_class_id])

accuracy = calculate_accuracy(predicted_labels, true_labels)
print('\nRoBERTa Test accuracy:', accuracy)
class GLU(Layer):
    def __init__(self, units, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.linear = Dense(self.units)
        self.sigmoid = Dense(self.units, activation='sigmoid')

    def call(self, inputs):
        linear_output = self.linear(inputs)
        gated_output = self.sigmoid(inputs)
        return Multiply()([linear_output, gated_output])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
        })
        return config
# Building the model
model = Sequential([
    Embedding(10000, 128),
    GLU(128),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

# Training the model
model.fit(data, ten_labels, epochs=10, batch_size=32)

# Evaluating the model
test_loss, test_acc = model.evaluate(data, ten_labels)
print('\nGLU Test accuracy:', test_acc)
def plot_model_accuracy(model_accuracies):
    models = list(model_accuracies.keys())
    accuracies = list(model_accuracies.values())
    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Different Models')
    plt.ylim(0, 1)
    plt.show()


model_accuracies = {
    "CNN": 0.9860764741897583,
    "RNN": 0.9851569533348083,
    "Bi-RNN": 0.982398509979248,
    "GLU": 0.6147273778915405,
    "RoBERTa": 0.8616666666666667
}

plot_model_accuracy(model_accuracies)

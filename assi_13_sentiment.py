import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv(r'C:\Users\samik\Downloads\sentiment_analysis.csv')

# Preprocessing
data['text'] = data['text'].str.lower()

# Encode sentiment labels
label_encoder = LabelEncoder()
data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

# Tokenize text
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])

# Pad sequences
max_length = 50
X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
y = np.array(data['sentiment'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM Model
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=max_length),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Predict sentiment for a new post
sample_post = "I feel amazing today, life is beautiful!"
sample_sequence = tokenizer.texts_to_sequences([sample_post.lower()])
sample_padded = pad_sequences(sample_sequence, maxlen=max_length, padding='post')
sample_predict=model.predict(sample_padded)
predicted_class = label_encoder.inverse_transform([np.argmax(sample_predict)])[0]
print(f"Predicted Sentiment: {predicted_class}")

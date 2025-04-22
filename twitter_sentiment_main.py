# -*- coding: utf-8 -*-
"""
Twitter Sentiment Analysis with Permanent GloVe Setup
"""

# ====================== IMPORTS ======================
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPool1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import Constant
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ====================== GLOVE SETUP ======================
# Set your permanent GloVe directory here (update this path)
GLOVE_DIR = r"C:\Users\DELL\Desktop\bia.DEC24\15_model_deployment_2\glove.6B"
GLOVE_FILE = "glove.6B.100d.txt"

def load_glove():
    """Load pre-downloaded GloVe embeddings"""
    glove_path = os.path.join(GLOVE_DIR, GLOVE_FILE)
    
    if not os.path.exists(glove_path):
        raise FileNotFoundError(
            f"\nGloVe file not found at {glove_path}\n"
            "Please:\n"
            "1. Download from: https://nlp.stanford.edu/data/glove.6B.zip\n"
            "2. Extract ALL files to: C:\\ML_Resources\\glove_embeddings\n"
            "3. Ensure this file exists: glove.6B.100d.txt"
        )
    
    print(f"\nUsing pre-downloaded GloVe embeddings from:\n{glove_path}")
    return glove_path

# ====================== DATA PREP ======================
def clean_text(text):
    """Clean text with proper regex escape"""
    return text.str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()

def load_data():
    """Load and preprocess data"""
    data = pd.read_csv('Twitter_Data.csv').dropna()
    data['text'] = clean_text(data['text'].astype(str))
    return data

# ====================== MODEL BUILDING ======================
def create_embedding_matrix(tokenizer, glove_path):
    """Create embedding matrix from GloVe file"""
    embeddings_index = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    embedding_dim = 100
    vocab_size = min(20000, len(tokenizer.word_index) + 1)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, i in tokenizer.word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix, vocab_size, embedding_dim

def build_model(vocab_size, embedding_dim, embedding_matrix, max_length=128):
    """Build the LSTM model"""
    model = tf.keras.Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  embeddings_initializer=Constant(embedding_matrix),
                  input_length=max_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPool1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=['accuracy']
    )
    return model

# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    # 0. Suppress TensorFlow oneDNN warnings (optional)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # 1. Load GloVe
    glove_path = load_glove()
    
    # 2. Load and prepare data
    data = load_data()
    sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    data['sentiment'] = data['sentiment'].map(sentiment_mapping)
    
    # 3. Split data (70% train, 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        data['text'], data['sentiment'],
        test_size=0.3, stratify=data['sentiment'], random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5, stratify=y_temp, random_state=42
    )
    
    # 4. Tokenization
    tokenizer = Tokenizer(num_words=20000, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    # 5. Convert to sequences
    max_length = 128
    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), 
                              maxlen=max_length, padding='post')
    X_val_pad = pad_sequences(tokenizer.texts_to_sequences(X_val),
                            maxlen=max_length, padding='post')
    X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test),
                             maxlen=max_length, padding='post')
    
    # 6. Create embedding matrix
    embedding_matrix, vocab_size, embedding_dim = create_embedding_matrix(tokenizer, glove_path)
    
    # 7. Handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    
    # 8. Build and train model
    model = build_model(vocab_size, embedding_dim, embedding_matrix)
    
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
    
    print("\nTraining model...")
    history = model.fit(
        X_train_pad, y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_val_pad, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # 9. Evaluate
    test_loss, test_acc = model.evaluate(X_test_pad, y_test)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    
    y_pred = np.argmax(model.predict(X_test_pad), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=sentiment_mapping.keys()))
    
    # 10. Save artifacts
    model.save('twitter_sentiment_model.h5')
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # 11. Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import os

st.set_page_config(page_title='ANN Dashboard', layout='wide')

st.title('ANN Classification Dashboard')

# Backend File Upload and Processing
def load_and_process_data(filepath):
    data = pd.read_csv(filepath)
    X = data.drop('target', axis=1)
    y = data['target']

    # Encoding categorical variables
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X.select_dtypes(include=['object', 'category'])).toarray()
    X_numeric = X.select_dtypes(exclude=['object', 'category']).values
    X_processed = np.hstack((X_encoded, X_numeric))

    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_processed)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

# Upload file in the backend
uploaded_file = 'https://raw.githubusercontent.com/Harshit-Badgamia/ANN_dashboard/refs/heads/main/clean_train.csv'

X_train, X_val, y_train, y_val = load_and_process_data(uploaded_file)

# Hyperparameter tuning
st.sidebar.header('Hyperparameter Tuning')
learning_rate = st.sidebar.slider('Learning Rate', 0.0001, 0.01, 0.001, step=0.0001)
batch_size = st.sidebar.selectbox('Batch Size', [32, 64, 128, 256, 512])
epochs = st.sidebar.slider('Epochs', 10, 100, 50)
num_layers = st.sidebar.slider('Number of Hidden Layers', 1, 10, 3)
neurons_per_layer = [st.sidebar.selectbox(f'Neurons in Layer {i+1}', [2**j for j in range(4, 10)]) for i in range(num_layers)]
dropout_rate = st.sidebar.slider('Dropout Rate', 0.0, 0.5, 0.2, step=0.05)

# Custom Model Building
def build_custom_model():
    model = Sequential()
    model.add(Dense(neurons_per_layer[0], activation='relu', input_shape=(X_train.shape[1],)))
    for i in range(1, num_layers):
        model.add(Dense(neurons_per_layer[i], activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))  # Match original output layer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Display model summary
if st.sidebar.button('Show Model Summary'):
    model = build_custom_model()
    model.summary(print_fn=lambda x: st.text(x))

# Accuracy and Loss Plot
def plot_metrics(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot(ax=ax[0])
    pd.DataFrame(history.history)[['loss', 'val_loss']].plot(ax=ax[1])
    ax[0].set_title('Accuracy')
    ax[1].set_title('Loss')
    st.pyplot(fig)

# Confusion Matrix Visualization
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(fig)

# Model training and evaluation
if st.button('Train Model') and uploaded_file:
    with st.spinner('Training...'):
        model = build_custom_model()
        history = model.fit(
            x=X_train, y=y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size, epochs=epochs, verbose=0)
        st.success('Model trained successfully!')
        plot_metrics(history)

        # Predictions and Confusion Matrices
        y_train_pred = np.argmax(model.predict(X_train), axis=1)
        y_val_pred = np.argmax(model.predict(X_val), axis=1)
        plot_confusion_matrix(y_train, y_train_pred, 'Training Set Confusion Matrix')
        plot_confusion_matrix(y_val, y_val_pred, 'Validation Set Confusion Matrix')

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
from sklearn.metrics import confusion_matrix, classification_report
import os

st.set_page_config(page_title='ANN Dashboard', layout='wide')

st.title('ANN Classification Dashboard')

# Backend File Upload and Processing
def load_and_process_data(hb11_filepath):
    hb11_data = pd.read_csv(hb11_filepath)
    hb11_X = hb11_data.drop('Credit_Score', axis=1)
    hb11_y = hb11_data['Credit_Score'].astype(str)
    hb11_label_encoder = OrdinalEncoder()
    hb11_y = hb11_label_encoder.fit_transform(hb11_y.values.reshape(-1, 1)).flatten()
    global hb11_label_mapping
    hb11_label_mapping = {hb11_i: hb11_label for hb11_i, hb11_label in enumerate(hb11_label_encoder.categories_[0])}

    # Encoding categorical variables
    hb11_encoder = OneHotEncoder()
    hb11_X_encoded = hb11_encoder.fit_transform(hb11_X.select_dtypes(include=['object', 'category'])).toarray()
    hb11_X_numeric = hb11_X.select_dtypes(exclude=['object', 'category']).values
    hb11_X_processed = np.hstack((hb11_X_encoded, hb11_X_numeric))

    # Scaling
    hb11_scaler = MinMaxScaler()
    hb11_X_scaled = hb11_scaler.fit_transform(hb11_X_processed)

    # Train-test split
    hb11_X_train, hb11_X_val, hb11_y_train, hb11_y_val = train_test_split(hb11_X_scaled, hb11_y, test_size=0.2, random_state=42)
    return hb11_X_train, hb11_X_val, hb11_y_train, hb11_y_val

# Upload file in the backend
hb11_uploaded_file = 'https://raw.githubusercontent.com/Harshit-Badgamia/ANN_dashboard/refs/heads/main/clean_train.csv'

hb11_X_train, hb11_X_val, hb11_y_train, hb11_y_val = load_and_process_data(hb11_uploaded_file)

# Hyperparameter tuning
st.sidebar.header('Hyperparameter Tuning')
hb11_learning_rate = st.sidebar.slider('Learning Rate', 0.0001, 0.01, 0.001, step=0.0001, format='%.4f')
hb11_batch_size = st.sidebar.selectbox('Batch Size', [32, 64, 128, 256, 512])
hb11_epochs = st.sidebar.selectbox('Epochs', [hb11_i * 10 for hb11_i in range(1, 11)])
hb11_num_layers = st.sidebar.slider('Number of Hidden Layers', 1, 10, 3)
hb11_neurons_per_layer = [st.sidebar.selectbox(f'Neurons in Layer {hb11_i+1}', [2**hb11_j for hb11_j in range(4, 10)]) for hb11_i in range(hb11_num_layers)]
hb11_dropout_rate = st.sidebar.slider('Dropout Rate', 0.0, 0.5, 0.2, step=0.05)

# Custom Model Building
def build_custom_model():
    hb11_model = Sequential()
    hb11_model.add(Dense(hb11_neurons_per_layer[0], activation='relu', input_shape=(hb11_X_train.shape[1],)))
    for hb11_i in range(1, hb11_num_layers):
        hb11_model.add(Dense(hb11_neurons_per_layer[hb11_i], activation='relu'))
        hb11_model.add(Dropout(hb11_dropout_rate))
    hb11_model.add(Dense(3, activation='softmax'))
    hb11_optimizer = tf.keras.optimizers.Adam(learning_rate=hb11_learning_rate)
    hb11_model.compile(optimizer=hb11_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return hb11_model

# Display model summary
if st.sidebar.button('Show Model Summary'):
    hb11_model = build_custom_model()
    hb11_model.summary(print_fn=lambda x: st.text(x))

# Accuracy and Loss Plot
def plot_metrics(hb11_history):
    hb11_fig, hb11_ax = plt.subplots(1, 2, figsize=(12, 5))
    pd.DataFrame(hb11_history.history)[['accuracy', 'val_accuracy']].plot(ax=hb11_ax[0])
    pd.DataFrame(hb11_history.history)[['loss', 'val_loss']].plot(ax=hb11_ax[1])
    hb11_ax[0].set_title('Accuracy')
    hb11_ax[1].set_title('Loss')
    st.pyplot(hb11_fig)

# Confusion Matrix Plot
def plot_confusion_matrix(hb11_y_true, hb11_y_pred, hb11_title):
    hb11_cm = confusion_matrix(hb11_y_true, hb11_y_pred)
    hb11_fig, hb11_ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(hb11_cm, annot=True, fmt='d', cmap='Blues', ax=hb11_ax, xticklabels=list(hb11_label_mapping.values()), yticklabels=list(hb11_label_mapping.values()))
    plt.title(hb11_title)
    st.pyplot(hb11_fig)

# Precision, Recall, and F1-Score Plot
def plot_classification_report(hb11_y_true, hb11_y_pred):
    hb11_report = classification_report(hb11_y_true, hb11_y_pred, output_dict=True)
    hb11_df_report = pd.DataFrame(hb11_report).transpose().iloc[:-3, :3]
    hb11_fig, hb11_ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=hb11_df_report, x=hb11_df_report.index, y='f1-score', color='skyblue', label='F1-Score')
    sns.barplot(data=hb11_df_report, x=hb11_df_report.index, y='precision', color='lightgreen', label='Precision')
    sns.barplot(data=hb11_df_report, x=hb11_df_report.index, y='recall', color='salmon', label='Recall')
    plt.title('Precision, Recall, and F1-Score')
    plt.legend()
    st.pyplot(hb11_fig)

# Class Distribution Plot
def plot_class_distribution(hb11_y):
    hb11_fig, hb11_ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x=hb11_y)
    plt.title('Class Distribution')
    st.pyplot(hb11_fig)

# Model training and evaluation
if st.button('Train Model') and hb11_uploaded_file:
    with st.spinner('Training...'):
        hb11_model = build_custom_model()
        hb11_history = hb11_model.fit(
            x=hb11_X_train, y=hb11_y_train,
            validation_data=(hb11_X_val, hb11_y_val),
            batch_size=hb11_batch_size, epochs=hb11_epochs, verbose=0)
        st.success('Model trained successfully!')
        plot_metrics(hb11_history)

        # Predictions and Confusion Matrices
        hb11_y_train_pred = np.argmax(hb11_model.predict(hb11_X_train), axis=1)
        hb11_y_val_pred = np.argmax(hb11_model.predict(hb11_X_val), axis=1)

        hb11_col1, hb11_col2 = st.columns(2)
        with hb11_col1:
            plot_confusion_matrix(hb11_y_train, hb11_y_train_pred, 'Training Set Confusion Matrix')
        with hb11_col2:
            plot_confusion_matrix(hb11_y_val, hb11_y_val_pred, 'Validation Set Confusion Matrix')

        plot_classification_report(hb11_y_val, hb11_y_val_pred)
        plot_class_distribution(hb11_y_train)

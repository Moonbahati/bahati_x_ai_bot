import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Attention, LayerNormalization, Concatenate
)
from tensorflow.keras.optimizers import Adam
import logging

# Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegendaryUltraLSTM")

class LegendaryLSTM:
    def __init__(self, input_dim=1, sequence_length=10, lstm_units=64,
                 dropout_rate=0.2, learning_rate=0.001):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self._build_model()

    def _build_model(self):
        inputs = Input(shape=(self.sequence_length, self.input_dim))
        x = LSTM(self.lstm_units, return_sequences=True)(inputs)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        # Attention mechanism
        attention = Attention()([x, x])
        x = Concatenate()([x, attention])
        x = LayerNormalization()(x)
        x = LSTM(self.lstm_units, return_sequences=False)(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs, outputs)
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )
        logger.info("🔥 Hybrid Legendary LSTM model constructed.")

    def train(self, X_train, y_train, epochs=10, batch_size=32, validation_data=None):
        logger.info(f"🚀 Training started: {epochs} epochs")
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        logger.info("✅ Training complete.")
        return history

    def predict(self, X):
        logger.debug(f"🔍 Predicting on input of shape {X.shape}")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        logger.info("📊 Evaluating model performance")
        return self.model.evaluate(X_test, y_test, verbose=0)

    def summary(self):
        self.model.summary()

    def save(self, filepath="legendary_lstm_model.h5"):
        self.model.save(filepath)
        logger.info(f"💾 Model saved to {filepath}")

    def load(self, filepath="legendary_lstm_model.h5"):
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"📥 Model loaded from {filepath}")

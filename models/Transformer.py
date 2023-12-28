

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow_addons.layers import MultiHeadAttention

class Transformer:
    
    def __init__(self):
        self.model = None
        self.history = None
        
    
    def build(self, input_dim,  embedding_dim, max_length ,num_heads= 16, num_classes = 3, dropout_rate = 0.4):
        inputs = Input(shape=(max_length,))

        # Token embedding layer
        embedding = Embedding(input_dim, embedding_dim, input_length=max_length)(inputs)

        # Positional encoding
        pos_encoding = self.positional_encoding(max_length, embedding_dim)
        embedded = embedding + pos_encoding

        # Multi-head self-attention
        attention = MultiHeadAttention(num_heads=num_heads, key_dim=num_heads)(embedded, embedded, embedded)
        attention = Dropout(dropout_rate)(attention)
        attention = LayerNormalization(epsilon=1e-6)(embedded + attention)

        # Feed-forward layer
        feed_forward = Dense(embedding_dim, activation='relu')(attention)
        feed_forward = Dropout(dropout_rate)(feed_forward)
        x = LayerNormalization(epsilon=1e-6)(attention + feed_forward)

        # Global average pooling
        x = GlobalAveragePooling1D()(x)

        # Output layer
        outputs = Dense(num_classes, activation='softmax')(x)

        # Model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Configure model
        optimizer = Adam(learning_rate=0.0001)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        


    
    # Fit the model and return the history of learning
    def train(self, X_train, y_train, X_valid, y_valid, epochs=20, batch_size=100, early_stopping =True):
        if(early_stopping):
            early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
            
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_valid,y_valid), callbacks= early_stopping)
        return self.history
    
    
    # Generate predicted labels
    def predict(self, X_test): 
        
        # predicte Probabilities of the 3 classes       
        predicted_labels = self.model.predict(X_test)
        
        # Extracting the index of the most probable class and adjusting to a convention of [-1, 0, 1].
        predicted_labels = np.argmax(predicted_labels, axis=1) - 1
        return predicted_labels


    def positional_encoding(self, max_length, d_model):
        position = tf.range(start=0, limit=max_length, delta=1, dtype=tf.float32)  # Ensure float32 dtype
        angle_rates = 1 / np.power(10000, (2 * np.arange(d_model) // 2) / np.float32(d_model))
        angle_rads = tf.reshape(position, (-1, 1)) * angle_rates

        # Apply sin to even indices in the array; 2i
        angle_rads = tf.where(tf.math.equal(tf.math.floormod(tf.range(d_model), 2), 0),
                              tf.sin(angle_rads),
                              angle_rads)

        # Apply cos to odd indices in the array; 2i+1
        angle_rads = tf.where(tf.math.equal(tf.math.floormod(tf.range(d_model), 2), 1),
                              tf.cos(angle_rads),
                              angle_rads)

        pos_encoding = tf.expand_dims(angle_rads, axis=0)
        return tf.cast(pos_encoding, dtype=tf.float32)
    
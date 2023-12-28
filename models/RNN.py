

from keras.models import Sequential
from keras.layers import Embedding, Dropout, SimpleRNN, Bidirectional, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np

class SRNN:
    def __init__ (self):
        self.model = Sequential()
        self.history = None
        
    # Model arcitecture    
    def build(self,vocab_size, embedding_dim, max_length, lstm_units=16):
        
        # Embedding layer
        self.model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
        self.model.add(Dropout(0.3))

        # Simple RNN layers
        self.model.add(SimpleRNN(units=64, activation='relu',return_sequences = True)) 
        self.model.add(Bidirectional(SimpleRNN(units=64, return_sequences = True)))
        self.model.add(Bidirectional(SimpleRNN(units=32)))

        self.model.add(Flatten())

        # Additional Dense layer
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dropout(0.3))

        # Output layer
        self.model.add(Dense(units=3, activation='softmax'))
        
        # Configure model

        optimizer = Adam(learning_rate=0.0001)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return self.model
    
    # Fit the model and return the history of learning
    def train(self, X_train, y_train, X_valid, y_valid, epochs=20, batch_size=100, early_stopping =True):
        
        # Applying early stopping to prevent overfitting 
        if(early_stopping):
            early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
            
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, use_multiprocessing=True,
                            validation_data=(X_valid,y_valid), callbacks= early_stopping)
        return self.history
   
  
   # Generate predicted labels
    def predict(self, X_test): 
        
        # Probabilities of the 3 classes       
        predicted_labels = self.model.predict(X_test)
        
        # Extracting the index of the most probable class and adjusting to a convention of [-1, 0, 1].
        predicted_labels = np.argmax(predicted_labels, axis=1) - 1
        return predicted_labels

        


import pickle
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import emoji
import numpy as np

class Preprocessor:
    
    def __init__(self):
        self.max_vocab_size = None
        self.max_sequence_length = None
        self.tokenizer = Tokenizer()
        self.label_encoder = LabelEncoder()
        
        
    def calculate_max_lengths(self, texts):
        self.max_vocab_size = len(set(' '.join(texts).split())) + 1
        self.max_sequence_length = max(len(text.split()) for text in texts)
        
        
    def tokenize(self, texts):
        # vocalbolary declaration
        self.tokenizer.fit_on_texts(texts)

    def encode(self, texts):
        # tokenization and indexing
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding=self.padding)
        return padded_sequences
         
    def remove_dublicate_emoji_(self,input_str):

        text = ''.join([c for c in input_str if not emoji.emoji_count(c) > 0])
        emojis = [c for c in input_str if emoji.emoji_count(c) > 0]

        unique_emojis = set(emojis)

        result_str = text + ''.join(unique_emojis)

        return result_str
    
    def remove_dublicate_emoji(self,text):
        
        v_remove_dublicate_emoji = np.vectorize(self.remove_dublicate_emoji_)
        return v_remove_dublicate_emoji(text)
        

    def labels_decoding(self,encoded_labels):
         return self.label_encoder.inverse_transform(self.label_encoder)
    
    


    def remove_stopwords(self, text):

        text = text.tolist()

        stop_words = set(stopwords.words('arabic'))

        filtered_texts = [' '.join([word for word in word_tokenize(sentence) if word not in stop_words]) for sentence in text]

        return filtered_texts

    
    def labels_encoding(self,labels, f = 0):

            if (f == 1):
                return self.label_encoder.fit_transform(labels)
            else:
                return self.label_encoder.transform(labels)


    def preprocess_text(self, text):

#         filtered_text = self.remove_stopwords(text)
        unique_emojis = self.remove_dublicate_emoji(text)

        return unique_emojis
   


   
    
    def train_preprocess(self, features,labels, padding = 'post'):

        self.padding = padding
        
        # stop words and emojis
        cleaned_text =self.preprocess_text(features)
        
        # encode sequences 
        self.calculate_max_lengths(cleaned_text)
        self.tokenize(cleaned_text)
        
        new_features = self.encode(cleaned_text)
        
        # encode labels
        labels_encoded = self.labels_encoding(labels, 1)
    
        return new_features, labels_encoded
    

    def test_preprocess(self, features, labels = None):

        # stop words and emojis
        cleaned_text =self.preprocess_text(features)

        # encode sequences
        new_features = self.encode(cleaned_text)
        
        if labels is None:
            return new_features
        
        # encode label
        labels_encoded = self.labels_encoding(labels)
        
        return new_features, labels_encoded
    
def plot_history(model_history):
        
    # Plot training & validation accuracy values
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def save_model_pkl(model, path = 'Model.pkl'):
    weights_path = 'Model.pkl'
    with open(weights_path, 'wb') as f:
        pickle.dump(model.get_weights(), f)
        
def load_model_pkl(model, path='Model.pkl'):
    with open(path, 'rb') as f:
        model.set_weights(pickle.load(f))
    return model

def oversampling(X, y, desired_count = 14000, label = 0):
    dataframe = pd.concat([X, pd.DataFrame({'rating': y})], axis=1)

    # Check distribution
    print("--------------------------")

    print("Before:")
    print(dataframe['rating'].value_counts())

    # Minority class
    minority_class = dataframe[dataframe['rating'] == label]

    # Oversample 
    oversampled_minority_class = minority_class.sample(n=desired_count, replace=True, random_state=42)

    # Concatenate the new samples with the minority class
    df_resampled = pd.concat([dataframe, oversampled_minority_class], ignore_index=True)

    # Shuffle the data
    df_resampled = shuffle(df_resampled, random_state=42).reset_index(drop=True)

    # Check distribution
    print('\nAfter:')
    print(df_resampled['rating'].value_counts())

    # Extract X_train & y_train
    X_oversampled = df_resampled.drop('rating', axis=1)
    y_oversampled = df_resampled['rating'].values
    
    return X_oversampled, y_oversampled

from models.LSTM import LSTModel
from models.RNN import SRNN
from models.Transformer import Transformer
from helper import *


# ======== Data preparation =========

# Reading data
df = pd.read_excel('/kaggle/input/review-data/data.xlsx')

# Extract features and labels
d_x = np.array(df['review_description'])
d_y = np.array(df['rating'])

X = d_x
y = d_y

y_test = X_test = None


# Split data into (train, validation, and test)

"""

Training Set (X_train, y_train): 90%
Validation Set (X_valid, y_valid): 5%
Test Set (X_test, y_test): 5%

"""
vt = 0.05
X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=vt*2, random_state=42)
X_test_, X_valid_, y_test_, y_valid_ = train_test_split(X_test_, y_test_, test_size= 0.5, random_state=38)
print("--------------------------")
print("X_train: ", X_train_.shape)
print("X_valid: ", X_valid_.shape)
print("X_test: ", X_test_.shape)



# Preprocessing
pre = Preprocessor()

# Apply preprocessing separately
# Note: y_train & y_valid encoded to [0, 1, 2]
X_train, y_train = pre.train_preprocess(X_train_, y_train_)
X_test = pre.test_preprocess(X_test_)
X_valid, y_valid = pre.test_preprocess(X_valid_, y_valid_)
y_test = y_test_ # [-1, 0, 1]

# Embedding layer paramters
vocab_size = len(pre.tokenizer.word_index) + 1
embedding_dim = 100
max_length = len(X_train[0])



# ======== Modeling =========

# Choose model : SRNN, Transformer, LSTModel
model = LSTModel()
model.build(vocab_size, embedding_dim, max_length)
model.train(X_train, y_train, X_valid, y_valid, early_stopping=True)

# Visulization
# plot_history(model.history)


# ======== Evaluation =========

predictes = model.predict(X_test)
accuracy = np.mean(predictes == y_test)

print(f"Accuracy on Test Data: {accuracy * 100:.2f}%")



# ======== Save model =========
save_model_pkl(model.model)



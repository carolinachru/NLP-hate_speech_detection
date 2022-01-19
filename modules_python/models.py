# Import packages
import tensorflow as tf
from tensorflow.keras import Sequential

# Architecture
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional

# Metrics for performance
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall
from tensorflow.keras.metrics import AUC

####################################### LSTM
############################################

def model_LSTM(units_layer1, units_layer2, vocab_len, embed_len, max_len_tok, embed_mat, n_dropout = 0, bi = False):
  model_lstm = Sequential()
  model_lstm.add(Embedding(input_dim = vocab_len, # Vocabulary size
                          output_dim = embed_len, # Dimension of dense embed
                          input_length = max_len_tok,
                          weights = [embed_mat],
                          mask_zero = True,
                            # This mask step to tell Model that we have padding so as to ignore this
                            # https://www.tensorflow.org/guide/keras/masking_and_padding
                          trainable = False))
  # First LSTM
  if bi == False:
    model_lstm.add(LSTM(units_layer1, return_sequences=True))
  else:
    model_lstm.add(Bidirectional(LSTM(units_layer1, return_sequences=True)))

  # Dropout
  if n_dropout!=0:
    model_lstm.add(Dropout(n_dropout))

  # Second LSTM
  if bi == False:
    model_lstm.add(LSTM(units_layer2))
  else:
    model_lstm.add(Bidirectional(LSTM(units_layer2)))

  # Dropout
  if n_dropout!=0:
    model_lstm.add(Dropout(n_dropout))

  model_lstm.add(Dense(1, activation = 'sigmoid'))

  model_lstm.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy', Precision(), Recall(), AUC()])

  print(model_lstm.summary())

  return model_lstm

######################################## GRU
############################################

def model_GRU(units_layer1, units_layer2, vocab_len, embed_len, max_len_tok, embed_mat, n_dropout = 0, bi = False):
  model_gru = Sequential()
  model_gru.add(Embedding(input_dim = vocab_len, # Vocabulary size
                          output_dim = embed_len, # Dimension of dense embed
                          input_length = max_len_tok,
                          weights = [embed_mat],
                          mask_zero = True, 
                            # This mask step to tell Model that we have padding so as to ignore this
                            # https://www.tensorflow.org/guide/keras/masking_and_padding
                          trainable = False))
  # First GRU Layer
  if bi == False:
    model_gru.add(GRU(units_layer1, return_sequences=True))
  else:
    model_gru.add(Bidirectional(GRU(units_layer1, return_sequences=True)))

  # Dropout
  if n_dropout!=0:
    model_gru.add(Dropout(n_dropout))

  # Second GRU Layer
  if bi == False:
    model_gru.add(GRU(units_layer2))
  else:
    model_gru.add(Bidirectional(GRU(units_layer2)))

  # Dropout
  if n_dropout!=0:
    model_gru.add(Dropout(n_dropout))

  model_gru.add(Dense(1, activation = 'sigmoid'))

  model_gru.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy', Precision(), Recall(), AUC()])

  print(model_gru.summary())

  return model_gru

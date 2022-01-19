from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def text_conversion(x_train, x_val, x_test):
    # Tokenizer object
    tokens = Tokenizer()
    tokens.fit_on_texts(x_train)
    print('Total input of documents for tokenizer:', tokens.document_count)

    # Tokens to number sequences
    text_encoded_train = tokens.texts_to_sequences(x_train)

    # Padding to a determined length
    max_length_tokens = 70 # Maximum characters in tweets is 280, assumption: 70 words (of 4 characters)
    text_padded_train = pad_sequences(text_encoded_train,
                                maxlen = max_length_tokens,
                                padding = 'post') # Padding after sequence

    # Validation
    # Tokens to number sequences
    text_encoded_val = tokens.texts_to_sequences(x_val)

    # Padding to a determined length
    text_padded_val = pad_sequences(text_encoded_val,
                                maxlen = max_length_tokens,
                                padding = 'post') # Padding after sequence

    # Test
    # Tokens to number sequences
    text_encoded_test = tokens.texts_to_sequences(x_test)

    # Padding to a determined length
    text_padded_test = pad_sequences(text_encoded_test,
                                maxlen = max_length_tokens,
                                padding = 'post') # Padding after sequence

    return tokens, text_padded_train, text_padded_val, text_padded_test

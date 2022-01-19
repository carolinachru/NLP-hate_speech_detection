
import matplotlib.pyplot as plt


def plot_lstm(history_lstm0, history_lstm_class0, history_lstmB0, history_lstmB_class0):
    plt.plot(history_lstm0.history['val_recall'], history_lstm0.history['val_precision'], '-o', label = 'LSTM - base')
    plt.plot(history_lstm_class0.history['val_recall_1'], history_lstm_class0.history['val_precision_1'], '-o', label = 'LSTM - class')
    plt.plot(history_lstmB0.history['val_recall_2'], history_lstmB0.history['val_precision_2'], '-o', label = 'LSTM - bi')
    plt.plot(history_lstmB_class0.history['val_recall_3'], history_lstmB_class0.history['val_precision_3'], '-o', label = 'LSTM - class & bi')
    plt.legend()
    plt.title('LSTM models, validation set, \nprecision and recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

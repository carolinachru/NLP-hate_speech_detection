import pandas as pd
import numpy as np
import json

def load_MMHS150K():
    with open('/content/gdrive/My Drive/ST449_final/MMHS150K_GT.json') as f:
      data = json.load(f)
    return data


# Training
id_train0 = open('/content/gdrive/My Drive/ST449_final/train_ids.txt').read()
id_train = id_train0.split()
print(id_train[0])
print('Number of training tweets:',len(id_train),
      'Type of data format:', type(id_train))

# Validation set
id_val0 = open('/content/gdrive/My Drive/ST449_final/val_ids.txt').read()
id_val = id_val0.split()
print(id_val[0])
print('Number of validation tweets:',len(id_val))

# Test
id_test0 = open('/content/gdrive/My Drive/ST449_final/test_ids.txt').read()
id_test = id_test0.split()
print(id_test[0])
print('Number of test tweets:',len(id_test))

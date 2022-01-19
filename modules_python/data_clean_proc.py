import re # Regex package

# Function for cleaning tweet text
def clean_symbols(astring):
  '''Cleans tweet text by homogenizing mentions (@user), hashtags (#hashtag),
  URL (https/), removing double or more spaces and transforms to lower case.'''
  return ' '.join(re.sub('\w+:\/\/\S+', 'https/',
                re.sub('#[A-Za-z0-9]+', '#hashtag',
                       re.sub('@[A-Za-z0-9]+', '@user', astring))).lower().split())


# Creating data and label dataset
def for_text_label(adict, subset = False):
  '''Extracts from dictionary adict, two lists:
  text and labels. Subset option is for managing only 10
  observations.'''
  count = 0
  text = []
  labels3 = []

  for i, j in adict.items():
    temp_text = clean_symbols(adict[i]['tweet_text'])
    text.append(temp_text)

    temp_labels = adict[i]['labels']
    labels3.append(temp_labels)

    if subset == True:
      count += 1
    if count == 5:
      break
  return text, labels3


# Majority vote function
def majority_vote(alist):
  '''Using majority vote, classifies whether it is
  hate speech (= 1) or not (= 0).'''
  label_res = []

  for i in alist:
    zero_count = 0

    for tag in i:
      if tag == 0:
        zero_count += 1

    if zero_count >= 2:
      hate = 0
    else:
      hate = 1

    label_res.append(hate)

  return label_res


# Annotators' agreement measurement

def annot_agreement(alist):
    '''Measures agreement based on a list with three votes. Returns the sum of
    all zeros, one zero, two zeros, all hate labels, and equal labelling of
    hate (given that all annotators label as hate) per tweet.'''
    all_zeros = 0
    one_zero = 0
    two_zero = 0
    all_hate = 0
    #equal = 0

    for i in alist:
      zero_count = 0
      for tag in i:
        if tag == 0:
          zero_count += 1



      if zero_count == 3:
        all_zeros += 1
      elif zero_count == 2:
        two_zero += 1
      elif zero_count == 1:
        one_zero += 1
      else:
        all_hate += 1

        # Count agreement
        #if i[0] == i[1] == i[2]:
          #equal += 1

    return all_zeros, one_zero, two_zero, all_hate

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn import svm
from sklearn.metrics import classification_report
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
# from keras.utils.np_utils import to_categorical
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

pos_rev = pd.read_csv("Data/pos.txt", sep = "\n", header = None, encoding = 'latin-1')
pos_rev = pd.concat([pos_rev,pd.Series(np.ones(pos_rev.shape[0]))], ignore_index=True, axis =1)
pos_rev.columns = ['review', 'mood']

print(pos_rev.shape)
print(pos_rev.head())

neg_rev = pd.read_csv("Data/negative.txt", sep = "\n", header = None, encoding = 'latin-1')
neg_rev = pd.concat([neg_rev,pd.Series(np.zeros(pos_rev.shape[0]))], ignore_index=True, axis =1)
neg_rev.columns = ['review', 'mood']
print(neg_rev.head())


# pre-processing on positive reviews
pos_rev.loc[:, 'review'] = pos_rev.loc[:, 'review'].apply(lambda x: x.lower())
pos_rev.loc[:, 'review'] = pos_rev.loc[:, 'review'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

# pre-processing on negative reviews
pos_rev.loc[:, 'review'] = pos_rev.loc[:, 'review'].apply(lambda x: x.lower())
pos_rev.loc[:, 'review'] = pos_rev.loc[:, 'review'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

#connecting both pos and negative review
com_rev = pd.concat([pos_rev, neg_rev], axis =0).reset_index()

print(com_rev.head(10))


X_train, X_test, Y_train, Y_test = train_test_split(com_rev['review'].values,com_rev['mood'].values, test_size = 0.33, random_state = 42)

train_data = pd.DataFrame({'review':X_train, 'mood':Y_train})
test_data = pd.DataFrame({'review':X_test, 'mood':Y_test})

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)

train_vectors = vectorizer.fit_transform(train_data['review'])
test_vectors = vectorizer.transform(test_data['review'])

#saving the transform model
pickle.dump(vectorizer, open('tranform.pkl', 'wb'))

classifier_linear = svm.SVC(kernel='linear')
classifier_linear.fit(train_vectors, train_data['mood'])
prediction_linear = classifier_linear.predict(test_vectors)

report = classification_report(test_data['mood'], prediction_linear, output_dict=True)
print('positive:', report['1.0']['recall'])
print('negative:', report['0.0']['recall'])

#saving the svm model
pickle.dump(classifier_linear, open('model.pkl', 'wb'))
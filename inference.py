import pickle

data = [" xyz do not verify that the information provided is updated and accurate"]
# data = ["xyz do not verify that the information provided is updated and accurate"]
#loading the transform model
tfidf=pickle.load(open('tranform.pkl','rb'))


# loading the model
clf = pickle.load(open('model.pkl', 'rb'))

vect = tfidf.transform(data).toarray()
my_prediction = clf.predict(vect)
print(my_prediction)
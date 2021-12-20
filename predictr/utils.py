import pickle
from sklearn.naive_bayes import GaussianNB

# define a Gaussain NB classifier
clf = GaussianNB()

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

bug_classes = {0: False, 1: True}
bug_r_classes = {y: x for x, y in bug_classes.items()}

# function to load the model
def load_model():
    global clf
    clf = pickle.load(open("models/iris_nb.pkl", "rb"))
    
def load_bug_model():
    global clf
    clf = pickle.load(open("models/bug_nb.pkl", "rb"))


# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])[0]
    return classes[prediction]


def predict_bug(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict_bug([x])[0]
    return bug_classes[prediction]
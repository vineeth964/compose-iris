import os
import pickle
from sklearn.naive_bayes import GaussianNB

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

bug_classes = {0: False, 1: True}
bug_r_classes = {y: x for x, y in bug_classes.items()}

# function to train and load the model during startup
def init_model():
    if not os.path.isfile("models/iris_nb.pkl"):
        clf = GaussianNB()
        pickle.dump(clf, open("models/iris_nb.pkl", "wb"))
    if not os.path.isfile("models/bug_nb.pkl"):
        clf = GaussianNB()
        pickle.dump(clf, open("models/bug_nb.pkl", "wb"))


# function to train and save the model as part of the feedback loop
def train_model(data):
    # load the model
    clf = pickle.load(open("models/iris_nb.pkl", "rb"))

    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)

    # save the model
    pickle.dump(clf, open("models/iris_nb.pkl", "wb"))


def train_bug_model(data):
    # load the model
    clf = pickle.load(open("models/bug_nb.pkl", "rb"))

    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [bug_r_classes[d.defects] for d in data]

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)

    # save the model
    pickle.dump(clf, open("models/bug_nb.pkl", "wb"))
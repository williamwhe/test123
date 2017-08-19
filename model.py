import pdb
from sklearn import metrics
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# import tensorflow as tf
import math
from sklearn.ensemble import VotingClassifier
import cPickle
from sklearn.externals import joblib
class Discriminator(object):
    """
    The basic class for discriminator
    """
    def train(self, X, y):
        """ train the model
        :param X: the data matrix
        :param y: the label
        """
        raise NotImplementedError("Abstract method")

    def predict(self, X):
        """ predict the label of X
        :param X: the data matrix
        :return: the label of X
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """ evaluate the classification performance of X with respect of y
        :param X: the test data
        :param y: the ground truth label of X
        :return: a dict of performance scores
        """
        return self._score(y, self.model.predict_proba(X)[:, 1])

    def predict_proba(self, X):
        return self.model.predict_proba(X)


    def _score(self, true_label, predicted_prob):
        """ calculate the performance score for binary calssification
        :param true_label: the ground truth score
        :param predicted_label: the predicted probability
        :return: a dict of scores
        """
        score_dict = dict()
        score_dict['AUC'] = metrics.roc_auc_score(true_label, predicted_prob)
        predicted_label = [0 if prob < 0.5 else 1 for prob in predicted_prob]
        score_dict['Accuracy'] = metrics.accuracy_score(true_label, predicted_label)
        cm = metrics.confusion_matrix(true_label, predicted_label)
        score_dict['Confusion Matrix'] = cm
        #TPR = TP /(TP + FN)
        score_dict['TPR'] = cm[1, 1] / float(cm[1, 0] + cm[1, 1])
        score_dict['FPR'] = cm[0, 1] / float(cm[0, 0] + cm[0, 1])
        # score_dict['FNR'] = cm[0, 0] / float(cm[0,0] + cm[0, 1])
        #FNR = FN/(TP + FN)
        score_dict['FNR'] = cm[1, 0] / float(cm[1,0] + cm[1, 1])
        return score_dict

    def save(self, fname):
        with open(fname, 'wb') as fid:
            cPickle.dump(self.model, fid)

    def load(self, fname):
        with open(fname, 'rb') as fid:
            self.model = cPickle.load(fid)
    
    # def save(self, fname):
    #     joblib.dump(self.model, fname + '.pkl')

    # def load(self, fname):
    #     self.model = joblib.load(fname + ".pkl")

    def getName(self):
        return self.name

class RandomForrest(Discriminator):
    """
    using RF as the discriminator
    """

    def __init__(self, num_trees=100, num_threads=25):
        self.num_trees = num_trees
        self.num_threads = num_threads
        self.model = None
        self.name = "rf"
    def train(self, X, y):
        self.model = RandomForestClassifier(n_estimators=self.num_trees, n_jobs=self.num_threads)
        self.model.fit(X, y)


class LR(Discriminator):
    """
    using RF as the discriminator
    """

    def __init__(self):
        self.model = None
        self.name = "lr"
    def train(self, X, y):
        self.model = LogisticRegression()
        self.model.fit(X, y)


class DT(Discriminator):
    """
    using RF as the discriminator
    """

    def __init__(self):
        self.model = None
        self.name = "dt"

    def train(self, X, y):
        self.model = DecisionTreeClassifier()
        self.model.fit(X, y)


class NB(Discriminator):
    """
    using RF as the discriminator
    """

    def __init__(self):
        self.model = None
        self.name = "nb"

    def train(self, X, y):
        self.model = BernoulliNB()
        self.model.fit(X, y)


class MLP(Discriminator):
    """
    using RF as the discriminator
    """

    def __init__(self, layers=(100,)):
        self.layers = layers
        self.model = None
        self.name = "mlp"
    def train(self, X, y):
        self.model = MLPClassifier(hidden_layer_sizes=self.layers, activation = 'relu', early_stopping=True)
        self.model.fit(X, y)
    def predict(self, X, y):
        y_pred = self.model.predict(X)
        y_binary = np.zeros( y.shape[0] ) 
        collision_id =np.where(  np.sum( abs(y_pred - y ), axis = 1) > 0)
#        collision_id = np.array_equal(y_pred == y)
        y_binary[collision_id]= 1
#        pdb.set_trace()
        return y_binary

class SVM(Discriminator):
    """
    using RF as the discriminator
    """

    def __init__(self):
        self.model = None
        self.name = "svm"
    def train(self, X, y):
        self.model = SVC(probability=True)
        self.model.fit(X, y)


class KNN(Discriminator):
    """
    using RF as the discriminator
    """

    def __init__(self):
        self.model = None
        self.name = "knn"
    def train(self, X, y):
        self.model = KNeighborsClassifier(n_jobs=10)
        self.model.fit(X, y)


class VOTE(Discriminator):
    """
    using RF as the discriminator
    """

    def __init__(self):
        self.model = None
        self.name = "vote"
    def train(self, X, y):
        clf1 = LogisticRegression()
        clf2 = RandomForestClassifier(n_estimators=100, n_jobs=10)
        clf3 = BernoulliNB()
        clf4 = DecisionTreeClassifier()
        clf5 = MLPClassifier(early_stopping=True)
        clf6 = SVC(probability=True)
        clf7 = KNeighborsClassifier(n_jobs=10)
        self.model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('bnb', clf3), ('dt', clf4),
                                                  ('mlp', clf5), ('svm', clf6), ('knn', clf7)], voting='soft')
        self.model.fit(X, y)


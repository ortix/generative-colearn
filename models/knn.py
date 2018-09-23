from sklearn.neighbors import KNeighborsRegressor as nb
from sklearn.externals import joblib
from timeit import default_timer as timer

class KNN():

    def __init__(self, n_neighbors=5,
                 weights='distance',
                 algorithm='auto',
                 leaf_size=10,
                 metric='minkowski', **kwargs):
        print("Initializing KNN")
        self.model = nb(n_neighbors=n_neighbors,
                        weights=weights,
                        algorithm=algorithm,
                        leaf_size=leaf_size,
                        metric=metric,
                        n_jobs=1)

    def save_weights(self, path):
        joblib.dump(self.model, path)

    def predict(self, labels):
        return self.model.predict(labels)

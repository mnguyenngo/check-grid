import datetime as dt
import spacy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from task import Task


class Agenda(object):
    """Agenda object holds task objects"""

    def __init__(self):
        """Creates a new agenda as an empty list"""
        self._created_at = dt.datetime.utcnow()
        self._agenda = []
        self.X = None

    def show_agenda(self):
        return [task.text for task in self._agenda]

    def add_task(self, task):
        """Add new task object to agenda"""
        self._agenda.append(Task(task))

    def build_matrix(self):
        """Returns matrix of agenda data"""
        # Load spaCy nlp model
        nlp = spacy.load('en')
        # Get the vector for each task
        vectorlist = [nlp(task.text).vector for task in self._agenda]
        # Stack the vectors to create a matrix
        X = np.stack(vectorlist)
        # Set X as an attribute of the class object
        self.X = X
        return X

    def get_X(self):
        """Returns the feature matrix of the tasks in the agenda
        Used by multiple methods
        """
        if self.X is None:
            X = self.build_matrix()
        else:
            X = self.X
        return X

    def kmeans_cat(self, n_clusters=3):
        X = self.get_X()
        # Initialize KMeans classifier and classify data
        km = KMeans(n_clusters)
        labels = km.fit_predict(X)

        # Print classified tasks
        for label in np.unique(labels):
            print(label)
            indices = np.argwhere(labels == label).flatten()
            for idx in indices:
                print(self._agenda[idx].text)

    def plot_agenda_2d(self):
        """Returns 2d plot of data with respect to its first and second
        principal components
        """
        X = self.get_X()
        # Run the PCA on the matrix
        pcd = PCA(2).fit(X)
        # Get the data points to plot in a 2d plot
        pcd_points = np.dot(X, pcd.components_.T)

        # Plot the points
        fig, ax = plt.subplots(figsize=(12, 12))
        x = pcd_points[:, 0]
        y = pcd_points[:, 1]
        ax.scatter(x, y)
        # Label each point with the task item
        for i, task in enumerate(self._agenda):
            ax.annotate(task.text, (x[i], y[i]))
        plt.show()

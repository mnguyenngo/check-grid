import datetime as dt
import spacy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from task import Task


class Agenda(object):
    """Agenda object holds task objects"""

    def __init__(self):
        """Creates a new agenda as an empty list"""
        self._created_at = dt.datetime.utcnow()
        self._agenda = []

    def show_agenda(self):
        return [task.text for task in self._agenda]

    def add_task(self, task):
        """Add new task object to agenda"""
        self._agenda.append(task)

    def plot_agenda_2d(self):
        """Returns 2d plot of data with respect to its first and second
        principal components
        """
        # Load spaCy nlp model
        nlp = spacy.load('en')
        # Get the vector for each task
        vectorlist = [nlp(task.text).vector for task in self._agenda]
        # Stack the vectors to create a matrix
        X = np.stack(vectorlist)
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

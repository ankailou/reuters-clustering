#!/usr/local/python-2.7.5/bin/python

""" kmeans.py
    ---------
    @author = Ankai Lou
"""

###############################################################################
################ modules required to compute k-means clusters #################
###############################################################################

from sklearn.cluster import KMeans
from math import log
import time

###############################################################################
############ class definition for k-means center-based clustering #############
###############################################################################

class KMeansCluster:
    def __init__(self, n_clusters=10):
        """ function: contructor
            --------------------
            instantiate a kmean clustering algorithm
        """
        self.name = "k-means"
        self.km = None
        self.n_clusters = n_clusters
        self.clusters = dict([])

    ###########################################################################
    ######### general helper functions for measuring cluster quality ##########
    ###########################################################################

    def __compute_variance(self):
        """ function: compute_variance
            --------------------------
            compute the variance/skew across cluster sizes

            returns: variance of the clustering
        """
        mean, variance = 0.0, 0.0
        for key, cluster in self.clusters.iteritems():
            mean += len(cluster)
        mean /= self.n_clusters
        for key, cluster in self.clusters.iteritems():
            variance += (len(cluster) - mean) ** 2
        variance /= self.n_clusters
        return variance


    def __compute_entropy(self,dataset):
        """ function: compute_entropy
            -------------------------
            compute the entropy of @self.vectors or @self.clusters

            returns: entropy scores of the dataset
        """
        entropy = 0.0
        for key, cluster in dataset.iteritems():
            length = len(cluster)
            factor = float(length) / len(self.vectors)
            temp = 0.0
            for topic in self.topics:
                inner = 0.0
                for fv in cluster:
                    if topic in fv.topics:
                        inner += 1.0
                if inner > 0:
                    inner /= float(length)
                    temp += -inner * log(inner, 2)
            entropy += factor * temp
        return entropy

    ###########################################################################
    ################ mains to generate and test the clustering ################
    ###########################################################################

    def generate_clusters(self, feature_vectors):
        """ function: generate_clusters
            ---------------------------
            generate k-means clusters for feature vectors

            :param feature_vectors: set of features to construct model
        """
        # begin with empty clusters
        for i in xrange(0,self.n_clusters): self.clusters[i] = []
        # generate clusters
        cluster_start = time.time()
        fv_space, topic_space = [], []
        for key, fv in feature_vectors.iteritems():
            fv_space.append(fv.vector)
            topic_space.append(fv.topics)
        self.km = KMeans(init='k-means++', n_clusters=self.n_clusters)
        clusters = self.km.fit_predict(fv_space)
        # split dataset based on clusters
        for i, index in enumerate(clusters):
            self.clusters[index].append(feature_vectors[i])
        cluster_time = time.time() - cluster_start
        # set object members for entropy calculation
        self.vectors = feature_vectors
        self.topics = set().union(*topic_space)
        # compute entropy of clusters + gain
        all = self.vectors.values()
        before = self.__compute_entropy({ 0 : self.vectors.values() })
        after = self.__compute_entropy(self.clusters)
        print "Entropy Before Clustering:",before
        print "Entropy After Clustering :",after
        print "Overall Gain in Entropy:", before - after
        # compute variance of cluster sizes + time
        print "Clustering Variance:", self.__compute_variance()
        print "Time for Clustering:", cluster_time, "seconds"
        # reset clusters
        self.clusters = dict([])

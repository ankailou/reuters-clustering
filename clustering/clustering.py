#!/usr/local/python-2.7.5/bin/python

""" clustering.py
    -------------
    @author = Ankai Lou
"""

###############################################################################
############ modules & libraries required for clustering articles #############
###############################################################################

from algorithm import *

###############################################################################
############################## global variables ###############################
###############################################################################

n_clusters = 10
epsilon = 0.1
min_pts = 5

###############################################################################
######################## list of clustering algorithms ########################
###############################################################################

clusterings = [kmeans.KMeansCluster(n_clusters),
               dbscan.DBScanCluster(epsilon, min_pts)]

###############################################################################
################ strings representing classifier experiements #################
###############################################################################

fv_dataset_name = ["standard feature vector","pared feature vector"]

###############################################################################
################# main function for single-point-of-execution #################
###############################################################################

def begin(feature_vectors):
    """ function: begin
        ---------------
        use N classifiers with M metrics on P feature vector datasets

        :param feature_vectors: standard dataset generated using tf-idf
    """
    for i, dataset in enumerate(feature_vectors):
        for j, cluster in enumerate(clusterings):
            print "\nExperiment:", cluster.name, "on", fv_dataset_name[i]
            cluster.generate_clusters(dataset)

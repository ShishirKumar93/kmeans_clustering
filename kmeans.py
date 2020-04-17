#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:58:52 2020

@author: MrMndFkr
"""
import numpy as np
import random
from collections import defaultdict
from typing import Mapping

def kmeans(X:np.ndarray, k=5, centroids=None, tolerance=1e-2, seed=42) -> (np.ndarray, Mapping[int,list]):
    """ 
    Returns centroids and list of row indexes for each cluster for given dataset and no of clusters 
    """
    if centroids == 'kmeans++':
        ## Kmeans++ algorithm for initial centroid identification
        ## pick first centroid randomly, pick next centroid where 
        ## minimum distance to the previous clusters is maximized
        first_centroid_idx = np.random.choice(X.shape[0], 1, replace=False)
        first_centroid = X[first_centroid_idx]
        dist = np.zeros((X.shape[0], k))
        dist[:] = np.nan ## setting all elements to nan for easier row wise minimum calculation
        init_centroids = first_centroid
        for j in range(k-1):
            dist[:, j] = np.linalg.norm(X - first_centroid, axis = 1)
            min_dist = np.nanmin(dist, axis = 1, keepdims = True)
            next_centroid_idx = np.argmax(min_dist, axis = 0)
            next_centroid = X[next_centroid_idx]
            init_centroids = np.concatenate((init_centroids, next_centroid), axis = 0)
            first_centroid = next_centroid
    else:
        ## Pick k elements randomly as the initial centroids
        X_unique = np.unique(X, axis=0)
        init_centroids_idx = np.random.choice(X_unique.shape[0], k, replace=False)
        init_centroids = X_unique[init_centroids_idx]
    while True:
        cluster_record_map = [[] for _ in range(k)]
        for i in range(X.shape[0]):
            cluster = np.argmin(np.linalg.norm(X[i] - init_centroids, axis=1))
            cluster_record_map[cluster].append(i)
        next_centroids = np.array([np.mean(X[cluster_record_map[j]], axis=0) for j in range(k)])
        dist_bw_centroids = np.all(np.linalg.norm(init_centroids - next_centroids, axis=1))
        if np.any(dist_bw_centroids) >= tolerance:
            init_centroids = next_centroids
        else:
            return next_centroids, cluster_record_map

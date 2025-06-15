import numpy as np
from sklearn.cluster import DBSCAN

def cluster_keypoints(features, eps=20, min_samples=2):
    """
    Applies DBSCAN clustering to the given features.

    Parameters:
    - features: np.array of feature vectors (usually keypoint coordinates or descriptors)
    - eps: DBSCAN epsilon parameter (maximum distance between two samples to be in the same cluster)
    - min_samples: minimum number of samples in a cluster

    Returns:
    - labels: array of cluster labels (-1 indicates noise)
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    return clustering.labels_


# Example usage
if __name__ == "__main__":
    # Dummy keypoint coordinates for testing
    dummy_features = np.array([
        [10, 10], [12, 12], [11, 11],   # cluster 1
        [100, 100], [102, 102],         # cluster 2
        [300, 300]                      # noise
    ])

    labels = cluster_keypoints(dummy_features, eps=5, min_samples=2)
    print("Cluster Labels:", labels)

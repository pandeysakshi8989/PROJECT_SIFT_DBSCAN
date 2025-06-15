import numpy as np

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    - point1: tuple or array (x1, y1)
    - point2: tuple or array (x2, y2)

    Returns:
    - distance: float value representing Euclidean distance
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)


def distance_matrix(points):
    """
    Compute pairwise Euclidean distance matrix for a list of points.

    Parameters:
    - points: list of tuples [(x1, y1), (x2, y2), ...]

    Returns:
    - matrix: 2D numpy array of shape (n_points, n_points)
    """
    n = len(points)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = euclidean_distance(points[i], points[j])
    return matrix


# Example usage
if __name__ == "__main__":
    sample_points = [(0, 0), (3, 4), (6, 8)]
    print("Pairwise Distance Matrix:")
    print(distance_matrix(sample_points))

import numpy as np

def compute_confidence(similarity_scores, threshold=0.5):
    """
    Compute confidence based on similarity scores.

    Parameters:
    - similarity_scores: list or numpy array of similarity scores (e.g., feature matching distances)
    - threshold: minimum similarity score to be considered confident

    Returns:
    - confidence: float between 0 and 1 representing the fraction of confident matches
    """
    similarity_scores = np.array(similarity_scores)
    confident_matches = similarity_scores >= threshold
    confidence = np.sum(confident_matches) / len(similarity_scores) if len(similarity_scores) > 0 else 0
    return confidence


# Example usage
if __name__ == "__main__":
    sample_scores = [0.8, 0.9, 0.3, 0.7, 0.6, 0.2]
    threshold = 0.5
    confidence = compute_confidence(sample_scores, threshold)
    print(f"Confidence (threshold={threshold}): {confidence:.2f}")

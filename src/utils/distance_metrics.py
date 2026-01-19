import torch
import numpy as np
import torch.nn.functional as F

def cosine_distance(X, Y, diagonal_only=False):
    """
    Compute cosine distance between two tensors.

    Args:
        x: Tensor of shape (N, D) or (D,)
        y: Tensor of shape (M, D) or (D,)
        eps: small value to avoid division by zero

    Returns:
        Cosine distance tensor of shape (N, M) if both are 2D,
        otherwise scalar distance if both are 1D.

        Defined as d = 1 - sim
        Values: [0, 2]
        0 = identical vectors
        2 = completely opposite

        
        distances: 
          - shape (N, M) if diagonal_only=False
          - shape (min(N, M),) if diagonal_only=True
    """
    # Normalize vectors to unit length
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    
    # Compute cosine similarity
    sim_matrix = np.dot(X_norm, Y_norm.T)
    
    # Convert to cosine distance
    dist_matrix = 1.0 - sim_matrix
    
    if diagonal_only:
        # Return only aligned pairs (min(N, M))
        return np.diag(dist_matrix)
    else:
        return dist_matrix


def cosine_similarity(X, Y, diagonal_only=True):
    """
    Compute cosine similarity between two arrays.
    """
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    
    sim_matrix = np.dot(X_norm, Y_norm.T)
    
    if diagonal_only:
        return np.diag(sim_matrix)
    else:
        return sim_matrix

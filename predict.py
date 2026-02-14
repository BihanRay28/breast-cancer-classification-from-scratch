import numpy as np
from nn_core import forward_prop

def predict(parameters, X, threshold=0.5):
    cache = forward_prop(parameters, X)
    A2 = cache["A2"]
    return (A2 > threshold).astype(int)

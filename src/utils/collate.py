import numpy as np

def numpy_collate(batch):
    return [np.stack(b, axis=0) for b in zip(*batch)]
import numpy as np


def dice3d(a, b, smooth=1e-10):
    """
    Compute the dice similarity coefficient for two 3D volumes.

    Arguments:
        a (numpy array): 3D array
        b (numpy array): 3D array
        smooth (float): small number to avoid division by 0
    Returns:
        dice coefficient (float)
    """
    a = a.cpu().detach().numpy()
    b = b.cpu().detach().numpy()
    intersection = np.sum((a * b))
    volumes = np.sum(a * a) + np.sum(b * b)
    return 2 * float(intersection) / (volumes + smooth)


def precision(label, pred):
    tp = (label[label==pred]).sum()
    fn = (label[label!=pred]).sum()
    return tp / (tp + fn)


def recall(label, pred):
    tp = (label[label==pred]).sum()
    fp = (pred[pred!=label]).sum()
    return tp / (tp + fp)
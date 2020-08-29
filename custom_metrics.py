from keras import backend as K
import numpy as np

SMOOTH = 1.

def dice_coef(y_true, y_pred):
    """

    :param y_true: the labeled mask corresponding to a mammogram scan
    :param y_pred: the predicted mask of the scan
    :return:  A metric that accounts for precision and recall
              on the scale from 0 - 1. The closer to 1, the
              better.
    Dice = 2 * (|X & Y|)/ |X|+ |Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|))
    Citation (MIT License): https://github.com/jocicmarko/
                            ultrasound-nerve-segmentation/blob/
                            master/train.py
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def dice_score(y_true, y_pred):
      if y_true.shape != y_pred.shape:
          raise ValueError("Shape mismatch: y_true and y_pred must have the same size.")
      else:
          y_true_f = y_true.flatten()
          y_pred_f = y_pred.flatten()
          intersection = np.sum(y_true_f * y_pred_f)
          
          dice_score = (2. * intersection + SMOOTH) / (np.sum(np.abs(y_true)) + np.sum(np.abs(y_pred)) + SMOOTH)

          return dice_score

def iou_score(y_true, y_pred):
    """
    Computes the IoU between the predicted mask and the true one
    :param y_true:
    :param y_pred:
    :return:
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch: y_true and y_pred must have the same size.")
    else:
        intersection = np.logical_and(y_true, y_pred)
        union = np.logical_or(y_true, y_pred)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score
from torchmetrics import CalibrationError
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def accuracy(output, target):
    return accuracy_score(target, output)


def precision(output, target):
    return precision_score(target, output, average='weighted', zero_division=True)


def recall(output, target):
    return recall_score(target, output, average='weighted', zero_division=True)


def f1(output, target):
    return f1_score(target, output, average='weighted')


def ECE(output, target):
    calibration_error = CalibrationError(task="multiclass", num_classes=2)
    calibration_error.update(output, target)
    return calibration_error.compute()

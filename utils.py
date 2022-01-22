import os
import logging

from sklearn.metrics import accuracy_score, f1_score


def set_logger(name):
    """
    Write logs to checkpoint and console.
    """

    log_file = os.path.join('./log', name)

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def evaluate(truth, prediction):
    accuracy = accuracy_score(truth, prediction)
    f1_micro = f1_score(truth, prediction, average='micro')
    f1_macro = f1_score(truth, prediction, average='macro')
    f1_weighted = f1_score(truth, prediction, average='weighted')
    return accuracy, f1_micro, f1_macro, f1_weighted
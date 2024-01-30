import datetime
import numpy as np


def logger(foo):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'[{time_str}] {foo}')


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, current_loss):
        if current_loss < self.min_validation_loss:
            self.min_validation_loss = current_loss
            self.counter = 0
        elif current_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

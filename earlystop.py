import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=1, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_max = -np.inf
        self.delta = delta

    def __call__(self, score, model):
        """
        In our context, the score refers to the validation set accuracy.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'Limited accuracy improvement, EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation accuracy increase.'''
        if self.verbose:
            print(f'Validation accuracy increased ({self.score_max:.6f} --> {score:.6f}).  Saving model ...')
        model.save_networks('best')
        self.score_max = score
        
    def adjust_delta(self, new_delta):
        print(f'earlystop delta adjusted from {self.delta} to {new_delta}')
        self.delta = new_delta
        self.early_stop = False
        self.counter = 0
        

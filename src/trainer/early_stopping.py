class EarlyStopping:
    def __init__(self, mode="min", patience=5, min_delta=0.0):
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.num_bad_epochs = 0

    def _is_improvement(self, current):
        if self.best_score is None:
            return True

        if self.mode == "min":
            return current < self.best_score - self.min_delta
        elif self.mode == "max":
            return current > self.best_score + self.min_delta
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def step(self, current):
        if self._is_improvement(current):
            self.best_score = current
            self.num_bad_epochs = 0
            return False, True   # stop=False, improved=True
        else:
            self.num_bad_epochs += 1
            stop = self.num_bad_epochs >= self.patience
            return stop, False
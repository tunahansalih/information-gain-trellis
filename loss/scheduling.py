import math


class TimeBasedDecay:
    def __init__(self, initial_value, decay):
        self.initial_value = initial_value
        self.decay = decay

    def get_current_value(self, step):
        return 1 / (1. + self.decay * step)


class StepDecay:
    def __init__(self, initial_value, decay, decay_step):
        self.initial_value = initial_value
        self.decay = decay
        self.decay_step = decay_step

    def get_current_value(self, step):
        return self.initial_value * math.pow(self.decay, math.floor((1 + step) / self.decay_step))


class ExponentialDecay:
    def __init__(self, initial_value, decay):
        self.initial_value = initial_value
        self.decay = decay

    def get_current_value(self, step):
        return self.initial_value * math.pow(math.e, -self.decay * step)


class EarlyStopping:
    def __init__(self, initial_value, stopping_step):
        self.initial_value = initial_value
        self.stopping_step = stopping_step

    def get_current_value(self, step):
        if step >= self.stopping_step:
            return 0
        return self.initial_value

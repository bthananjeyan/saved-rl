import numpy as np

class LinearSchedule(object):

    def __init__(self, start, end, end_time):
        self.start = start
        self.end = end
        self.end_time = end_time
        self.rate = (end - start) / end_time

    def __call__(self, t):
        if t > self.end_time:
            return self.end
        return max(self.start + self.rate * t, 0)
    
class InterpolatingLinearSchedule(object):
    
    def __init__(self, start, end, end_time, window_length):
        self.start = start
        self.end = end
        self.end_time = end_time
        self.rate = (end - start) / end_time
        self.switch_interval = window_length//2

    def __call__(self, t):
        if t > self.end_time:
            return self.end
        
        if (t // self.switch_interval) % 2 == 0:    
            return max(self.start + self.rate * t, 0)
        else:
            return 0
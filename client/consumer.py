import time
import numpy as np


class Timelet():
    time_let = []

    def __init__(self,window=100):
        self.window=window

    def append(self,sample):
        if self.time_let is None:
            self.time_let = [sample]
        else:
             self.time_let = np.vstack((self.time_let,sample))
        if self.time_let[-1][0]-self.time_let[0][0] >= self.window:
            return self.time_let
        else:
            return None

class TSConsumer:

    processes = []
    timelet = None

    def new_sample(self, value, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        if self.timelet is None:
           self.timelet = Timelet()
        ts = self.timelet.append([timestamp, value])
        if ts is not None:
            for process in self.processes:
                pass
            self.timelet = Timelet()












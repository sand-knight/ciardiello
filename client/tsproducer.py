from connectors import FakeConnector

import time
class TSProd:

    connector =None
    def __init__(self, tsfile, connector,name, freq=None):
        self.tsfile =tsfile
        self.connector = connector
        self.freq = freq
        self.name = name


    def start(self):
        with  open(self.tsfile, "r") as infile:
             while True:
               sample = float(infile.readline())
               if self.freq is not None:
                   self.connector.send([sample])
                   time.sleep(1/self.freq)


if __name__=="__main__":
    tsp = TSProd("data/experiments/gold.csv",FakeConnector("fake1"), "prod",50)
    tsp.start()


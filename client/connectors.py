import consumer

class Connector:
    def __init__(self, name):
        self.name=name
    def send(self, sample):
        pass


class MqttConnector(Connector):

    def send(self,sample):
        pass


class FakeConnector(Connector):


      def __init__(self, name):
          self.name=name
          self.consumer = consumer.TSConsumer()

      def send(self, sample):
          self.consumer.new_sample(sample)
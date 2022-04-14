

class TopicModel:

    def __init__(self, dataset, topic_model, k):
        self.dataset = dataset
        self.topic_model = topic_model
        self.k = k
        
        
    def train(self):
        raise NotImplementedError("Train function has not been defined!")
        
        
    def evaluate(self):
        raise NotImplementedError("Evaluate function has not been defined!")
        
    
    def get_topics(self):
        raise NotImplementedError("Get topics function has not been defined!")

            


class TopicModel:

    def __init__(self, dataset, topic_model, num_topics):
        self.dataset = dataset
        self.topic_model = topic_model
        self.num_topics = num_topics
        
        
    def train(self):
        raise NotImplementedError("Train function has not been defined!")
        
        
    def evaluate(self):
        raise NotImplementedError("Evaluate function has not been defined!")
        
    
    def get_topics(self):
        raise NotImplementedError("Get topics function has not been defined!")

            
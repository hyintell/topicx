from baselines.topic_model import TopicModel
from octis.models.CTM import CTM
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence


class ZeroShotTM(TopicModel):
    def __init__(self, dataset, topic_model, num_topics, embedding):
        super().__init__(dataset, topic_model, num_topics)
        print(f'Initialize ZeroShotTM with num_topics={num_topics}, embedding={embedding}')
        self.embedding = embedding
        self.model = CTM(num_topics=num_topics, inference_type='zeroshot', bert_model=embedding)
        
    
    def train(self):
        self.output = self.model.train_model(self.dataset)
    
    
    def evaluate(self):
        # Initialize metric
        npmi = Coherence(texts=self.dataset.get_corpus(), topk=10, measure='c_npmi')
        cv = Coherence(texts=self.dataset.get_corpus(), topk=10, measure='c_v')
        topic_diversity = TopicDiversity(topk=10)
        
        # score
        td_score = topic_diversity.score(self.output)
        cv_score = cv.score(self.output)
        npmi_score = npmi.score(self.output)
        
        return td_score, cv_score, npmi_score
    
    def get_topics(self):
        return self.output['topics']
    
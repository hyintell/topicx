from baselines.topic_model import TopicModel
from octis.models.ProdLDA import ProdLDA
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence


class ProdLDATM(TopicModel):
    def __init__(self, dataset, topic_model, num_topics):
        super().__init__(dataset, topic_model, num_topics)
        print(f'Initialize ProdLDA with num_topics={num_topics}')
        self.model = ProdLDA(num_topics=num_topics)
        
        
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
    
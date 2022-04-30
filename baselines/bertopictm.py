from baselines.topic_model import TopicModel
from baselines.bertopic import BERTopic
import pandas as pd
import gensim.corpora as corpora
from flair.embeddings import TransformerDocumentEmbeddings
from gensim.models.coherencemodel import CoherenceModel


class BERTopicTM(TopicModel):
    def __init__(self, dataset, topic_model, num_topics, embedding):
        super().__init__(dataset, topic_model, num_topics)
        print(f'Initialize BERTopicTM with num_topics={num_topics}, embedding={embedding}')
        self.embedding = embedding
        
        # make sentences and token_lists
        token_lists = self.dataset.get_corpus()
        self.sentences = [' '.join(text_list) for text_list in token_lists]
        embedding_model = TransformerDocumentEmbeddings(embedding)
        self.model = BERTopic(embedding_model=embedding_model, nr_topics=num_topics)     
        
        
    def train(self):
        self.topics, _ = self.model.fit_transform(self.sentences)
    
    
    def evaluate(self):
        td_score = self._calculate_topic_diversity()
        cv_score, npmi_score = self._calculate_cv_npmi(self.sentences, self.topics)
        
        return td_score, cv_score, npmi_score
    
    
    def get_topics(self):
        return self.model.get_topics()
    
    
    def _calculate_topic_diversity(self):
        topic_keywords = self.model.get_topics()

        bertopic_topics = []
        for k,v in topic_keywords.items():
            temp = []
            for tup in v:
                temp.append(tup[0])
            bertopic_topics.append(temp)  

        unique_words = set()
        for topic in bertopic_topics:
            unique_words = unique_words.union(set(topic[:10]))
        td = len(unique_words) / (10 * len(bertopic_topics))

        return td


    def _calculate_cv_npmi(self, docs, topics): 

        doc = pd.DataFrame({"Document": docs,
                        "ID": range(len(docs)),
                        "Topic": topics})
        documents_per_topic = doc.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = self.model._preprocess_text(documents_per_topic.Document.values)

        vectorizer = self.model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        words = vectorizer.get_feature_names()
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topic_words = [[words for words, _ in self.model.get_topic(topic)] 
                    for topic in range(len(set(topics))-1)]

        coherence_model = CoherenceModel(topics=topic_words, 
                                      texts=tokens, 
                                      corpus=corpus,
                                      dictionary=dictionary, 
                                      coherence='c_v')
        cv_coherence = coherence_model.get_coherence()

        coherence_model_npmi = CoherenceModel(topics=topic_words, 
                                      texts=tokens, 
                                      corpus=corpus,
                                      dictionary=dictionary, 
                                      coherence='c_npmi')
        npmi_coherence = coherence_model_npmi.get_coherence()

        return cv_coherence, npmi_coherence 
    
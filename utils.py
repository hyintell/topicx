from octis.dataset.dataset import Dataset

def prepare_dataset(dataset_name):
    
    dataset = Dataset()
    
    if dataset_name == '20ng':
        dataset.fetch_dataset('20NewsGroup')
    elif dataset_name == 'bbc':
        dataset.fetch_dataset('BBC_news')
    elif dataset_name == 'm10':
        dataset.fetch_dataset('M10')
        
    # make sentences and token_lists
    token_lists = dataset.get_corpus()
    sentences = [' '.join(text_list) for text_list in token_lists]
    
    return dataset, sentences

from ._base import BaseEmbedder
from ._sentencetransformers import SentenceTransformerBackend


def select_backend(embedding_model):

    # Flair word embeddings
    if "flair" in str(type(embedding_model)):
        from ._flair import FlairBackend
        return FlairBackend(embedding_model)

    # Create a Sentence Transformer model based on a string
    if isinstance(embedding_model, str):
        return SentenceTransformerBackend(embedding_model)


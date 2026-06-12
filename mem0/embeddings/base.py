from abc import ABC, abstractmethod
from typing import Literal, Optional

from mem0.configs.embeddings.base import BaseEmbedderConfig


class EmbeddingBase(ABC):
    """Initialized a base embedding class

    :param config: Embedding configuration option class, defaults to None
    :type config: Optional[BaseEmbedderConfig], optional
    """

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        if config is None:
            self.config = BaseEmbedderConfig()
        else:
            self.config = config

    @abstractmethod
    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]]):
        """
        Get the embedding for the given text.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        pass

    def embed_batch(self, texts, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get embeddings for multiple texts.

        Args:
            texts: Iterable of text strings to embed.
            memory_action (optional): The type of embedding to use. Must be one of
                "add", "search", or "update". Defaults to None.

        Returns:
            list: A list of embedding vectors in input order.
        """
        texts_list = list(texts)
        if not texts_list:
            return []
        return [self.embed(text, memory_action) for text in texts_list]

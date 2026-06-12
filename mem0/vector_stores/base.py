from abc import ABC, abstractmethod


class VectorStoreBase(ABC):
    @abstractmethod
    def create_col(self, name, vector_size, distance):
        """Create a new collection."""
        pass

    @abstractmethod
    def insert(self, vectors, payloads=None, ids=None):
        """Insert vectors into a collection."""
        pass

    @abstractmethod
    def search(self, query, vectors, limit=5, filters=None):
        """Search for similar vectors."""
        pass

    @abstractmethod
    def delete(self, vector_id):
        """Delete a vector by ID."""
        pass

    @abstractmethod
    def update(self, vector_id, vector=None, payload=None):
        """Update a vector and its payload."""
        pass

    @abstractmethod
    def get(self, vector_id):
        """Retrieve a vector by ID."""
        pass

    @abstractmethod
    def list_cols(self):
        """List all collections."""
        pass

    @abstractmethod
    def delete_col(self):
        """Delete a collection."""
        pass

    @abstractmethod
    def col_info(self):
        """Get information about a collection."""
        pass

    @abstractmethod
    def list(self, filters=None, limit=None):
        """List all memories."""
        pass

    @abstractmethod
    def reset(self):
        """Reset by delete the collection and recreate it."""
        pass

    def iter_all(self, filters=None, page_size=512, with_vectors=False, consistent=False, timeout_seconds=5.0):
        """Default fallback: single list() call. Override in backends that support pagination."""
        import logging as _logging
        results = self.list(filters=filters, limit=page_size)
        flat = results[0] if isinstance(results, tuple) else results
        if len(flat) >= page_size:
            _logging.getLogger(__name__).warning(
                "iter_all fallback: result count (%d) hit page_size (%d); may be truncated.",
                len(flat), page_size,
            )
        yield from flat

    def iter_ids(self, filters=None, page_size=1024):
        """Iterate over all point IDs matching filters. Override for efficient backends."""
        raise NotImplementedError("iter_ids is not implemented for this vector store")

    def iter_id_versions(self, filters=None, page_size=1024):
        """Iterate over (id, version_int) pairs matching filters. Override for efficient backends."""
        raise NotImplementedError("iter_id_versions is not implemented for this vector store")

    def count(self, filters=None, exact=False):
        """Return count of points matching filters. Override for efficient backends."""
        raise NotImplementedError("count is not implemented for this vector store")

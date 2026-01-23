import logging
from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Configure a standard logger
logger = logging.getLogger(__name__)

class LegalContentRetriever:
    """
    Production-grade retriever for querying legal documents by hierarchy.
    Supports both synchronous and asynchronous execution.
    """

    def __init__(
        self, 
        vectorstore: FAISS, 
        metadata_keys: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the retriever.
        
        Args:
            vectorstore: The LangChain vector store instance.
            metadata_keys: Optional mapping to override default metadata field names.
                           Default: {'section': 'section', 'chapter': 'chapter', 'article': 'article'}
        """
        self.vectorstore = vectorstore
        self.keys = {
            "section": "section",
            "chapter": "chapter", 
            "article": "article",
            **(metadata_keys or {})
        }

    async def a_retrieve_by_section(self, section_id: str, k: int = 20) -> List[Document]:
        """Async version: Retrieve articles by section number."""
        return await self._a_retrieve_with_filter({self.keys["section"]: str(section_id)}, k)

    def retrieve_by_section(self, section_id: str, k: int = 20) -> List[Document]:
        """Sync version: Retrieve articles by section number."""
        return self._retrieve_with_filter({self.keys["section"]: str(section_id)}, k)

    def retrieve_by_chapter(self, chapter_id: str, k: int = 30) -> List[Document]:
        """Retrieve articles by chapter, handling Roman/Arabic numeral mismatches."""
        # 1. Try exact match
        docs = self._retrieve_with_filter({self.keys["chapter"]: str(chapter_id)}, k)
        
        # 2. If empty, try alternative format (e.g., '3' <-> 'III')
        if not docs:
            alt_id = self._normalize_chapter_format(str(chapter_id))
            if alt_id != str(chapter_id):
                logger.info(f"Retrying Chapter '{chapter_id}' as '{alt_id}'")
                docs = self._retrieve_with_filter({self.keys["chapter"]: alt_id}, k)
        
        return docs

    def _retrieve_with_filter(self, filters: Dict[str, str], k: int) -> List[Document]:
        """
        Internal: Executes the filtered search against the vector store.
        """
        try:
            # Metadata filtering is the Industry Standard for exact hierarchical retrieval
            docs = self.vectorstore.similarity_search(
                query="legal provision",  # Generic query required by API, result dominated by filter
                k=k,
                filter=filters
            )
            return self._deduplicate_and_sort(docs)
        except Exception as e:
            logger.error(f"Error retrieving documents with filter {filters}: {str(e)}", exc_info=True)
            return []

    async def _a_retrieve_with_filter(self, filters: Dict[str, str], k: int) -> List[Document]:
        """
        Internal: Async execution for high-concurrency environments.
        """
        try:
            # Ensure your vectorstore supports a_similarity_search (FAISS/Chroma/pgvector usually do)
            docs = await self.vectorstore.asimilarity_search(
                query="legal provision",
                k=k,
                filter=filters
            )
            return self._deduplicate_and_sort(docs)
        except Exception as e:
            logger.error(f"Async retrieval failed for {filters}: {str(e)}", exc_info=True)
            return []

    def _deduplicate_and_sort(self, docs: List[Document]) -> List[Document]:
        """
        Removes duplicate articles and sorts them numerically.
        """
        unique_map = {}
        art_key = self.keys["article"]

        for doc in docs:
            # Use article number as the unique key
            art_num = doc.metadata.get(art_key)
            if art_num and art_num not in unique_map:
                unique_map[art_num] = doc
        
        # Sort logic: Try to convert to int, fallback to 9999 if fails
        def sort_key(d):
            val = d.metadata.get(art_key, "")
            return int(val) if str(val).isdigit() else 9999

        return sorted(unique_map.values(), key=sort_key)

    @staticmethod
    def _normalize_chapter_format(value: str) -> str:
        """
        Helper to swap between Roman and Arabic numerals.
        Kept static and pure for testability.
        """
        # Simple lookup for common legal chapters (1-20)
        roman_map = {
            '1': 'I', '2': 'II', '3': 'III', '4': 'IV', '5': 'V', 
            '6': 'VI', '7': 'VII', '8': 'VIII', '9': 'IX', '10': 'X',
            '11': 'XI', '12': 'XII', '13': 'XIII', '14': 'XIV', '15': 'XV'
        }
        inv_map = {v: k for k, v in roman_map.items()}
        
        val_upper = value.upper()
        if value in roman_map: return roman_map[value]
        if val_upper in inv_map: return inv_map[val_upper]
        return value
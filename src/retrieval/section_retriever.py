import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.logger import get_logger

logger = get_logger("SectionRetriever")

class SectionRetriever:
    """
    Retrieves all articles within a specific section or chapter.
    
    CRITICAL FOR:
    - "What is in Section 4?"
    - "Where does Chapter III start?"
    - "Show me all articles in Chapter 2"
    """
    
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore

    def retrieve_by_section(self, section_id: str, k: int = 15) -> List[Document]:
        """
        Fetches documents tagged with specific section metadata.
        Args:
            section_id: The section number (e.g., "4")
            k: Max docs to return (default 15 to cover most sections)
        """
        logger.info(f"Retrieving Section {section_id}")
        return self._retrieve_and_sort({"section": str(section_id)}, k)

    def retrieve_by_chapter(self, chapter_id: str, k: int = 20) -> List[Document]:
        """
        Fetches documents tagged with specific chapter metadata.
        Handles Roman/Arabic conversion (e.g., '3' -> 'III').
        """
        logger.info(f"Retrieving Chapter {chapter_id}")
        
        # 1. Try exact match (e.g., user asks for "III", DB has "III")
        docs = self._retrieve_and_sort({"chapter": str(chapter_id)}, k)
        
        # 2. If no results, try swapping Roman/Arabic (e.g., user asks "3", DB has "III")
        if not docs:
            alt_id = self._swap_numeral(str(chapter_id))
            if alt_id != str(chapter_id):
                logger.info(f"Retrying Chapter '{chapter_id}' as '{alt_id}'")
                docs = self._retrieve_and_sort({"chapter": alt_id}, k)
            
        return docs

    def _retrieve_and_sort(self, filters: dict, k: int) -> List[Document]:
        """
        Internal Method:
        1. Query VectorDB with Metadata Filter (Exact Match).
        2. Sort results by Article Number (Ascending).
        """
        try:
            # We use a generic query ("legal provision") because we rely 100% on the filter.
            # k=500 fetches a large batch to ensure we capture the whole chapter/section
            # before we sort and slice it.
            docs = self.vectorstore.similarity_search(
                "legal provision", 
                k=500, 
                filter=filters
            )
            
            if not docs:
                return []

            # --- SORTING LOGIC ---
            # Sort by Article Number so the "start" of the chapter is first.
            def get_sort_key(doc):
                try:
                    # extract "15" from metadata "article": "15"
                    val = doc.metadata.get("article", "9999")
                    return int(val) if str(val).isdigit() else 9999
                except:
                    return 9999

            sorted_docs = sorted(docs, key=get_sort_key)
            
            # Remove duplicates (keep only the first occurrence of each article)
            unique_docs = []
            seen_articles = set()
            for doc in sorted_docs:
                art = doc.metadata.get("article")
                if art and art not in seen_articles:
                    seen_articles.add(art)
                    unique_docs.append(doc)
            
            # Return top k
            return unique_docs[:k]
            
        except Exception as e:
            logger.error(f"Section retrieval failed for {filters}: {e}")
            return []

    def _swap_numeral(self, val: str) -> str:
        """
        Helper to swap between Arabic ('3') and Roman ('III') numerals.
        Supports common chapters I through XX.
        """
        roman_map = {
            '1': 'I', '2': 'II', '3': 'III', '4': 'IV', '5': 'V', 
            '6': 'VI', '7': 'VII', '8': 'VIII', '9': 'IX', '10': 'X',
            '11': 'XI', '12': 'XII', '13': 'XIII', '14': 'XIV', '15': 'XV',
            '16': 'XVI', '17': 'XVII', '18': 'XVIII', '19': 'XIX', '20': 'XX'
        }
        
        # Create reverse map (III -> 3)
        inv_map = {v: k for k, v in roman_map.items()}
        
        val_upper = val.upper()
        
        # Check Arabic -> Roman
        if val in roman_map: 
            return roman_map[val]
            
        # Check Roman -> Arabic
        if val_upper in inv_map: 
            return inv_map[val_upper]
            
        return val
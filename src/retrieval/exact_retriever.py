from typing import List, Optional, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.retrieval.query_analyzer import QueryAnalysis, QueryType
from src.logger import get_logger

logger = get_logger("ExactRetriever")

class ExactRetriever:
    """
    Retrieves documents using exact metadata matching.
    Used for queries like "Article 15.1.a" or "Recital 42" where we know exactly what to retrieve.
    """
    
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore
    
    def retrieve(self, analysis: QueryAnalysis, k: int = 5) -> List[Document]:
        """
        Retrieve documents using metadata filtering.
        
        Args:
            analysis: QueryAnalysis object with extracted article/recital/subsection/point
            k: Number of documents to retrieve
        
        Returns:
            List of matching documents
        """
        try:
            # Build metadata filter
            metadata_filter = self._build_metadata_filter(analysis)
            
            if not metadata_filter:
                logger.warning("No metadata filter could be built")
                return []
            
            logger.info(f"Exact retrieval with filter: {metadata_filter}")
            
            # Get all documents (FAISS specific workaround for metadata filtering)
            # In a production DB like Chroma/Pinecone, you would push the filter to the query
            all_docs = self._get_all_documents()
            
            # Filter documents
            filtered_docs = self._filter_documents(all_docs, metadata_filter)
            
            # Sort by specificity
            filtered_docs = self._sort_by_specificity(filtered_docs, analysis)
            
            logger.info(f"Found {len(filtered_docs)} matching documents")
            return filtered_docs[:k]
        
        except Exception as e:
            logger.error(f"Exact retrieval failed: {e}")
            return []
    
    def _build_metadata_filter(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Build metadata filter from query analysis"""
        filter_dict = {}
        
        # Handle Recitals (New)
        if analysis.recital:
            filter_dict["recital"] = analysis.recital
            # Recitals are distinct from articles, so we return immediately
            return filter_dict
        
        # Handle Articles
        if analysis.article:
            filter_dict["article"] = analysis.article
        
        if analysis.subsection:
            filter_dict["subsection"] = analysis.subsection
        
        if analysis.point:
            filter_dict["point"] = analysis.point
        
        return filter_dict
    
    def _get_all_documents(self) -> List[Document]:
        """Get all documents from vectorstore"""
        try:
            # Fetch a large number of docs to ensure we catch the metadata match
            # Note: This is a limitation of FAISS in-memory. 
            # Ideally, use vectorstore.get() if the underlying store supports it.
            docs = self.vectorstore.similarity_search("article recital", k=2000)
            return docs
        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return []
    
    def _filter_documents(self, documents: List[Document], filter_dict: Dict[str, Any]) -> List[Document]:
        """Filter documents by metadata"""
        filtered = []
        
        for doc in documents:
            match = True
            for key, value in filter_dict.items():
                # Compare as strings to avoid type mismatches (int vs str)
                doc_value = doc.metadata.get(key)
                if str(doc_value) != str(value):
                    match = False
                    break
            
            if match:
                filtered.append(doc)
        
        return filtered
    
    def _sort_by_specificity(self, documents: List[Document], analysis: QueryAnalysis) -> List[Document]:
        """Sort documents by specificity level to return the most precise chunk first"""
        def specificity_score(doc: Document) -> int:
            score = 0
            if doc.metadata.get("point"):
                score += 100
            if doc.metadata.get("subsection"):
                score += 10
            if doc.metadata.get("article"):
                score += 1
            if doc.metadata.get("recital"):
                score += 1
            return score
        
        target_level = None
        if analysis.recital:
            target_level = "recital"
        elif analysis.point:
            target_level = "point"
        elif analysis.subsection:
            target_level = "subsection"
        elif analysis.article:
            target_level = "article"
        
        def custom_sort(doc: Document) -> Tuple[int, int]:
            # 1. Prioritize documents that match the target level exactly
            level_match = 0
            doc_level = doc.metadata.get("level")
            
            if target_level and str(doc_level) == target_level:
                level_match = 1
            
            # 2. Then sort by how specific the chunk is (Point > Subsection > Article)
            specificity = specificity_score(doc)
            
            # Python sorts ascending, so we use negative values for descending sort
            return (-level_match, -specificity)
        
        return sorted(documents, key=custom_sort)
    
    def retrieve_with_context(self, analysis: QueryAnalysis, k: int = 3) -> List[Document]:
        """Retrieve exact matches and add parent context if available"""
        results = []
        
        # 1. Get exact match (e.g., Point (a))
        exact_matches = self.retrieve(analysis, k=1)
        results.extend(exact_matches)
        
        # Stop here if it's a Recital (no parents needed)
        if analysis.recital:
            return results

        # 2. Get parent contexts
        # If we found a point, try to get the parent subsection
        if analysis.point and analysis.subsection:
            parent_analysis = QueryAnalysis(
                query_type=QueryType.EXACT_REFERENCE,
                original_query=analysis.original_query,
                article=analysis.article,
                subsection=analysis.subsection
            )
            parent_docs = self.retrieve(parent_analysis, k=1)
            results.extend(parent_docs)
        
        # If we found a subsection, try to get the parent article
        if analysis.subsection:
            parent_analysis = QueryAnalysis(
                query_type=QueryType.ARTICLE_LOOKUP,
                original_query=analysis.original_query,
                article=analysis.article
            )
            parent_docs = self.retrieve(parent_analysis, k=1)
            results.extend(parent_docs)
        
        # 3. Deduplicate
        seen = set()
        unique_results = []
        for doc in results:
            doc_id = doc.metadata.get("chunk_id")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(doc)
        
        return unique_results[:k]
"""
Smart Hybrid Retriever
Routes queries to the best retrieval strategy based on LLM analysis.

Strategies:
1. Exact Lookup: Use metadata filtering for specific references
2. Range Query: Find all articles in a chapter/section
3. Semantic Search: Vector similarity for conceptual questions
"""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.retrieval.llm_query_analyzer import LLMQueryAnalyzer, QueryAnalysis, QueryIntent
from src.config import Config
from src.logger import get_logger

logger = get_logger("SmartRetriever")


class SmartRetriever:
    """
    Intelligent retriever that uses LLM analysis to choose the best strategy.
    """
    
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore
        self.analyzer = LLMQueryAnalyzer()
    
    def retrieve(self, query: str, k: int = None) -> Tuple[List[Document], QueryAnalysis]:
        """
        Main retrieval entry point.
        
        Returns:
            (documents, analysis) tuple
        """
        k = k or Config.RETRIEVER_K_FINAL
        
        try:
            # Step 1: Analyze query with LLM
            analysis = self.analyzer.analyze(query)
            
            # Step 2: Route to appropriate strategy
            if analysis.intent == QueryIntent.EXACT_LOOKUP:
                docs = self._exact_lookup(analysis, k)
            
            elif analysis.intent == QueryIntent.RANGE_QUERY:
                docs = self._range_query(analysis, k)
            
            elif analysis.intent in [QueryIntent.CONCEPTUAL, QueryIntent.COMPARISON]:
                docs = self._semantic_search(query, analysis, k)
            
            else:  # GENERAL
                docs = self._semantic_search(query, analysis, k)
            
            # Fallback if no results
            if not docs:
                logger.warning("Primary retrieval returned no results, trying semantic fallback")
                docs = self._semantic_search(query, analysis, k)
            
            logger.info(f"Retrieved {len(docs)} documents for query: '{query}'")
            return docs, analysis
        
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            
            # Emergency fallback
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs, analysis
    
    def _exact_lookup(self, analysis: QueryAnalysis, k: int) -> List[Document]:
        """
        Exact metadata lookup for specific references.
        Example: "What is Article 15.1.a?"
        """
        logger.info(f"Exact lookup: {analysis.to_filter_dict()}")
        
        # Get all documents (FAISS limitation workaround)
        all_docs = self.vectorstore.similarity_search("", k=5000)
        
        # Filter by metadata
        filter_dict = analysis.to_filter_dict()
        filtered_docs = []
        
        for doc in all_docs:
            match = True
            for key, value in filter_dict.items():
                if str(doc.metadata.get(key)) != str(value):
                    match = False
                    break
            
            if match:
                filtered_docs.append(doc)
        
        # Sort by specificity (subpoint > point > article)
        filtered_docs = self._sort_by_specificity(filtered_docs)
        
        # Add parent context if available
        if filtered_docs and analysis.subpoint:
            # Also include the parent point
            parent_filter = {**filter_dict}
            parent_filter.pop("subpoint", None)
            
            for doc in all_docs:
                if doc in filtered_docs:
                    continue
                match = True
                for key, value in parent_filter.items():
                    if str(doc.metadata.get(key)) != str(value):
                        match = False
                        break
                if match:
                    filtered_docs.append(doc)
        
        logger.info(f"Exact lookup found {len(filtered_docs)} matches")
        return filtered_docs[:k]
    
    def _range_query(self, analysis: QueryAnalysis, k: int) -> List[Document]:
        """
        Range query for finding all articles in a chapter/section.
        Example: "What articles are in Chapter 2 Section 3?"
        """
        logger.info(f"Range query: Chapter {analysis.chapter}, Section {analysis.section}")
        
        # Get all documents
        all_docs = self.vectorstore.similarity_search("", k=5000)
        
        # Filter by chapter/section
        filtered_docs = []
        for doc in all_docs:
            match = True
            
            if analysis.chapter and str(doc.metadata.get("chapter")) != str(analysis.chapter):
                match = False
            
            if analysis.section and str(doc.metadata.get("section")) != str(analysis.section):
                match = False
            
            # Only include article-level chunks
            if match and doc.metadata.get("level") == "article":
                filtered_docs.append(doc)
        
        # Sort by article number
        filtered_docs.sort(key=lambda d: int(d.metadata.get("article", 0) or 0))
        
        logger.info(f"Range query found {len(filtered_docs)} articles")
        return filtered_docs[:k]
    
    def _semantic_search(self, query: str, analysis: QueryAnalysis, k: int) -> List[Document]:
        """
        Semantic search using vector similarity.
        Example: "What are the consent requirements?"
        """
        logger.info("Performing semantic search")
        
        # Use more results for reranking
        docs = self.vectorstore.similarity_search(query, k=k * 3)
        
        # If analysis found specific references, boost those
        if analysis.has_exact_reference():
            filter_dict = analysis.to_filter_dict()
            
            scored_docs = []
            for doc in docs:
                score = 0
                
                # Boost matching metadata
                for key, value in filter_dict.items():
                    if str(doc.metadata.get(key)) == str(value):
                        score += 10
                
                scored_docs.append((score, doc))
            
            # Sort by score
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            docs = [doc for _, doc in scored_docs]
        
        return docs[:k]
    
    def _sort_by_specificity(self, docs: List[Document]) -> List[Document]:
        """Sort documents by how specific they are"""
        def specificity_score(doc: Document) -> int:
            score = 0
            if doc.metadata.get("subpoint"):
                score += 1000
            if doc.metadata.get("point"):
                score += 100
            if doc.metadata.get("article"):
                score += 10
            if doc.metadata.get("section"):
                score += 1
            return score
        
        return sorted(docs, key=specificity_score, reverse=True)

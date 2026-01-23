from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.retrieval.query_analyzer import QueryAnalyzer, QueryAnalysis, QueryType
from src.retrieval.exact_retriever import ExactRetriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.config import Config
from src.logger import get_logger

logger = get_logger("HybridRetriever")

class HybridRetriever:
    """
    Intelligent retriever that routes between exact and semantic search.
    
    Strategy:
    1. Analyze query to determine type (Recital, Article, Concept, etc.)
    2. Route to appropriate retriever
    3. Apply fallback if needed
    4. Combine results if beneficial
    """
    
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore
        self.query_analyzer = QueryAnalyzer()
        self.exact_retriever = ExactRetriever(vectorstore)
        self.semantic_retriever = SemanticRetriever(vectorstore)
    
    def retrieve(self, query: str, k: int = None) -> Tuple[List[Document], QueryAnalysis]:
        """
        Main retrieval method that intelligently routes the query.
        
        Args:
            query: User query string
            k: Number of documents to retrieve
        
        Returns:
            Tuple of (documents, query_analysis)
        """
        k = k or Config.RETRIEVER_K_FINAL
        
        try:
            # Step 1: Analyze query
            analysis = self.query_analyzer.analyze(query)
            logger.info(f"Query classified as: {analysis.query_type.value} (confidence: {analysis.confidence})")
            
            # Step 2: Route based on query type
            
            # --- NEW: Handle Recitals ---
            if analysis.query_type == QueryType.RECITAL_LOOKUP:
                docs = self._handle_recital_lookup(analysis, k)
            
            # --- Existing Handlers ---
            elif analysis.query_type == QueryType.EXACT_REFERENCE:
                docs = self._handle_exact_reference(analysis, k)
            
            elif analysis.query_type == QueryType.ARTICLE_LOOKUP:
                docs = self._handle_article_lookup(analysis, k)
            
            elif analysis.query_type == QueryType.CONCEPTUAL:
                docs = self._handle_conceptual(query, analysis, k)
            
            elif analysis.query_type == QueryType.COMPARISON:
                docs = self._handle_comparison(query, analysis, k)
            
            else:  # GENERAL
                docs = self._handle_general(query, k)
            
            # Step 3: Apply fallback if no results
            if not docs:
                logger.warning("Primary retrieval returned no results, trying fallback...")
                docs = self._fallback_retrieval(query, k)
            
            logger.info(f"Final retrieval: {len(docs)} documents")
            return docs, analysis
        
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            # Fallback to basic semantic search
            docs = self.semantic_retriever.retrieve(query, k)
            # Return a generic analysis object so the pipeline doesn't break
            return docs, QueryAnalysis(query_type=QueryType.GENERAL, original_query=query)
    
    def _handle_recital_lookup(self, analysis: QueryAnalysis, k: int) -> List[Document]:
        """
        Handle queries like "What is Recital 42?"
        Strategy: Strict exact retrieval using 'recital' metadata.
        """
        logger.info(f"Using exact retrieval for Recital {analysis.recital}")
        
        # Use ExactRetriever which now supports 'recital' metadata filtering
        docs = self.exact_retriever.retrieve(analysis, k=k)
        
        if not docs:
            logger.warning(f"Recital {analysis.recital} not found in metadata, falling back to semantic")
            # If the parser missed it (rare), semantic search might find the text "Recital (42)"
            docs = self.semantic_retriever.retrieve(analysis.original_query, k=k)
        
        return docs

    def _handle_exact_reference(self, analysis: QueryAnalysis, k: int) -> List[Document]:
        """
        Handle queries like "What is Article 15.1.a?"
        Strategy: Exact metadata lookup with context (getting parent Article/Subsection).
        """
        logger.info(f"Using exact retrieval for Article {analysis.article}.{analysis.subsection}.{analysis.point}")
        docs = self.exact_retriever.retrieve_with_context(analysis, k=k)
        
        if not docs:
            logger.warning("Exact retrieval failed, falling back to semantic")
            docs = self.semantic_retriever.retrieve(analysis.original_query, k=k)
        
        return docs
    
    def _handle_article_lookup(self, analysis: QueryAnalysis, k: int) -> List[Document]:
        """
        Handle queries like "Show me Article 15"
        Strategy: Exact article match (all chunks belonging to that article).
        """
        logger.info(f"Using article lookup for Article {analysis.article}")
        docs = self.exact_retriever.retrieve(analysis, k=k)
        
        if not docs:
            logger.warning("Article lookup failed, falling back to semantic")
            docs = self.semantic_retriever.retrieve(analysis.original_query, k=k)
        
        return docs
    
    def _handle_conceptual(self, query: str, analysis: QueryAnalysis, k: int) -> List[Document]:
        """
        Handle queries like "What are the consent requirements?" or "What does Article 6 say about lawfulness?"
        Strategy: Semantic search, optionally boosted by mentioned article.
        """
        logger.info("Using semantic retrieval for conceptual query")
        
        if analysis.article:
            # Query mentions an article, boost results from that article
            logger.info(f"Boosting results from Article {analysis.article}")
            docs = self.semantic_retriever.retrieve_with_metadata_boost(
                query, 
                article=analysis.article, 
                k=k
            )
        else:
            # Pure semantic search
            docs = self.semantic_retriever.retrieve(query, k=k)
        
        return docs
    
    def _handle_comparison(self, query: str, analysis: QueryAnalysis, k: int) -> List[Document]:
        """
        Handle queries like "What's the difference between Article 6 and 7?"
        Strategy: Get results from both articles if mentioned, mixed with semantic search.
        """
        logger.info("Using comparison retrieval")
        
        # Use semantic search to get relevant chunks first (broad context)
        docs = self.semantic_retriever.retrieve(query, k=k*2)
        
        # If specific articles are mentioned, ensure we have chunks from both
        if len(analysis.extracted_concepts) >= 2:
            article1 = analysis.extracted_concepts[0]
            article2 = analysis.extracted_concepts[1]
            
            # Check if we naturally found results from both articles
            has_article1 = any(doc.metadata.get("article") == article1 for doc in docs)
            has_article2 = any(doc.metadata.get("article") == article2 for doc in docs)
            
            # If missing, fetch explicitly using ExactRetriever
            if not has_article1:
                logger.info(f"Explicitly fetching Article {article1}")
                analysis1 = QueryAnalysis(
                    query_type=QueryType.ARTICLE_LOOKUP,
                    original_query=query,
                    article=article1
                )
                docs.extend(self.exact_retriever.retrieve(analysis1, k=2))
            
            if not has_article2:
                logger.info(f"Explicitly fetching Article {article2}")
                analysis2 = QueryAnalysis(
                    query_type=QueryType.ARTICLE_LOOKUP,
                    original_query=query,
                    article=article2
                )
                docs.extend(self.exact_retriever.retrieve(analysis2, k=2))
        
        # Deduplicate and limit
        docs = self._deduplicate_documents(docs)
        return docs[:k]
    
    def _handle_general(self, query: str, k: int) -> List[Document]:
        """
        Handle general queries.
        Strategy: Pure semantic search.
        """
        logger.info("Using semantic retrieval for general query")
        return self.semantic_retriever.retrieve(query, k=k)
    
    def _fallback_retrieval(self, query: str, k: int) -> List[Document]:
        """
        Fallback strategy when primary retrieval fails.
        """
        logger.info("Executing fallback retrieval")
        return self.semantic_retriever.retrieve(query, k=k)
    
    def _deduplicate_documents(self, docs: List[Document]) -> List[Document]:
        """Remove duplicate documents based on chunk_id"""
        seen = set()
        unique_docs = []
        
        for doc in docs:
            # Fallback to content hash if chunk_id is missing
            chunk_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_docs.append(doc)
        
        return unique_docs
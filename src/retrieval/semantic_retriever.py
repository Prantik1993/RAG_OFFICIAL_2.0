from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from src.config import Config
from src.logger import get_logger

logger = get_logger("SemanticRetriever")

class SemanticRetriever:
    """
    Retrieves documents using semantic similarity search with reranking.
    Used for conceptual queries like "What are consent requirements?"
    """
    
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore
        self._setup_reranker()
    
    def _setup_reranker(self):
        """Setup Flashrank reranker"""
        try:
            # Fix for pydantic v2 compatibility
            try:
                FlashrankRerank.model_rebuild()
            except Exception:
                pass
            
            logger.info("Initializing Flashrank reranker...")
            self.compressor = FlashrankRerank(
                model=Config.RERANKER_MODEL,
                top_n=Config.RETRIEVER_K_RERANKED
            )
            
            # Base retriever fetches broad results
            base_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": Config.RETRIEVER_K_BASE}
            )
            
            # Compression retriever reranks them
            self.retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=base_retriever
            )
            
            logger.info("Semantic retriever initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to setup reranker: {e}")
            # Fallback to basic retriever without reranking
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": Config.RETRIEVER_K_FINAL}
            )
    
    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """
        Retrieve documents using semantic search + reranking.
        
        Args:
            query: User query string
            k: Number of final documents (if None, uses config default)
        
        Returns:
            List of most relevant documents
        """
        try:
            k = k or Config.RETRIEVER_K_FINAL
            logger.info(f"Semantic retrieval for: '{query[:50]}...'")
            
            # Get reranked results
            docs = self.retriever.get_relevant_documents(query)
            
            # Limit to k results
            docs = docs[:k]
            
            logger.info(f"Retrieved {len(docs)} documents")
            return docs
        
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            return []
    
    def retrieve_with_metadata_boost(self, query: str, article: str = None, k: int = None) -> List[Document]:
        """
        Semantic retrieval with metadata boosting.
        If article is specified, boost results from that article.
        
        Useful for queries like "What does Article 15 say about consent?"
        where we want semantic search but biased toward Article 15.
        """
        try:
            k = k or Config.RETRIEVER_K_FINAL
            
            # Get more results initially
            docs = self.retrieve(query, k=k*3)
            
            if not article:
                return docs[:k]
            
            # Score and sort by article match
            scored_docs = []
            for doc in docs:
                score = 0
                if doc.metadata.get("article") == article:
                    score = 10  # High boost for matching article
                elif doc.metadata.get("article"):
                    score = 1   # Small boost for any article mention
                
                scored_docs.append((score, doc))
            
            # Sort by score (descending) and take top k
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            result = [doc for _, doc in scored_docs[:k]]
            
            logger.info(f"Retrieved {len(result)} documents with metadata boost (article={article})")
            return result
        
        except Exception as e:
            logger.error(f"Metadata-boosted retrieval failed: {e}")
            return []
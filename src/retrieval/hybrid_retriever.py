from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import Config
from src.logger import get_logger
from src.retrieval.exact_retriever import ExactRetriever
from src.retrieval.query_analyzer import QueryAnalysis, QueryAnalyzer, QueryType
from src.retrieval.semantic_retriever import SemanticRetriever

logger = get_logger("HybridRetriever")


class HybridRetriever:
    def __init__(self, vectorstore: FAISS):
        self.query_analyzer = QueryAnalyzer()
        self.exact_retriever = ExactRetriever(vectorstore)
        self.semantic_retriever = SemanticRetriever(vectorstore)

    def retrieve(self, query: str, k: int = None) -> Tuple[List[Document], QueryAnalysis]:
        k = k or Config.RETRIEVER_K_FINAL
        analysis = self.query_analyzer.analyze(query)

        exact_types = {
            QueryType.RECITAL_LOOKUP,
            QueryType.EXACT_REFERENCE,
            QueryType.ARTICLE_LOOKUP,
            QueryType.SECTION_LOOKUP,
            QueryType.CHAPTER_LOOKUP,
            QueryType.CHAPTER_SECTION_LOOKUP,
        }

        if analysis.query_type in exact_types:
            docs = self.exact_retriever.retrieve_with_context(analysis, k=k)
        elif analysis.query_type in {QueryType.CONCEPTUAL, QueryType.COMPARISON}:
            docs = self.semantic_retriever.retrieve_with_metadata_boost(query, article=analysis.article, k=k)
        else:
            docs = self.semantic_retriever.retrieve(query, k=k)

        if not docs:
            logger.warning("Primary retrieval empty. Falling back to semantic search")
            docs = self.semantic_retriever.retrieve(query, k=k)

        return docs, analysis

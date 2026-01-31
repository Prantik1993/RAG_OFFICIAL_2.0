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
    Industry-standard Hybrid Retriever.

    Responsibilities:
    1. Analyze query intent
    2. Route to exact OR semantic retrieval
    3. Apply fallback safely

    NON-responsibilities:
    - Manual chapter/section traversal
    - Legal hierarchy reconstruction
    - Vector DB workarounds scattered everywhere
    """

    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore
        self.query_analyzer = QueryAnalyzer()
        self.exact_retriever = ExactRetriever(vectorstore)
        self.semantic_retriever = SemanticRetriever(vectorstore)

    def retrieve(self, query: str, k: int = None) -> Tuple[List[Document], QueryAnalysis]:
        """
        Main retrieval entrypoint.
        """
        k = k or Config.RETRIEVER_K_FINAL

        try:
            analysis = self.query_analyzer.analyze(query)

            logger.info(
                f"Query routed as {analysis.query_type.value} "
                f"(confidence={analysis.confidence})"
            )

            # ---------- ROUTING ----------
            if analysis.query_type == QueryType.RECITAL_LOOKUP:
                docs = self._handle_recital(analysis, k)

            elif analysis.query_type in (
                QueryType.EXACT_REFERENCE,
                QueryType.ARTICLE_LOOKUP,
            ):
                docs = self._handle_exact(analysis, k)

            elif analysis.query_type in (
                QueryType.CONCEPTUAL,
                QueryType.COMPARISON,
            ):
                docs = self._handle_semantic(query, analysis, k)

            else:  # GENERAL / fallback
                docs = self._handle_general(query, k)

            # ---------- FALLBACK ----------
            if not docs:
                logger.warning("Primary retrieval empty → semantic fallback")
                docs = self.semantic_retriever.retrieve(query, k)

            return docs, analysis

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            docs = self.semantic_retriever.retrieve(query, k)
            return docs, QueryAnalysis(
                query_type=QueryType.GENERAL,
                original_query=query,
                confidence=0.3,
            )

    # ------------------------------------------------------------------
    # HANDLERS
    # ------------------------------------------------------------------

    def _handle_recital(self, analysis: QueryAnalysis, k: int) -> List[Document]:
        """
        Strict exact lookup for Recitals.
        """
        logger.info(f"Exact retrieval for Recital {analysis.recital}")

        docs = self.exact_retriever.retrieve(analysis, k=k)

        if not docs:
            logger.warning("Recital exact match failed → semantic fallback")
            docs = self.semantic_retriever.retrieve(
                analysis.original_query, k=k
            )

        return docs

    def _handle_exact(self, analysis: QueryAnalysis, k: int) -> List[Document]:
        """
        Handles:
        - Article 15
        - Article 15.1
        - Article 15.1.a
        """
        logger.info("Exact article retrieval")

        docs = self.exact_retriever.retrieve_with_context(analysis, k=k)

        if not docs:
            logger.warning("Exact retrieval failed → semantic fallback")
            docs = self.semantic_retriever.retrieve(
                analysis.original_query, k=k
            )

        return docs

    def _handle_semantic(
        self, query: str, analysis: QueryAnalysis, k: int
    ) -> List[Document]:
        """
        Semantic retrieval with optional metadata boost.
        """
        logger.info("Semantic retrieval")

        if analysis.article:
            logger.info(f"Boosting semantic results for Article {analysis.article}")
            return self.semantic_retriever.retrieve_with_metadata_boost(
                query,
                article=analysis.article,
                k=k,
            )

        return self.semantic_retriever.retrieve(query, k=k)

    def _handle_general(self, query: str, k: int) -> List[Document]:
        """
        Generic fallback retrieval.
        """
        logger.info("General semantic retrieval")
        return self.semantic_retriever.retrieve(query, k=k)

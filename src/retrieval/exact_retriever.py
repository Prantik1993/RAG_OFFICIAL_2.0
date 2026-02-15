from typing import Any, Dict, List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.logger import get_logger
from src.retrieval.query_analyzer import QueryAnalysis, QueryType

logger = get_logger("ExactRetriever")


class ExactRetriever:
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore

    def retrieve(self, analysis: QueryAnalysis, k: int = 5) -> List[Document]:
        metadata_filter = self._build_metadata_filter(analysis)
        if not metadata_filter:
            return []

        all_docs = self._get_all_documents()
        filtered = self._filter_documents(all_docs, metadata_filter)
        return self._sort_by_specificity(filtered, analysis)[:k]

    def _build_metadata_filter(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        if analysis.recital:
            return {"recital": analysis.recital}

        filter_dict: Dict[str, Any] = {}
        if analysis.chapter:
            filter_dict["chapter_num"] = analysis.chapter
        if analysis.section:
            filter_dict["section"] = analysis.section
        if analysis.article:
            filter_dict["article"] = analysis.article
        if analysis.subsection:
            filter_dict["subsection"] = analysis.subsection
        if analysis.point:
            filter_dict["point"] = analysis.point
        return filter_dict

    def _get_all_documents(self) -> List[Document]:
        try:
            return self.vectorstore.similarity_search("gdpr legal article recital section chapter", k=4000)
        except Exception as exc:
            logger.error(f"Failed to get documents: {exc}")
            return []

    def _filter_documents(self, documents: List[Document], filter_dict: Dict[str, Any]) -> List[Document]:
        filtered = []
        for doc in documents:
            if all(str(doc.metadata.get(key)) == str(value) for key, value in filter_dict.items()):
                filtered.append(doc)
        return filtered

    def _sort_by_specificity(self, documents: List[Document], analysis: QueryAnalysis) -> List[Document]:
        target_level = {
            QueryType.RECITAL_LOOKUP: "recital",
            QueryType.ARTICLE_LOOKUP: "article",
            QueryType.SECTION_LOOKUP: "article",
            QueryType.CHAPTER_LOOKUP: "article",
            QueryType.CHAPTER_SECTION_LOOKUP: "article",
            QueryType.EXACT_REFERENCE: "point" if analysis.point else "subsection" if analysis.subsection else "article",
        }.get(analysis.query_type, "article")

        def score(doc: Document) -> Tuple[int, int]:
            level = doc.metadata.get("level")
            specificity = 0
            if doc.metadata.get("chapter_num"):
                specificity += 1
            if doc.metadata.get("section"):
                specificity += 2
            if doc.metadata.get("article"):
                specificity += 4
            if doc.metadata.get("subsection"):
                specificity += 8
            if doc.metadata.get("point"):
                specificity += 16
            return (1 if level == target_level else 0, specificity)

        return sorted(documents, key=score, reverse=True)

    def retrieve_with_context(self, analysis: QueryAnalysis, k: int = 4) -> List[Document]:
        docs: List[Document] = []
        docs.extend(self.retrieve(analysis, k=1))

        if analysis.recital:
            return docs

        if analysis.point and analysis.subsection and analysis.article:
            docs.extend(
                self.retrieve(
                    QueryAnalysis(
                        query_type=QueryType.EXACT_REFERENCE,
                        original_query=analysis.original_query,
                        chapter=analysis.chapter,
                        section=analysis.section,
                        article=analysis.article,
                        subsection=analysis.subsection,
                    ),
                    k=1,
                )
            )

        if analysis.subsection and analysis.article:
            docs.extend(
                self.retrieve(
                    QueryAnalysis(
                        query_type=QueryType.ARTICLE_LOOKUP,
                        original_query=analysis.original_query,
                        chapter=analysis.chapter,
                        section=analysis.section,
                        article=analysis.article,
                    ),
                    k=1,
                )
            )

        seen = set()
        deduped: list[Document] = []
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                deduped.append(doc)

        return deduped[:k]

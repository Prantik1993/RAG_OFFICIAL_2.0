from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

class DocumentLevel(Enum):
    """Hierarchy levels in legal documents"""
    REGULATION = "regulation"
    RECITAL = "recital"  # Added for (1), (2) intro text
    CHAPTER = "chapter"
    SECTION = "section"
    ARTICLE = "article"
    SUBSECTION = "subsection"
    POINT = "point"

@dataclass
class LegalReference:
    """Represents a complete legal reference path"""
    recital: Optional[str] = None
    chapter: Optional[str] = None
    chapter_title: Optional[str] = None
    section: Optional[str] = None
    section_title: Optional[str] = None
    article: Optional[str] = None
    article_title: Optional[str] = None
    subsection: Optional[str] = None
    point: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recital": self.recital,
            "chapter": self.chapter,
            "chapter_title": self.chapter_title,
            "section": self.section,
            "section_title": self.section_title,
            "article": self.article,
            "article_title": self.article_title,
            "subsection": self.subsection,
            "point": self.point
        }
    
    def get_full_reference(self) -> str:
        """Get human-readable reference string"""
        if self.recital:
            return f"Regulation Point ({self.recital})"
            
        parts = []
        if self.chapter:
            parts.append(f"Chapter {self.chapter}")
        if self.section:
            parts.append(f"Section {self.section}")
        if self.article:
            ref = f"Article {self.article}"
            if self.subsection:
                ref += f"({self.subsection})"
            if self.point:
                ref += f"({self.point})"
            parts.append(ref)
        return " → ".join(parts) if parts else "Unknown"

@dataclass
class DocumentChunk:
    """Represents a processed document chunk with metadata"""
    content: str
    reference: LegalReference
    page: int
    chunk_id: str
    level: DocumentLevel
    parent_content: Optional[str] = None
    
    def to_langchain_document(self):
        """Convert to LangChain Document format"""
        from langchain_core.documents import Document
        
        metadata = {
            "page": self.page,
            "chunk_id": self.chunk_id,
            "level": self.level.value,
            "full_reference": self.reference.get_full_reference(),
            **self.reference.to_dict()
        }
        
        # Combine content with parent context if available for better embedding
        full_content = self.content
        if self.parent_content:
            full_content = f"Context: {self.parent_content}\n---\n{self.content}"
        
        return Document(page_content=full_content, metadata=metadata)

@dataclass
class ArticleStructure:
    """Represents a complete structure (Recital OR Article)"""
    id: str  # Number/ID
    title: str
    page: int
    full_text: str
    subsections: List[Dict[str, Any]] = field(default_factory=list)
    chapter: Optional[str] = None
    section: Optional[str] = None
    is_recital: bool = False
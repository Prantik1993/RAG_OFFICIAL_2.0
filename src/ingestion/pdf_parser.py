import re
from typing import List, Dict, Optional
from langchain_community.document_loaders import PyPDFLoader
from src.ingestion.document_structure import (
    LegalReference, DocumentChunk, DocumentLevel, ArticleStructure
)
from src.logger import get_logger
from src.exceptions import ParsingError

logger = get_logger("PDFParser")

class LegalDocumentParser:
    """
    State-machine parser for EU Regulations.
    Handles: Recitals -> Chapters -> Sections -> Articles
    """
    
    def __init__(self):
        # Regex patterns derived from provided images
        self.chapter_pattern = re.compile(r"^CHAPTER\s+([IVX\d]+)", re.IGNORECASE)
        self.section_pattern = re.compile(r"^Section\s+(\d+)", re.IGNORECASE)
        self.article_pattern = re.compile(r"^Article\s+(\d+)", re.IGNORECASE)
        self.recital_pattern = re.compile(r"^\((\d+)\)") 
        self.subsection_pattern = re.compile(r"^(\d+)\.\s") 
        self.point_pattern = re.compile(r"^\(([a-z])\)\s")

    def load_pdf(self, pdf_path: str) -> List[Dict]:
        try:
            logger.info(f"Loading PDF from: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            return loader.load()
        except Exception as e:
            raise ParsingError(f"PDF loading failed: {e}")
    
    def extract_document_structure(self, pages: List) -> List[ArticleStructure]:
        """
        Parses document using a 2-Phase State Machine:
        Phase 1: Recitals (Intro text starting with (1))
        Phase 2: Legislative (Articles, Chapters)
        """
        structures = []
        
        # Flatten all pages into a single stream of lines with page numbers
        all_lines = []
        for p in pages:
            lines = p.page_content.split('\n')
            for line in lines:
                clean_line = line.strip()
                if clean_line:
                    all_lines.append({"text": clean_line, "page": p.metadata.get("page", 0)})

        # State Variables
        current_chapter = None
        current_section = None
        current_struct = None  # The current Article or Recital being built
        in_recital_phase = True
        
        i = 0
        while i < len(all_lines):
            line_data = all_lines[i]
            text = line_data["text"]
            page = line_data["page"]
            
            # --- Global Structure Checks (Chapter/Section) ---
            
            # Detect Chapter (e.g., CHAPTER III)
            chap_match = self.chapter_pattern.match(text)
            if chap_match:
                in_recital_phase = False # Chapters definitely end recitals
                current_chapter = chap_match.group(1)
                i += 1
                continue

            # Detect Section (e.g., Section 2)
            sec_match = self.section_pattern.match(text)
            if sec_match:
                current_section = sec_match.group(1)
                i += 1
                continue

            # --- Phase 1: Recitals ---
            if in_recital_phase:
                # If we hit an Article, we exit recital phase immediately
                if self.article_pattern.match(text):
                    in_recital_phase = False
                    # Don't increment i, let Phase 2 handle this line
                else:
                    recital_match = self.recital_pattern.match(text)
                    if recital_match:
                        # Save previous structure
                        if current_struct:
                            structures.append(current_struct)
                        
                        # Start new Recital
                        rec_num = recital_match.group(1)
                        content = text[recital_match.end():].strip()
                        current_struct = ArticleStructure(
                            id=rec_num, title="Recital", page=page,
                            full_text=content, is_recital=True
                        )
                    elif current_struct and current_struct.is_recital:
                        # Continuation of previous recital
                        current_struct.full_text += " " + text
                    
                    if in_recital_phase:
                        i += 1
                        continue

            # --- Phase 2: Articles ---
            
            # Detect Article Start: "Article 1"
            art_match = self.article_pattern.match(text)
            if art_match:
                # Save previous structure
                if current_struct:
                    if not current_struct.is_recital:
                        # Parse subsections before saving
                        current_struct.subsections = self._parse_subsections(current_struct.full_text)
                    structures.append(current_struct)

                art_num = art_match.group(1)
                
                # LOOKAHEAD: Identify Title on next line
                # Logic: If next line is short and NOT a subsection (1.), it's likely the title.
                title = f"Article {art_num}" # Fallback
                next_idx = i + 1
                if next_idx < len(all_lines):
                    next_line = all_lines[next_idx]["text"]
                    if not self.subsection_pattern.match(next_line) and len(next_line) < 200:
                        title = next_line
                        i += 1 # Skip title line in main loop
                
                current_struct = ArticleStructure(
                    id=art_num, title=title, page=page, full_text="",
                    chapter=current_chapter, section=current_section, is_recital=False
                )
                i += 1
                continue
            
            # Accumulate Article Text
            if current_struct and not current_struct.is_recital:
                current_struct.full_text += "\n" + text
            
            i += 1

        # Save final structure
        if current_struct:
            if not current_struct.is_recital:
                current_struct.subsections = self._parse_subsections(current_struct.full_text)
            structures.append(current_struct)

        return structures

    def _parse_subsections(self, text: str) -> List[Dict]:
        """
        Parses nested structure: 1. -> (a) -> Text
        """
        subsections = []
        lines = text.split('\n')
        
        curr_sub_num = None
        curr_sub_text = []
        curr_points = []
        
        # Helper to close a subsection
        def save_subsection():
            if curr_sub_num:
                subsections.append({
                    "number": curr_sub_num,
                    "text": " ".join(curr_sub_text).strip(),
                    "points": list(curr_points)
                })

        for line in lines:
            line = line.strip()
            if not line: continue

            # Check 1. 
            sub_match = self.subsection_pattern.match(line)
            # Check (a)
            point_match = self.point_pattern.match(line)

            if sub_match:
                save_subsection()
                curr_sub_num = sub_match.group(1)
                curr_sub_text = [line[sub_match.end():].strip()]
                curr_points = []
                
            elif point_match:
                # If point appears before any numbered subsection, assume it belongs to "Intro"
                if not curr_sub_num:
                    curr_sub_num = "0"
                
                p_letter = point_match.group(1)
                p_text = line[point_match.end():].strip()
                curr_points.append({"letter": p_letter, "text": p_text})
                
            else:
                # Continuation line
                if curr_points:
                    # Append to last point if active
                    curr_points[-1]["text"] += " " + line
                elif curr_sub_num:
                    # Append to subsection
                    curr_sub_text.append(line)
                else:
                    # Article intro text (before 1.)
                    pass

        save_subsection()
        return subsections

    def create_chunks_from_articles(self, articles: List[ArticleStructure]) -> List[DocumentChunk]:
        chunks = []
        
        for struct in articles:
            # Handle Recitals
            if struct.is_recital:
                ref = LegalReference(recital=struct.id)
                chunks.append(DocumentChunk(
                    content=struct.full_text,
                    reference=ref,
                    page=struct.page,
                    chunk_id=f"recital_{struct.id}",
                    level=DocumentLevel.RECITAL
                ))
                continue

            # Handle Articles
            base_ref = LegalReference(
                chapter=struct.chapter,
                section=struct.section,
                article=struct.id,
                article_title=struct.title
            )
            
            # 1. Full Article Chunk (Good for broad context)
            chunks.append(DocumentChunk(
                content=struct.full_text,
                reference=base_ref,
                page=struct.page,
                chunk_id=f"article_{struct.id}",
                level=DocumentLevel.ARTICLE
            ))
            
            # 2. Subsection Chunks (1., 2.)
            for sub in struct.subsections:
                sub_ref = LegalReference(
                    chapter=struct.chapter,
                    section=struct.section,
                    article=struct.id,
                    article_title=struct.title,
                    subsection=sub["number"]
                )
                
                # Combine subsection text + points for context
                combined_text = sub["text"]
                for p in sub["points"]:
                    combined_text += f"\n({p['letter']}) {p['text']}"

                chunks.append(DocumentChunk(
                    content=combined_text,
                    reference=sub_ref,
                    page=struct.page,
                    chunk_id=f"article_{struct.id}_{sub['number']}",
                    level=DocumentLevel.SUBSECTION,
                    parent_content=struct.title
                ))
                
                # 3. Point Chunks (Granular)
                for p in sub["points"]:
                    point_ref = LegalReference(
                        chapter=struct.chapter,
                        section=struct.section,
                        article=struct.id,
                        article_title=struct.title,
                        subsection=sub["number"],
                        point=p["letter"]
                    )
                    chunks.append(DocumentChunk(
                        content=p["text"],
                        reference=point_ref,
                        page=struct.page,
                        chunk_id=f"article_{struct.id}_{sub['number']}_{p['letter']}",
                        level=DocumentLevel.POINT,
                        parent_content=f"Article {struct.id}({sub['number']})"
                    ))
                    
        return chunks

    def parse_document(self, pdf_path: str) -> List[DocumentChunk]:
        pages = self.load_pdf(pdf_path)
        structures = self.extract_document_structure(pages)
        chunks = self.create_chunks_from_articles(structures)
        return chunks
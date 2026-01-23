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
    Advanced parser for EU Regulations.
    """
    
    def __init__(self):
        # --- REGEX PATTERNS ---
        self.chapter_pattern = re.compile(r"^CHAPTER\s+([IVX\d]+)", re.IGNORECASE)
        self.section_pattern = re.compile(r"^Section\s+(\d+)", re.IGNORECASE)
        self.article_pattern = re.compile(r"^Article\s+(\d+)", re.IGNORECASE)
        
        # Matches "(1)", "1.", "(1 )" at the start of a line
        self.recital_pattern = re.compile(r"^\(?\s*(\d+)\s*\)(?:\.|)") 
        
        self.subsection_pattern = re.compile(r"^(\d+)\.\s") 
        self.point_pattern = re.compile(r"^\(([a-z])\)\s")

    def load_pdf(self, pdf_path: str) -> List[Dict]:
        """Load PDF pages"""
        try:
            logger.info(f"Loading PDF from: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            return loader.load()
        except Exception as e:
            raise ParsingError(f"PDF loading failed: {e}")
    
# In src/ingestion/pdf_parser.py, replace the extract_document_structure method:

    def extract_document_structure(self, pages: List) -> List[ArticleStructure]:
        """
        Parses document using a 2-Phase Strategy.
        Phase 1: Regulation Points (Recitals)
        Phase 2: Articles (Legislative Act)
        
        FIXED: Better chapter/section tracking to prevent missed assignments
        """
        structures = []
        
        # Flatten pages into a stream of lines
        all_lines = []
        for p in pages:
            lines = p.page_content.split('\n')
            for line in lines:
                clean_line = line.strip()
                if clean_line:
                    all_lines.append({"text": clean_line, "page": p.metadata.get("page", 0)})

        # State Variables
        current_chapter = None
        current_chapter_title = None
        current_section = None
        current_section_title = None
        current_struct = None
        in_regulation_phase = True 
        
        i = 0
        while i < len(all_lines):
            line_data = all_lines[i]
            text = line_data["text"]
            page = line_data["page"]
            
            # --- PHASE SWITCHING DETECTION ---
            is_start_of_legislation = False
            
            if "HAVE ADOPTED THIS REGULATION" in text.upper() or "HAVE ADOPTED THIS DIRECTIVE" in text.upper():
                is_start_of_legislation = True
            elif self.chapter_pattern.match(text):
                match = self.chapter_pattern.match(text)
                if match.group(1) in ['1', 'I', 'i']:
                    is_start_of_legislation = True
            elif self.article_pattern.match(text):
                match = self.article_pattern.match(text)
                if match.group(1) == '1':
                    is_start_of_legislation = True

            if is_start_of_legislation:
                logger.info(f"Switching to Legislative Phase at: {text}")
                in_regulation_phase = False

            # --- PHASE 1: REGULATION POINTS ---
            if in_regulation_phase:
                reg_match = self.recital_pattern.match(text)
                
                if reg_match:
                    if current_struct:
                        structures.append(current_struct)
                    
                    reg_id = reg_match.group(1)
                    content = text[reg_match.end():].strip()
                    
                    current_struct = ArticleStructure(
                        id=reg_id,
                        title="Regulation Point",
                        page=page,
                        full_text=content,
                        is_recital=True 
                    )
                elif current_struct and current_struct.is_recital:
                    current_struct.full_text += " " + text
                
                i += 1
                continue

            # --- PHASE 2: LEGISLATIVE ARTICLES ---
            
            # Detect Chapter (and capture title)
            chap_match = self.chapter_pattern.match(text)
            if chap_match:
                current_chapter = chap_match.group(1)
                # Look ahead for chapter title (next line)
                if i + 1 < len(all_lines):
                    next_text = all_lines[i + 1]["text"]
                    # If next line isn't a section/article, it's the chapter title
                    if not self.section_pattern.match(next_text) and not self.article_pattern.match(next_text):
                        current_chapter_title = next_text
                        logger.info(f"Detected Chapter {current_chapter}: {current_chapter_title}")
                        i += 1  # Skip the title line
                i += 1
                continue

            # Detect Section (and capture title)
            sec_match = self.section_pattern.match(text)
            if sec_match:
                current_section = sec_match.group(1)
                # Look ahead for section title
                if i + 1 < len(all_lines):
                    next_text = all_lines[i + 1]["text"]
                    if not self.article_pattern.match(next_text) and len(next_text) < 200:
                        current_section_title = next_text
                        logger.info(f"Detected Section {current_section}: {current_section_title}")
                        i += 1  # Skip the title line
                i += 1
                continue

            # Detect Article
            art_match = self.article_pattern.match(text)
            if art_match:
                # Save previous article
                if current_struct:
                    if not current_struct.is_recital:
                        current_struct.subsections = self._parse_subsections(current_struct.full_text)
                    structures.append(current_struct)

                art_id = art_match.group(1)
                
                # Lookahead for Title (next line)
                title = f"Article {art_id}"
                next_idx = i + 1
                if next_idx < len(all_lines):
                    next_line = all_lines[next_idx]["text"]
                    # If next line is not a subsection and reasonably short, it's a title
                    if not self.subsection_pattern.match(next_line) and len(next_line) < 200:
                        title = next_line
                        i += 1  # Skip title line
                
                # CRITICAL: Assign current chapter/section to this article
                current_struct = ArticleStructure(
                    id=art_id,
                    title=title,
                    page=page,
                    full_text="",
                    chapter=current_chapter,  # ← This captures the chapter
                    section=current_section,  # ← This captures the section
                    is_recital=False
                )
                
                logger.debug(f"Article {art_id} assigned to Chapter {current_chapter}, Section {current_section}")
                
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

        logger.info(f"Extracted {len(structures)} structures.")
        
        # Log chapter/section distribution for debugging
        chapters_found = set(s.chapter for s in structures if s.chapter)
        sections_found = set(s.section for s in structures if s.section)
        logger.info(f"Chapters found: {sorted(chapters_found)}")
        logger.info(f"Sections found: {sorted(sections_found)}")
        
        return structures

    def _parse_subsections(self, text: str) -> List[Dict]:
        subsections = []
        lines = text.split('\n')
        
        curr_sub_num = None
        curr_sub_text = []
        curr_points = []
        
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

            sub_match = self.subsection_pattern.match(line)
            point_match = self.point_pattern.match(line)

            if sub_match:
                save_subsection()
                curr_sub_num = sub_match.group(1)
                curr_sub_text = [line[sub_match.end():].strip()]
                curr_points = []
            elif point_match:
                if not curr_sub_num: curr_sub_num = "0"
                curr_points.append({
                    "letter": point_match.group(1),
                    "text": line[point_match.end():].strip()
                })
            else:
                if curr_points:
                    curr_points[-1]["text"] += " " + line
                elif curr_sub_num:
                    curr_sub_text.append(line)

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
                    chunk_id=f"regulation_point_{struct.id}",
                    level=DocumentLevel.RECITAL
                ))
                continue

            # Handle Articles
            base_ref = LegalReference(
                chapter=struct.chapter,
                section=struct.section,
                section_title=struct.section_title, 
                article=struct.id,
                article_title=struct.title
            )
            
            chunks.append(DocumentChunk(
                content=struct.full_text,
                reference=base_ref,
                page=struct.page,
                chunk_id=f"article_{struct.id}",
                level=DocumentLevel.ARTICLE
            ))
            
            for sub in struct.subsections:
                sub_ref = LegalReference(
                    chapter=struct.chapter,
                    section=struct.section,
                    section_title=struct.section_title, 
                    article=struct.id,
                    article_title=struct.title,
                    subsection=sub["number"]
                )
                
                content_with_points = sub["text"]
                for p in sub["points"]:
                    content_with_points += f"\n({p['letter']}) {p['text']}"

                chunks.append(DocumentChunk(
                    content=content_with_points,
                    reference=sub_ref,
                    page=struct.page,
                    chunk_id=f"article_{struct.id}_{sub['number']}",
                    level=DocumentLevel.SUBSECTION,
                    parent_content=struct.title
                ))
                
                for p in sub["points"]:
                    point_ref = LegalReference(
                        chapter=struct.chapter,
                        section=struct.section,
                        section_title=struct.section_title, 
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
        try:
            pages = self.load_pdf(pdf_path)
            structures = self.extract_document_structure(pages)
            chunks = self.create_chunks_from_articles(structures)
            return chunks
        except Exception as e:
            logger.error(f"Document parsing failed: {e}")
            raise ParsingError(f"Failed to parse document: {e}")
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
    
    STRATEGY:
    1. Regulation Phase: Captures points (1), (2), etc. as "Regulation Points" (Recitals).
    2. Legislative Phase: Captures Articles, Chapters, and Sections.
    """
    
    def __init__(self):
        # --- REGEX PATTERNS ---
        
        # 1. Structure Detectors
        self.chapter_pattern = re.compile(r"^CHAPTER\s+([IVX\d]+)", re.IGNORECASE)
        self.section_pattern = re.compile(r"^Section\s+(\d+)", re.IGNORECASE)
        self.article_pattern = re.compile(r"^Article\s+(\d+)", re.IGNORECASE)
        
        # 2. Regulation Point Detector (Recitals)
        # Matches "(1)", "1.", "(1 )" at the start of a line
        # Also handles plain numbers if they appear in the right context
        self.recital_pattern = re.compile(r"^\(?\s*(\d+)\s*\)(?:\.|)") 
        
        # 3. Internal Article Structure (Subsections and Points)
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
    
    def extract_document_structure(self, pages: List) -> List[ArticleStructure]:
        """
        Parses document using a 2-Phase Strategy.
        Phase 1: Regulation Points (Recitals)
        Phase 2: Articles (Legislative Act)
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
        current_section = None
        current_struct = None
        in_regulation_phase = True 
        
        i = 0
        while i < len(all_lines):
            line_data = all_lines[i]
            text = line_data["text"]
            page = line_data["page"]
            
            # --- CRITICAL FIX: STRICTER PHASE SWITCHING ---
            # Old code switched on ANY "Article X". This caused premature exits.
            # New code only switches on specific markers that denote the start of the law.
            
            is_start_of_legislation = False
            
            # 1. Standard EU Transition Phrase
            if "HAVE ADOPTED THIS REGULATION" in text.upper() or "HAVE ADOPTED THIS DIRECTIVE" in text.upper():
                is_start_of_legislation = True
            
            # 2. Start of Chapter I
            elif self.chapter_pattern.match(text):
                match = self.chapter_pattern.match(text)
                # Only switch if it is Chapter 1 (I)
                if match.group(1) in ['1', 'I', 'i']:
                    is_start_of_legislation = True
            
            # 3. Start of Article 1 (Specific check)
            elif self.article_pattern.match(text):
                match = self.article_pattern.match(text)
                # Only switch if it is specifically Article 1
                if match.group(1) == '1':
                    is_start_of_legislation = True

            if is_start_of_legislation:
                logger.info(f"Switching to Legislative Phase at: {text}")
                in_regulation_phase = False

            # --- PHASE 1: REGULATION POINTS ((1)...(100)) ---
            if in_regulation_phase:
                reg_match = self.recital_pattern.match(text)
                
                # We also check that the number is sequential or reasonable to avoid false positives
                # (e.g., preventing a list item "1." inside a summary from being a recital)
                is_valid_recital = False
                if reg_match:
                    # Optional: Add logic here to ensure Recital 5 follows Recital 4
                    # For now, we trust the regex and the phase
                    is_valid_recital = True

                if is_valid_recital:
                    # Save previous structure
                    if current_struct:
                        structures.append(current_struct)
                    
                    # Create new Regulation Point
                    reg_id = reg_match.group(1)
                    # Clean the content (remove the number)
                    content = text[reg_match.end():].strip()
                    
                    current_struct = ArticleStructure(
                        id=reg_id,
                        title="Regulation Point", # Internal Label
                        page=page,
                        full_text=content,
                        is_recital=True 
                    )
                elif current_struct and current_struct.is_recital:
                    # Append text to current Regulation Point
                    current_struct.full_text += " " + text
                
                i += 1
                continue

            # --- PHASE 2: LEGISLATIVE ARTICLES ---
            
            # Detect Chapter
            chap_match = self.chapter_pattern.match(text)
            if chap_match:
                current_chapter = chap_match.group(1)
                i += 1
                continue

            # Detect Section
            sec_match = self.section_pattern.match(text)
            if sec_match:
                current_section = sec_match.group(1)
                i += 1
                continue

            # Detect Article
            art_match = self.article_pattern.match(text)
            if art_match:
                if current_struct:
                    # Parse subsections for the finished article
                    if not current_struct.is_recital:
                        current_struct.subsections = self._parse_subsections(current_struct.full_text)
                    structures.append(current_struct)

                art_id = art_match.group(1)
                
                # Lookahead for Title (next line)
                title = f"Article {art_id}"
                next_idx = i + 1
                if next_idx < len(all_lines):
                    next_line = all_lines[next_idx]["text"]
                    # If next line is not a subsection "1." and reasonably short, it's a title
                    if not self.subsection_pattern.match(next_line) and len(next_line) < 200:
                        title = next_line
                        i += 1 # Skip title line
                
                current_struct = ArticleStructure(
                    id=art_id,
                    title=title,
                    page=page,
                    full_text="",
                    chapter=current_chapter,
                    section=current_section,
                    is_recital=False
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

        logger.info(f"Extracted {len(structures)} structures.")
        return structures

    def _parse_subsections(self, text: str) -> List[Dict]:
        """
        Parses the hierarchy within an Article text:
        1. Subsection text
           (a) Point text
        """
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
        """Convert extracted structures into searchable chunks"""
        chunks = []
        
        for struct in articles:
            # --- HANDLE REGULATION POINTS (Recitals) ---
            if struct.is_recital:
                # We map this to 'recital' metadata so exact retriever finds it
                ref = LegalReference(recital=struct.id)
                chunks.append(DocumentChunk(
                    content=struct.full_text,
                    reference=ref,
                    page=struct.page,
                    chunk_id=f"regulation_point_{struct.id}",
                    level=DocumentLevel.RECITAL # Mapped to Recital Level
                ))
                continue

            # --- HANDLE ARTICLES ---
            base_ref = LegalReference(
                chapter=struct.chapter,
                section=struct.section,
                article=struct.id,
                article_title=struct.title
            )
            
            # 1. Full Article Chunk
            chunks.append(DocumentChunk(
                content=struct.full_text,
                reference=base_ref,
                page=struct.page,
                chunk_id=f"article_{struct.id}",
                level=DocumentLevel.ARTICLE
            ))
            
            # 2. Subsection Chunks
            for sub in struct.subsections:
                sub_ref = LegalReference(
                    chapter=struct.chapter,
                    section=struct.section,
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
                
                # 3. Point Chunks
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
        try:
            pages = self.load_pdf(pdf_path)
            structures = self.extract_document_structure(pages)
            chunks = self.create_chunks_from_articles(structures)
            return chunks
        except Exception as e:
            logger.error(f"Document parsing failed: {e}")
            raise ParsingError(f"Failed to parse document: {e}")
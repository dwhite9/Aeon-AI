"""
Aeon RAG Pipeline
Document processing and semantic chunking module
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime

# Document processors
from pypdf import PdfReader
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
import markdown

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a semantically chunked piece of text"""
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    char_count: int
    created_at: str


class DocumentProcessor:
    """Process various document types and extract text"""

    @staticmethod
    def process_pdf(file_path: Path) -> str:
        """Extract text from PDF"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise

    @staticmethod
    def process_docx(file_path: Path) -> str:
        """Extract text from DOCX"""
        try:
            doc = DocxDocument(file_path)
            text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            return text.strip()
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
            raise

    @staticmethod
    def process_html(file_path: Path) -> str:
        """Extract text from HTML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text(separator='\n\n')
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                return text
        except Exception as e:
            logger.error(f"Error processing HTML {file_path}: {e}")
            raise

    @staticmethod
    def process_markdown(file_path: Path) -> str:
        """Extract text from Markdown"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_text = f.read()
                # Convert to HTML then extract text for better structure preservation
                html = markdown.markdown(md_text)
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text(separator='\n\n').strip()
        except Exception as e:
            logger.error(f"Error processing Markdown {file_path}: {e}")
            raise

    @staticmethod
    def process_text(file_path: Path) -> str:
        """Extract text from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            raise

    @classmethod
    def process_document(cls, file_path: Path) -> str:
        """
        Process a document and extract text based on file extension

        Args:
            file_path: Path to the document

        Returns:
            Extracted text content

        Raises:
            ValueError: If file type is not supported
        """
        suffix = file_path.suffix.lower()

        processors = {
            '.pdf': cls.process_pdf,
            '.docx': cls.process_docx,
            '.doc': cls.process_docx,
            '.html': cls.process_html,
            '.htm': cls.process_html,
            '.md': cls.process_markdown,
            '.markdown': cls.process_markdown,
            '.txt': cls.process_text,
        }

        processor = processors.get(suffix)
        if not processor:
            raise ValueError(f"Unsupported file type: {suffix}")

        logger.info(f"Processing document: {file_path.name}")
        return processor(file_path)


class SemanticChunker:
    """
    Chunk documents semantically for optimal RAG performance

    Uses a combination of strategies:
    1. Sentence-based chunking with overlap
    2. Respects paragraph boundaries
    3. Maintains semantic coherence
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 100,
        respect_paragraphs: bool = True
    ):
        """
        Initialize semantic chunker

        Args:
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Overlap between chunks for context preservation
            min_chunk_size: Minimum chunk size (avoid tiny chunks)
            respect_paragraphs: Try to avoid breaking paragraphs
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_paragraphs = respect_paragraphs

    def chunk_text(
        self,
        text: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Chunk text into semantic pieces

        Args:
            text: Text to chunk
            document_id: Unique identifier for the document
            metadata: Additional metadata to attach to chunks

        Returns:
            List of DocumentChunk objects
        """
        if not text or len(text) < self.min_chunk_size:
            logger.warning("Text too short to chunk effectively")
            return []

        # Generate document ID if not provided
        if not document_id:
            document_id = self._generate_document_id(text)

        metadata = metadata or {}
        chunks = []

        # Split into paragraphs if respect_paragraphs is enabled
        if self.respect_paragraphs:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        else:
            paragraphs = [text]

        current_chunk = ""
        chunk_index = 0

        for paragraph in paragraphs:
            # If paragraph is larger than chunk_size, split it
            if len(paragraph) > self.chunk_size:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, document_id, metadata, chunk_index
                    ))
                    chunk_index += 1
                    current_chunk = ""

                # Split large paragraph into smaller chunks
                para_chunks = self._split_large_text(paragraph)
                for para_chunk in para_chunks:
                    chunks.append(self._create_chunk(
                        para_chunk, document_id, metadata, chunk_index
                    ))
                    chunk_index += 1
            else:
                # Check if adding this paragraph exceeds chunk_size
                if len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                    if current_chunk:
                        chunks.append(self._create_chunk(
                            current_chunk, document_id, metadata, chunk_index
                        ))
                        chunk_index += 1

                        # Add overlap from previous chunk
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap_text + "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph

        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(self._create_chunk(
                current_chunk, document_id, metadata, chunk_index
            ))

        logger.info(f"Created {len(chunks)} chunks from document {document_id}")
        return chunks

    def _split_large_text(self, text: str) -> List[str]:
        """Split large text into chunks respecting sentence boundaries"""
        chunks = []
        current = ""

        # Split by sentences (simple approach)
        sentences = text.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current) + len(sentence) + 1 > self.chunk_size:
                if current:
                    chunks.append(current)
                    # Add overlap
                    overlap = ' '.join(current.split()[-20:])  # Last ~20 words
                    current = overlap + " " + sentence
                else:
                    # Sentence itself is too long, force split
                    chunks.append(sentence[:self.chunk_size])
                    current = sentence[self.chunk_size:]
            else:
                if current:
                    current += " " + sentence
                else:
                    current = sentence

        if current:
            chunks.append(current)

        return chunks

    def _create_chunk(
        self,
        content: str,
        document_id: str,
        metadata: Dict[str, Any],
        chunk_index: int
    ) -> DocumentChunk:
        """Create a DocumentChunk object"""
        chunk_id = self._generate_chunk_id(document_id, chunk_index)

        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content.strip(),
            metadata={
                **metadata,
                "chunk_method": "semantic",
                "chunk_size_target": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            },
            chunk_index=chunk_index,
            char_count=len(content),
            created_at=datetime.utcnow().isoformat()
        )

    @staticmethod
    def _generate_document_id(text: str) -> str:
        """Generate a unique ID for a document based on its content"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @staticmethod
    def _generate_chunk_id(document_id: str, chunk_index: int) -> str:
        """Generate a unique ID for a chunk"""
        return f"{document_id}_{chunk_index:04d}"


# Convenience function
def process_and_chunk_document(
    file_path: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    metadata: Optional[Dict[str, Any]] = None
) -> List[DocumentChunk]:
    """
    Process a document and chunk it in one step

    Args:
        file_path: Path to the document
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        metadata: Additional metadata

    Returns:
        List of DocumentChunk objects
    """
    # Extract text
    text = DocumentProcessor.process_document(file_path)

    # Add file metadata
    file_metadata = {
        "filename": file_path.name,
        "file_type": file_path.suffix,
        "file_size": file_path.stat().st_size,
        **(metadata or {})
    }

    # Chunk text
    chunker = SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return chunker.chunk_text(
        text,
        metadata=file_metadata
    )

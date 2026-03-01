import PyPDF2
from typing import List
from config import MAX_CHUNK_SIZE

class DocumentProcessor:    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = []
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text.append(f"[Page {page_num + 1}]\n{page_text}")
                return '\n\n'.join(text).strip()
        except Exception as e:
            raise RuntimeError(f"PDF extraction failed: {str(e)}")

    @staticmethod
    def chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE) -> List[str]:
        """Split text into chunks with overlap for better context."""
        chunks = []
        
        paragraphs = text.split('\n\n')
        
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            if para_length > max_size:
                chunks.extend(DocumentProcessor._split_long_paragraph(para, max_size, current_chunk, current_length))
                current_chunk = []
                current_length = 0
            else:
                current_chunk, current_length = DocumentProcessor._add_to_chunk(
                    para, current_chunk, current_length, max_size, chunks
                )
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return DocumentProcessor._create_overlapping_chunks(chunks, max_size)

    @staticmethod
    def _split_long_paragraph(para: str, max_size: int, current_chunk: List, current_length: int) -> List[str]:
        """Split a long paragraph into sentences."""
        chunks = []
        sentences = para.split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if not sentence.endswith('.'):
                sentence += '.'
            
            if len(sentence) > max_size:
                chunks.append(sentence[:max_size])
            else:
                if current_length + len(sentence) > max_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                current_chunk.append(sentence)
                current_length += len(sentence) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    @staticmethod
    def _add_to_chunk(para: str, current_chunk: List, current_length: int, max_size: int, chunks: List) -> tuple:
        """Add paragraph to current chunk or create new chunk."""
        para_length = len(para)
        
        if current_length + para_length > max_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(para)
        current_length += para_length + 2
        
        return current_chunk, current_length

    @staticmethod
    def _create_overlapping_chunks(chunks: List[str], max_size: int) -> List[str]:
        """Create overlapping chunks for better retrieval."""
        overlapping_chunks = []
        for i in range(len(chunks)):
            if i > 0:
                combined = chunks[i-1] + " " + chunks[i]
                if len(combined) <= max_size * 1.5:
                    overlapping_chunks.append(combined)
            overlapping_chunks.append(chunks[i])
        
        return overlapping_chunks
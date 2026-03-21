import re
import PyPDF2
import tiktoken
from typing import List, Union, IO


MAX_TOKENS = 400
_ENCODER = tiktoken.get_encoding("cl100k_base")


class DocumentProcessor:

    @staticmethod
    def extract_text_from_pdf(pdf_source: Union[str, IO]) -> str:

        try:
            if isinstance(pdf_source, str):
                file = open(pdf_source, 'rb')
                should_close = True
            else:
                file = pdf_source
                should_close = False

            reader = PyPDF2.PdfReader(file)
            text = []
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text.append(f"[Page {page_num + 1}]\n{page_text}")

            if should_close:
                file.close()

            return '\n\n'.join(text).strip()

        except Exception as e:
            raise RuntimeError(f"PDF extraction failed: {str(e)}")

    @staticmethod
    def _count_tokens(text: str) -> int:
        return len(_ENCODER.encode(text))

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Sentence splitter using regex boundary detection."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap_sentences: int = 2) -> List[str]:

        paragraphs = text.split('\n\n')
        all_sentences: List[str] = []
        for para in paragraphs:
            para = para.strip()
            if para:
                all_sentences.extend(DocumentProcessor._split_into_sentences(para))

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_tokens = 0

        for sentence in all_sentences:
            sentence_tokens = DocumentProcessor._count_tokens(sentence)

            # Hard-truncate sentences that exceed the limit on their own
            if sentence_tokens > max_tokens:
                token_ids = _ENCODER.encode(sentence)
                sentence = _ENCODER.decode(token_ids[:max_tokens])
                sentence_tokens = max_tokens

            if current_tokens + sentence_tokens > max_tokens and current_sentences:
                chunks.append(' '.join(current_sentences))
                # Carry over last N sentences as overlap into the next chunk
                current_sentences = current_sentences[-overlap_sentences:]
                current_tokens = DocumentProcessor._count_tokens(' '.join(current_sentences))

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        if current_sentences:
            chunks.append(' '.join(current_sentences))

        return chunks

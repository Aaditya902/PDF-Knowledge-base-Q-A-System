import logging
from typing import List, Tuple
from google import genai

from config import GOOGLE_API_KEY, TEMPERATURE, TOP_P, MAX_OUTPUT_TOKENS

logger = logging.getLogger(__name__)


class QAEngine:

    def __init__(self, retriever, model_name: str):
        self.retriever = retriever
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = model_name

    def generate_answer(self, query: str) -> Tuple[str, float, List[Tuple[str, float]]]:

        try:
            results = self.retriever.query(query)

            if not results:
                return self._fallback_to_single_result(query)

            avg_similarity = sum(r[1] for r in results) / len(results)
            return self._generate_with_context(results, query, avg_similarity)

        except Exception as e:
            logger.exception("Error generating answer for query: %s", query)
            return f"Error generating response: {str(e)}", 0.0, []

    def _fallback_to_single_result(self, query: str) -> Tuple[str, float, List]:
        results = self.retriever.query(query, k=1)
        if results:
            logger.info("Primary query returned no results; using single-result fallback")
            return self._generate_with_context(results, query, results[0][1])
        return (
            "I couldn't find relevant information in the document. "
            "Please try rephrasing your question or the document might "
            "not contain this information."
        ), 0.0, []

    def _generate_with_context(
        self,
        results: List[Tuple[str, float]],
        query: str,
        avg_similarity: float
    ) -> Tuple[str, float, List]:
        context = self._prepare_context(results[:3])
        prompt = self._build_prompt(context, query)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            }
        )

        return response.text, avg_similarity, results

    def _prepare_context(self, results: List[Tuple[str, float]]) -> str:
        parts = []
        for i, (chunk, score) in enumerate(results):
            parts.append(f"[Context {i+1} (Relevance: {score:.2f})]:\n{chunk}")
        return '\n\n'.join(parts)

    def _build_prompt(self, context: str, query: str) -> str:
        return f"""You are a helpful assistant that answers questions based on the provided document context.

CONTEXT FROM DOCUMENT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain the exact answer, use the most relevant information available
3. If the information is completely absent, say "Based on the document, I cannot find information about [topic]"
4. Be specific and quote relevant parts when possible
5. Keep your answer concise but informative

ANSWER:"""

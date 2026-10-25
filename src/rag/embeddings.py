from typing import Sequence


class GeminiEmbeddings:
    """Small adapter around the Google GenAI embedding API."""

    def __init__(self, model: str) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed. Install requirements.txt first."
            ) from exc
        self._client = genai.Client()
        self.model = model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        from google.genai import types

        vectors: list[list[float]] = []
        for text in texts:
            response = self._client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            if not response.embeddings:
                raise RuntimeError("Gemini returned no embedding for a document")
            vectors.append(list(response.embeddings[0].values))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        from google.genai import types

        response = self._client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return list(response.embeddings[0].values)

from .qdrant_store import LocalQdrantStore


class RerankingRetriever:
    def __init__(self, store: LocalQdrantStore, top_n: int = 6) -> None:
        try:
            from flashrank import Ranker
        except ImportError as exc:
            raise RuntimeError(
                "flashrank is not installed. Install requirements.txt first."
            ) from exc
        self.store = store
        self.top_n = top_n
        self.ranker = Ranker()

    def retrieve(self, query: str) -> list[dict]:
        from flashrank import RerankRequest

        candidates = self.store.search(query)
        if not candidates:
            return []
        passages = [
            {
                "id": index,
                "text": candidate["text"],
                "meta": candidate,
            }
            for index, candidate in enumerate(candidates)
        ]
        ranked = self.ranker.rerank(
            RerankRequest(query=query, passages=passages)
        )[: self.top_n]
        results = [
            {
                **item["meta"],
                "rerank_score": item.get("score", 0.0),
            }
            for item in ranked
        ]
        return sorted(
            results,
            key=lambda item: (item.get("priority", 5), -item["rerank_score"]),
        )

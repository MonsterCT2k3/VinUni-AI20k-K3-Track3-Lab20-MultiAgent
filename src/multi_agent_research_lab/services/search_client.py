"""Search client abstraction for ResearcherAgent.

Supports Tavily online search, local offline benchmark corpus, and mock fallback.
"""

import json
import logging
import ssl
import urllib.request
from pathlib import Path
from typing import Any

import certifi

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client with Tavily, local corpus, and fallback support."""

    def __init__(
        self,
        settings: Settings | None = None,
        corpus_dir: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.corpus_dir = corpus_dir or Path("ai_agent_offline_research_corpus_v2/topics")

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        """Perform web search via Tavily API with SSL protection."""
        if not self.settings.tavily_api_key:
            return []

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "MultiAgentResearchLab/1.0"},
        )

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(
            req, timeout=float(self.settings.timeout_seconds), context=ssl_ctx
        ) as resp:
            raw_data = json.loads(resp.read().decode("utf-8"))

        results: list[SourceDocument] = []
        for item in raw_data.get("results", []):
            results.append(
                SourceDocument(
                    title=item.get("title", "Untitled Source"),
                    url=item.get("url"),
                    snippet=item.get("content", ""),
                    metadata={"provider": "tavily", "score": item.get("score", 0.0)},
                )
            )
        return results

    def _score_topic_match(self, query: str, topic_path: Path) -> int:
        """Calculate word overlap score between query and topic metadata."""
        query_words = set(query.lower().replace("-", " ").replace("_", " ").split())
        score = 0
        name_words = set(topic_path.stem.lower().replace("_", " ").replace("-", " ").split())
        score += len(query_words.intersection(name_words)) * 3

        try:
            with open(topic_path, encoding="utf-8") as f:
                data = json.load(f)
            topic_meta = data.get("topic", {})
            title = topic_meta.get("name", "").lower()
            tags = [t.lower() for t in topic_meta.get("tags", [])]
            for word in query_words:
                if word in title:
                    score += 2
                if any(word in t for t in tags):
                    score += 2
        except Exception:
            pass

        return score

    def _search_offline_corpus(self, query: str, max_results: int) -> list[SourceDocument]:
        """Retrieve source documents from local benchmark corpus."""
        if not self.corpus_dir.exists() or not self.corpus_dir.is_dir():
            return []

        json_files = list(self.corpus_dir.glob("*.json"))
        if not json_files:
            return []

        # Sắp xếp các topic theo điểm khớp từ khóa với query
        scored_files = [(self._score_topic_match(query, p), p) for p in json_files]
        scored_files.sort(key=lambda x: x[0], reverse=True)
        best_file = scored_files[0][1]

        try:
            with open(best_file, encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)

            kb = data.get("knowledge_base", {})
            raw_sources = kb.get("source_documents", [])
            topic_id = data.get("benchmark_metadata", {}).get("topic_id", best_file.stem)

            docs: list[SourceDocument] = []
            for src in raw_sources:
                title = src.get("title", "Untitled Document")
                url = src.get("provenance_url")
                # Trích xuất snippet tóm tắt từ key_takeaways hoặc full_text
                takeaways = src.get("key_takeaways", [])
                snippet = " ".join(takeaways) if takeaways else src.get("full_text", "")[:300]
                docs.append(
                    SourceDocument(
                        title=title,
                        url=url,
                        snippet=snippet,
                        metadata={
                            "document_id": src.get("document_id", ""),
                            "topic_id": topic_id,
                            "citation_label": src.get("citation_label", ""),
                            "is_synthetic": src.get("is_synthetic", False),
                            "provider": "offline_corpus",
                        },
                    )
                )

            # Nếu chưa đủ tài liệu, bổ sung từ knowledge_articles
            if len(docs) < max_results:
                articles = kb.get("knowledge_articles", [])
                for art in articles:
                    docs.append(
                        SourceDocument(
                            title=f"Article: {art.get('title', 'Knowledge Article')}",
                            url=None,
                            snippet=art.get("content", "")[:350],
                            metadata={
                                "article_id": art.get("article_id", ""),
                                "topic_id": topic_id,
                                "provider": "offline_corpus",
                            },
                        )
                    )

            return docs[:max_results]
        except Exception as exc:
            logger.warning("Lỗi đọc offline corpus file %s: %s", best_file, exc)
            return []

    def _fallback_mock_docs(self, max_results: int) -> list[SourceDocument]:
        """Hardcoded mock documents as final safety fallback."""
        mock_data = [
            SourceDocument(
                title="AutoGen: Multi-Agent Conversation Framework",
                url="https://arxiv.org/abs/2308.08155",
                snippet="AutoGen mô tả framework xây dựng ứng dụng LLM dưới dạng hội thoại "
                "giữa các agent chuyên biệt hóa với state handoff.",
                metadata={"source": "mock"},
            ),
            SourceDocument(
                title="Anthropic: Building Effective Agents",
                url="https://www.anthropic.com/engineering/building-effective-agents",
                snippet="Khuyên dùng các pattern từ đơn giản đến phức tạp: Orchestrator-Workers, "
                "Routing, Evaluator-Optimizer loop.",
                metadata={"source": "mock"},
            ),
            SourceDocument(
                title="LangGraph: Stateful Multi-Agent Orchestration",
                url="https://langchain-ai.github.io/langgraph/",
                snippet="LangGraph cung cấp StateGraph quản lý chu kỳ trạng thái và handoff giữa "
                "các node điều phối Agent.",
                metadata={"source": "mock"},
            ),
        ]
        return mock_data[:max_results]

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        # 1. Thử gọi Tavily API nếu có key
        if self.settings.tavily_api_key:
            try:
                results = self._search_tavily(query, max_results)
                if results:
                    return results
            except Exception as exc:
                logger.warning("Tavily search thất bại (%s), chuyển sang offline corpus.", exc)

        # 2. Tìm kiếm trong Offline Corpus
        corpus_results = self._search_offline_corpus(query, max_results)
        if corpus_results:
            return corpus_results

        # 3. Fallback sang dữ liệu mẫu Mock
        return self._fallback_mock_docs(max_results)

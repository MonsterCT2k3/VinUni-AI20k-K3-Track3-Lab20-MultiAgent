"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Bảng giá ước tính USD / 1M tokens cho một số model phổ biến
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # model: (input_cost_per_1m, output_cost_per_1m)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client with retry, timeout, cost estimation, and mock fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: OpenAI | None = None
        if self.settings.openai_api_key:
            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                timeout=float(self.settings.timeout_seconds),
            )

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate total cost in USD based on model pricing."""
        input_rate, output_rate = MODEL_PRICING.get(model, (0.15, 0.60))
        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * output_rate
        return round(input_cost + output_cost, 6)

    def _mock_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Offline mock response when OPENAI_API_KEY is not configured."""
        sys_lower = system_prompt.lower()
        if "analyst" in sys_lower or "phân tích" in sys_lower:
            content = (
                "### Phân Tích & Đối Chiếu Luận Điểm (Mock Analysis)\n\n"
                "1. **Điểm đồng thuận**: Cả hai kiến trúc đều giải quyết bài toán "
                "mở rộng tri thức cho LLM.\n"
                "2. **Điểm khác biệt / Trade-offs**:\n"
                "   - Single-Agent: Đơn giản, độ trễ thấp, nhưng dễ loãng context.\n"
                "   - Multi-Agent: Phân tách vai trò rõ ràng, độ sâu cao hơn, "
                "nhưng tốn chi phí và độ trễ cao hơn.\n"
                "3. **Đánh giá bằng chứng**: Nguồn dữ liệu cung cấp đầy đủ thông tin."
            )
        elif "writer" in sys_lower or "báo cáo" in sys_lower:
            content = (
                "# Báo Cáo Nghiên Cứu Chuyên Sâu (Mock Report)\n\n"
                "## 1. Tổng Quan\n"
                "Nghiên cứu về kiến trúc Single-Agent và Multi-Agent cho thấy "
                "sự đánh đổi rõ ràng giữa hiệu năng và chi phí [1].\n\n"
                "## 2. So Sánh Chi Tiết\n"
                "- **Single-Agent Baseline**: Thích hợp cho các tác vụ nhanh [1].\n"
                "- **Multi-Agent Pipeline**: Giúp chia nhỏ bài toán thành các vai trò "
                "chuyên biệt (Researcher, Analyst, Writer) [2].\n\n"
                "## 3. Kết Luận\n"
                "Lựa chọn kiến trúc phụ thuộc vào yêu cầu bài toán cụ thể.\n\n"
                "### Tài Liệu Tham Khảo (References)\n"
                "[1] Anthropic: Building Effective Agents\n"
                "[2] LangGraph Multi-Agent Architecture Guide"
            )
        else:
            content = (
                f"Mock LLM Response cho truy vấn: '{user_prompt[:60]}...'. "
                "Hệ thống đang hoạt động ở chế độ giả lập offline."
            )

        input_tokens = max(10, (len(system_prompt) + len(user_prompt)) // 4)
        output_tokens = max(10, len(content) // 4)
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with automatic retry and token tracking."""
        if not self._client:
            logger.info("OPENAI_API_KEY không có sẵn, sử dụng Mock LLM Response.")
            return self._mock_complete(system_prompt, user_prompt)

        @retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(Exception),
        )
        def _call_openai() -> LLMResponse:
            assert self._client is not None
            response = self._client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            content = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            cost_usd = self._estimate_cost(
                self.settings.openai_model, input_tokens, output_tokens
            )

            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )

        try:
            return _call_openai()
        except Exception as exc:
            logger.error("Lỗi khi gọi OpenAI API sau các lần retry: %s", exc)
            raise

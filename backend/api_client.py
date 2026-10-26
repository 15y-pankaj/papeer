import json
import os
from collections.abc import Iterator
from typing import Any

import httpx


class APIClientError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code


class PapeerAPIClient:
    """HTTP client used by Streamlit; all domain work stays in the API service."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (
            base_url or os.getenv("PAPEER_API_URL", "http://127.0.0.1:8000")
        ).rstrip("/")
        self.timeout = httpx.Timeout(60, connect=10)
        self._client = httpx.Client(timeout=self.timeout)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, f"{self.base_url}{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise APIClientError(0, f"Could not connect to the Papeer API: {exc}") from exc
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except (ValueError, AttributeError):
                detail = response.text
            raise APIClientError(response.status_code, str(detail))
        if response.status_code == 204:
            return None
        return response.json()

    def get(self, path: str) -> Any:
        return self._request("GET", path)

    def post(self, path: str, **kwargs: Any) -> Any:
        return self._request("POST", path, **kwargs)

    def stream(self, path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        try:
            with self._client.stream(
                "POST",
                f"{self.base_url}{path}",
                json=payload,
                timeout=httpx.Timeout(60, connect=10, read=None),
                headers={"Accept": "text/event-stream"},
            ) as response:
                if response.is_error:
                    try:
                        detail = response.json().get("detail", response.text)
                    except (ValueError, AttributeError):
                        detail = response.text
                    raise APIClientError(response.status_code, str(detail))

                event_name = "message"
                data_lines: list[str] = []
                for line in response.iter_lines():
                    if not line:
                        if data_lines:
                            try:
                                payload = json.loads("\n".join(data_lines))
                            except json.JSONDecodeError as exc:
                                raise APIClientError(
                                    response.status_code,
                                    "The API returned malformed streaming data.",
                                ) from exc
                            yield {"event": event_name, "data": payload}
                        event_name = "message"
                        data_lines = []
                    elif line.startswith("event:"):
                        event_name = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].strip())

                if data_lines:
                    yield {
                        "event": event_name,
                        "data": json.loads("\n".join(data_lines)),
                    }
        except APIClientError:
            raise
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise APIClientError(0, f"The API stream failed: {exc}") from exc

    def list_sessions(self) -> list[dict[str, Any]]:
        return self.get("/api/sessions")

    def create_session(self) -> dict[str, Any]:
        return self.post("/api/sessions")

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        result = self.get(f"/api/sessions/{session_id}/history")
        return result["messages"]

    def list_documents(self, session_id: str) -> list[str]:
        result = self.get(f"/api/sessions/{session_id}/documents")
        return result["documents"]

    def upload_documents(
        self, session_id: str, files: list[tuple[str, bytes, str]]
    ) -> list[dict[str, Any]]:
        multipart = [
            ("files", (filename, content, content_type))
            for filename, content, content_type in files
        ]
        result = self.post(
            f"/api/sessions/{session_id}/documents",
            files=multipart,
        )
        return result["results"]

    def load_urls(self, session_id: str, urls: list[str]) -> list[dict[str, Any]]:
        result = self.post(
            f"/api/sessions/{session_id}/documents/urls",
            json={"urls": urls},
        )
        return result["results"]

    def load_arxiv(self, session_id: str, query: str) -> dict[str, Any]:
        return self.post(
            f"/api/sessions/{session_id}/documents/arxiv",
            json={"query": query},
        )

    def chat_stream(self, session_id: str, query: str) -> Iterator[dict[str, Any]]:
        return self.stream(f"/api/sessions/{session_id}/chat", {"query": query})

    def btw_stream(self, query: str) -> Iterator[dict[str, Any]]:
        return self.stream("/api/btw", {"query": query})

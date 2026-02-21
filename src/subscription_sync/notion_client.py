"""Notion API repository with correct schema mapping.

Demonstrates:
- Repository pattern: isolate persistence from domain logic
- Correct Notion API property type serialization:
  - title: {"title": [{"text": {"content": "..."}}]}
  - number: {"number": float}
  - status: {"status": {"name": "Active"}} (NOT select!)
  - select: {"select": {"name": "..."}}
  - date: {"date": {"start": "YYYY-MM-DD"}}
  - relation: {"relation": [{"id": "page-uuid"}]}
- Idempotent upsert (query → match → create or update)
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from subscription_sync.config import Settings
from subscription_sync.models import SubscriptionRecord

logger = structlog.get_logger()

NOTION_API_VERSION = "2022-06-28"
NOTION_BASE_URL = "https://api.notion.com/v1"


class NotionSubscriptionRepo:
    """Repository for subscription records in Notion.

    Encapsulates all Notion API interaction. Domain logic never
    touches HTTP payloads directly.
    """

    def __init__(self, settings: Settings) -> None:
        self._database_id = settings.notion_database_id
        self._headers = {
            "Authorization": f"Bearer {settings.notion_api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "Notion-Version": NOTION_API_VERSION,
        }
        self._client = httpx.Client(
            base_url=NOTION_BASE_URL,
            headers=self._headers,
            timeout=30.0,
        )
        self._existing_names: list[str] | None = None

    def get_existing_subscription_names(self) -> list[str]:
        """Query all subscription names from the database (cached)."""
        if self._existing_names is not None:
            return self._existing_names

        names: list[str] = []
        payload: dict[str, Any] = {"page_size": 100}
        has_more = True

        while has_more:
            response = self._client.post(
                f"/databases/{self._database_id}/query",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            for page in data.get("results", []):
                title_prop = page.get("properties", {}).get("Name", {})
                title_items = title_prop.get("title", [])
                if title_items:
                    names.append(title_items[0].get("text", {}).get("content", ""))

            has_more = data.get("has_more", False)
            if has_more:
                payload["start_cursor"] = data["next_cursor"]

        self._existing_names = names
        logger.info("loaded_existing_subscriptions", count=len(names))
        return names

    def create_subscription(self, record: SubscriptionRecord) -> str:
        """Create a new subscription page in Notion.

        Returns the page ID of the created page.
        """
        properties = self._build_properties(record)

        response = self._client.post(
            "/pages",
            json={
                "parent": {"database_id": self._database_id},
                "properties": properties,
            },
        )
        response.raise_for_status()
        page_id = response.json()["id"]

        logger.info("created_subscription", name=record.name, page_id=page_id)

        # Invalidate cache
        self._existing_names = None
        return page_id

    def update_subscription(
        self, page_id: str, record: SubscriptionRecord
    ) -> None:
        """Update an existing subscription page."""
        properties = self._build_properties(record)

        response = self._client.patch(
            f"/pages/{page_id}",
            json={"properties": properties},
        )
        response.raise_for_status()
        logger.info("updated_subscription", name=record.name, page_id=page_id)

        # Invalidate cache
        self._existing_names = None

    def find_page_by_name(self, name: str) -> str | None:
        """Find a page ID by subscription name (exact match)."""
        response = self._client.post(
            f"/databases/{self._database_id}/query",
            json={
                "filter": {
                    "property": "Name",
                    "title": {"equals": name},
                },
                "page_size": 1,
            },
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        return results[0]["id"] if results else None

    def upsert_subscription(self, record: SubscriptionRecord) -> str:
        """Idempotent create-or-update.

        If a page with the same name exists, update it.
        Otherwise, create a new page.
        """
        existing_page_id = self.find_page_by_name(record.name)

        if existing_page_id:
            self.update_subscription(existing_page_id, record)
            return existing_page_id
        else:
            return self.create_subscription(record)

    def test_connection(self) -> bool:
        """Verify Notion API credentials and database access."""
        try:
            response = self._client.get(f"/databases/{self._database_id}")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error("notion_connection_failed", error=str(e))
            return False

    # ── Property Builders ─────────────────────────────────────────────────

    def _build_properties(self, record: SubscriptionRecord) -> dict[str, Any]:
        """Build Notion API properties payload with correct types.

        Critical: Each property type requires a specific payload shape.
        Getting this wrong causes silent failures or 400 errors.
        """
        notion_props = record.to_notion_properties()
        properties: dict[str, Any] = {}

        for key, value in notion_props.items():
            # Handle date properties (prefixed with "date:")
            if key.startswith("date:"):
                parts = key.split(":")
                prop_name = parts[1]
                field = parts[2]  # "start", "end", or "is_datetime"

                if field == "is_datetime":
                    continue  # Consumed by the start/end handler

                if prop_name not in properties:
                    properties[prop_name] = {"date": {"start": value}}
                else:
                    properties[prop_name]["date"][field] = value
                continue

            # Map property names to correct Notion API types
            if key == "Name":
                properties[key] = {
                    "title": [{"text": {"content": value}}]
                }
            elif key == "Amount":
                properties[key] = {"number": value}
            elif key == "Status":
                # CRITICAL: Status is a 'status' type, NOT 'select'
                properties[key] = {"status": {"name": value}}
            elif key == "Cycle":
                properties[key] = {"select": {"name": value}}
            elif key == "Categories":
                # Relation type: requires page ID
                properties[key] = {"relation": [{"id": value}]}

        return properties

    def close(self) -> None:
        """Clean up HTTP client."""
        self._client.close()

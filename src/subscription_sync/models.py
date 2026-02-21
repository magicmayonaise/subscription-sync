"""Domain models with strict validation and Notion schema mapping.

Demonstrates:
- Pydantic v2 with custom validators (model_validator)
- Enum-driven type safety for status/cycle/category
- Domain invariants (canceled_date ↔ status consistency)
- Permissive → strict promotion pattern (EmailExtraction → SubscriptionRecord)
- Notion property serialization with correct types per schema
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ═══════════════════════════════════════════════════════════════════════════
# Enums (aligned to Notion database options)
# ═══════════════════════════════════════════════════════════════════════════


class SubscriptionStatus(str, Enum):
    """Notion 'Status' property options (status type, not select)."""

    ACTIVE = "Active"
    CANCELED = "Canceled"


class BillingCycle(str, Enum):
    """Notion 'Cycle' property options (select type)."""

    MONTHLY = "Monthly"
    QUARTERLY = "Quarterly"
    YEARLY = "Yearly"


class CategoryName(str, Enum):
    """Notion 'Categories' relation targets (page names in Categories DB)."""

    ENTERTAINMENT = "Entertainment"
    BUSINESS = "Business"
    EDUCATION = "Education"
    HEALTH = "Health"
    DECORATION = "Decoration"


# Hardcoded page IDs for category relation lookup.
# These are the page IDs in the Categories database that the
# 'Categories' relation property points to.
CATEGORY_PAGE_IDS: dict[CategoryName, str] = {
    CategoryName.ENTERTAINMENT: "30e3fef8-fd21-81f9-80fb-d236e6c97f5a",
    CategoryName.BUSINESS: "30e3fef8-fd21-81ca-9ffb-c9a4b1b1312b",
    CategoryName.EDUCATION: "30e3fef8-fd21-8130-8aed-c81e7f2b6fde",
    CategoryName.HEALTH: "30e3fef8-fd21-81c1-bd3e-e3200f9f5ce3",
    CategoryName.DECORATION: "30e3fef8-fd21-81a3-b5e9-e22db9b978b1",
}


# ═══════════════════════════════════════════════════════════════════════════
# Core Domain Model
# ═══════════════════════════════════════════════════════════════════════════


class SubscriptionRecord(BaseModel):
    """Validated subscription record matching Notion database schema.

    Invariants:
    - Active subscriptions must NOT have a canceled_date
    - Canceled subscriptions auto-set canceled_date to today if missing
    - Amount must be positive and rounded to cents
    - Name must be non-empty
    """

    name: str = Field(..., min_length=1, description="Service name")
    amount: float = Field(..., gt=0, description="Billing amount in USD")
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    cycle: BillingCycle = BillingCycle.MONTHLY
    subscribed_date: date = Field(default_factory=date.today)
    canceled_date: Optional[date] = None
    category: Optional[CategoryName] = None

    @model_validator(mode="after")
    def validate_status_date_consistency(self) -> SubscriptionRecord:
        """Enforce invariant: canceled_date ↔ status must be consistent."""
        if self.status == SubscriptionStatus.ACTIVE and self.canceled_date is not None:
            raise ValueError("canceled_date must be None when status is Active")
        if self.status == SubscriptionStatus.CANCELED and self.canceled_date is None:
            self.canceled_date = date.today()
        # Round amount to cents
        self.amount = round(self.amount, 2)
        return self

    @property
    def monthly_cost_estimate(self) -> float:
        """Estimate monthly cost from billing cycle."""
        divisors = {
            BillingCycle.MONTHLY: 1,
            BillingCycle.QUARTERLY: 3,
            BillingCycle.YEARLY: 12,
        }
        return round(self.amount / divisors[self.cycle], 2)

    @property
    def category_page_id(self) -> Optional[str]:
        """Look up the Notion page ID for this record's category."""
        if self.category is None:
            return None
        return CATEGORY_PAGE_IDS.get(self.category)

    def to_notion_properties(self) -> dict:
        """Serialize to Notion create-page property format.

        Uses the correct property names and types from the live database:
        - Name: title
        - Amount: number (dollar format)
        - Status: status type (NOT select)
        - Cycle: select
        - Subscribed Date / Canceled Date: date
        - Categories: relation (page ID lookup)
        - Next Billing / Monthly Expense / Lifetime Spent: formula (READ-ONLY)
        """
        props: dict = {
            "Name": self.name,
            "Amount": self.amount,
            "Status": self.status.value,
            "Cycle": self.cycle.value,
            "date:Subscribed Date:start": self.subscribed_date.isoformat(),
            "date:Subscribed Date:is_datetime": 0,
        }

        if self.canceled_date is not None:
            props["date:Canceled Date:start"] = self.canceled_date.isoformat()
            props["date:Canceled Date:is_datetime"] = 0

        # Categories is a relation — requires page ID, not a string
        if self.category_page_id is not None:
            props["Categories"] = self.category_page_id

        return props


# ═══════════════════════════════════════════════════════════════════════════
# LLM Extraction Intermediate Representation
# ═══════════════════════════════════════════════════════════════════════════


class EmailExtraction(BaseModel):
    """Permissive model for LLM-extracted data.

    This is intentionally loose — the LLM may return partial or uncertain
    data. The quality gate is the promotion to SubscriptionRecord.
    """

    service_name: str
    amount: Optional[float] = None
    billing_cycle: str = "monthly"
    is_cancellation: bool = False
    is_renewal: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    raw_subject: str = ""
    billing_date: Optional[str] = None

    def to_subscription_record(self) -> Optional[SubscriptionRecord]:
        """Promote to strict domain model. Returns None if data insufficient."""
        if not self.service_name or self.amount is None or self.amount <= 0:
            return None

        # Map cycle string to enum
        cycle_map = {
            "monthly": BillingCycle.MONTHLY,
            "quarterly": BillingCycle.QUARTERLY,
            "yearly": BillingCycle.YEARLY,
            "annual": BillingCycle.YEARLY,
        }
        cycle = cycle_map.get(self.billing_cycle.lower(), BillingCycle.MONTHLY)

        status = (
            SubscriptionStatus.CANCELED
            if self.is_cancellation
            else SubscriptionStatus.ACTIVE
        )

        # Parse billing_date if present
        subscribed_date = date.today()
        if self.billing_date:
            try:
                subscribed_date = date.fromisoformat(self.billing_date)
            except ValueError:
                pass

        try:
            return SubscriptionRecord(
                name=self.service_name,
                amount=self.amount,
                status=status,
                cycle=cycle,
                subscribed_date=subscribed_date,
            )
        except ValueError:
            return None


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Observability
# ═══════════════════════════════════════════════════════════════════════════


class PipelineResult(BaseModel):
    """Metrics from a pipeline run for observability."""

    extractions_attempted: int = 0
    extractions_succeeded: int = 0
    duplicates_skipped: int = 0
    records_created: int = 0
    records_updated: int = 0
    errors: list[str] = Field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.extractions_attempted == 0:
            return 0.0
        return self.extractions_succeeded / self.extractions_attempted

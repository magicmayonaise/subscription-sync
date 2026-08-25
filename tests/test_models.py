"""Tests for domain models, matcher, and pipeline logic.

Demonstrates:
- Property-based testing patterns
- Fixture composition
- Testing invariants (not just happy paths)
- Mocking external services for deterministic tests
"""

from __future__ import annotations

from datetime import date

import pytest

from subscription_sync.email_parser import _write_token_file, redact_sensitive_text
from subscription_sync.matcher import enrich_record, find_duplicate, infer_category
from subscription_sync.models import (
    BillingCycle,
    CategoryName,
    EmailExtraction,
    PipelineResult,
    SubscriptionRecord,
    SubscriptionStatus,
)


# ═══════════════════════════════════════════════════════════════════════════
# Domain Model Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSubscriptionRecord:
    """Test the core domain model validates correctly."""

    def test_valid_active_subscription(self) -> None:
        record = SubscriptionRecord(
            name="Netflix",
            amount=15.99,
            status=SubscriptionStatus.ACTIVE,
            cycle=BillingCycle.MONTHLY,
            subscribed_date=date(2024, 1, 15),
        )
        assert record.name == "Netflix"
        assert record.amount == 15.99
        assert record.canceled_date is None

    def test_canceled_subscription_auto_sets_date(self) -> None:
        """If status=Canceled but no canceled_date, it auto-sets to today."""
        record = SubscriptionRecord(
            name="Hulu",
            amount=7.99,
            status=SubscriptionStatus.CANCELED,
            cycle=BillingCycle.MONTHLY,
            subscribed_date=date(2023, 6, 1),
        )
        assert record.canceled_date == date.today()

    def test_active_with_canceled_date_raises(self) -> None:
        """Cannot have canceled_date when status is Active."""
        with pytest.raises(ValueError, match="canceled_date must be None"):
            SubscriptionRecord(
                name="Test",
                amount=10.0,
                status=SubscriptionStatus.ACTIVE,
                cycle=BillingCycle.MONTHLY,
                subscribed_date=date(2024, 1, 1),
                canceled_date=date(2024, 6, 1),
            )

    def test_amount_rounded_to_cents(self) -> None:
        record = SubscriptionRecord(
            name="Test",
            amount=9.999,
            cycle=BillingCycle.MONTHLY,
            subscribed_date=date(2024, 1, 1),
        )
        assert record.amount == 10.0

    def test_negative_amount_rejected(self) -> None:
        with pytest.raises(ValueError):
            SubscriptionRecord(
                name="Test",
                amount=-5.0,
                cycle=BillingCycle.MONTHLY,
                subscribed_date=date(2024, 1, 1),
            )

    def test_zero_amount_rejected(self) -> None:
        with pytest.raises(ValueError):
            SubscriptionRecord(
                name="Test",
                amount=0,
                cycle=BillingCycle.MONTHLY,
                subscribed_date=date(2024, 1, 1),
            )

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError):
            SubscriptionRecord(
                name="",
                amount=10.0,
                cycle=BillingCycle.MONTHLY,
                subscribed_date=date(2024, 1, 1),
            )

    def test_to_notion_properties_format(self) -> None:
        """Verify the Notion property dict uses correct field names."""
        record = SubscriptionRecord(
            name="Spotify",
            amount=9.99,
            cycle=BillingCycle.MONTHLY,
            subscribed_date=date(2024, 3, 15),
        )
        props = record.to_notion_properties()

        assert props["Name"] == "Spotify"
        assert props["Amount"] == 9.99
        assert props["Status"] == "Active"
        assert props["Cycle"] == "Monthly"
        assert props["date:Subscribed Date:start"] == "2024-03-15"
        assert props["date:Subscribed Date:is_datetime"] == 0

        # These should NOT be present (they're formulas)
        assert "Next Billing" not in props
        assert "Monthly Expense" not in props
        assert "Lifetime Spent" not in props

    def test_to_notion_properties_with_cancellation(self) -> None:
        record = SubscriptionRecord(
            name="Hulu",
            amount=7.99,
            status=SubscriptionStatus.CANCELED,
            cycle=BillingCycle.MONTHLY,
            subscribed_date=date(2023, 1, 1),
            canceled_date=date(2024, 6, 15),
        )
        props = record.to_notion_properties()
        assert props["Status"] == "Canceled"
        assert props["date:Canceled Date:start"] == "2024-06-15"

    def test_monthly_cost_estimate(self) -> None:
        monthly = SubscriptionRecord(
            name="A", amount=12.0, cycle=BillingCycle.MONTHLY, subscribed_date=date(2024, 1, 1)
        )
        quarterly = SubscriptionRecord(
            name="B", amount=30.0, cycle=BillingCycle.QUARTERLY, subscribed_date=date(2024, 1, 1)
        )
        yearly = SubscriptionRecord(
            name="C", amount=120.0, cycle=BillingCycle.YEARLY, subscribed_date=date(2024, 1, 1)
        )
        assert monthly.monthly_cost_estimate == 12.0
        assert quarterly.monthly_cost_estimate == 10.0
        assert yearly.monthly_cost_estimate == 10.0

    def test_category_page_id_lookup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NOTION_API_KEY", "test-notion-key")
        monkeypatch.setenv("NOTION_DATABASE_ID", "test-database-id")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        monkeypatch.setenv(
            "NOTION_CATEGORY_ENTERTAINMENT_PAGE_ID", "env-entertainment-page-id"
        )
        record = SubscriptionRecord(
            name="Netflix",
            amount=15.99,
            cycle=BillingCycle.MONTHLY,
            subscribed_date=date(2024, 1, 1),
            category=CategoryName.ENTERTAINMENT,
        )
        assert record.category_page_id == "env-entertainment-page-id"

    def test_category_page_id_none_when_no_category(self) -> None:
        record = SubscriptionRecord(
            name="Test",
            amount=10.0,
            cycle=BillingCycle.MONTHLY,
            subscribed_date=date(2024, 1, 1),
        )
        assert record.category_page_id is None


# ═══════════════════════════════════════════════════════════════════════════
# Email Extraction Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEmailExtraction:
    """Test the LLM extraction intermediate representation."""

    def test_promotion_to_record_succeeds(self) -> None:
        extraction = EmailExtraction(
            service_name="Netflix",
            amount=15.99,
            billing_cycle="monthly",
            is_cancellation=False,
            is_renewal=True,
            confidence=0.95,
            raw_subject="Your Netflix receipt",
            billing_date="2024-06-15",
        )
        record = extraction.to_subscription_record()
        assert record is not None
        assert record.name == "Netflix"
        assert record.amount == 15.99
        assert record.cycle == BillingCycle.MONTHLY

    def test_promotion_fails_without_amount(self) -> None:
        extraction = EmailExtraction(
            service_name="Unknown Service",
            amount=None,
            billing_cycle="monthly",
            is_cancellation=False,
            is_renewal=False,
            confidence=0.3,
            raw_subject="Something happened",
        )
        assert extraction.to_subscription_record() is None

    def test_unknown_cycle_defaults_to_monthly(self) -> None:
        extraction = EmailExtraction(
            service_name="SomeApp",
            amount=5.0,
            billing_cycle="unknown",
            is_cancellation=False,
            is_renewal=True,
            confidence=0.8,
            raw_subject="Payment received",
        )
        record = extraction.to_subscription_record()
        assert record is not None
        assert record.cycle == BillingCycle.MONTHLY

    def test_cancellation_sets_status(self) -> None:
        extraction = EmailExtraction(
            service_name="Hulu",
            amount=7.99,
            billing_cycle="monthly",
            is_cancellation=True,
            is_renewal=False,
            confidence=0.9,
            raw_subject="Your Hulu subscription has been canceled",
        )
        record = extraction.to_subscription_record()
        assert record is not None
        assert record.status == SubscriptionStatus.CANCELED
        assert record.canceled_date is not None


# ═══════════════════════════════════════════════════════════════════════════
# Matcher Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMatcher:
    """Test fuzzy matching and category inference."""

    def test_exact_match(self) -> None:
        record = SubscriptionRecord(
            name="Netflix",
            amount=15.99,
            cycle=BillingCycle.MONTHLY,
            subscribed_date=date(2024, 1, 1),
        )
        match = find_duplicate(record, ["Netflix", "Spotify", "Hulu"])
        assert match == "Netflix"

    def test_fuzzy_match(self) -> None:
        record = SubscriptionRecord(
            name="Adobe Photoshop",
            amount=22.99,
            cycle=BillingCycle.MONTHLY,
            subscribed_date=date(2024, 1, 1),
        )
        match = find_duplicate(record, ["Photoshop Adobe", "Canva", "Figma"])
        assert match == "Photoshop Adobe"

    def test_no_match_below_threshold(self) -> None:
        record = SubscriptionRecord(
            name="Completely Different Service",
            amount=10.0,
            cycle=BillingCycle.MONTHLY,
            subscribed_date=date(2024, 1, 1),
        )
        match = find_duplicate(record, ["Netflix", "Spotify"])
        assert match is None

    def test_case_insensitive_match(self) -> None:
        record = SubscriptionRecord(
            name="NETFLIX",
            amount=15.99,
            cycle=BillingCycle.MONTHLY,
            subscribed_date=date(2024, 1, 1),
        )
        match = find_duplicate(record, ["netflix", "Spotify"])
        assert match == "netflix"

    def test_infer_entertainment_category(self) -> None:
        assert infer_category("Netflix Premium") == CategoryName.ENTERTAINMENT
        assert infer_category("Spotify Family") == CategoryName.ENTERTAINMENT

    def test_infer_business_category(self) -> None:
        assert infer_category("Notion Plus") == CategoryName.BUSINESS
        assert infer_category("GitHub Pro") == CategoryName.BUSINESS

    def test_infer_education_category(self) -> None:
        assert infer_category("Coursera Plus") == CategoryName.EDUCATION

    def test_infer_returns_none_for_unknown(self) -> None:
        assert infer_category("Totally Random Service XYZ") is None

    def test_enrich_adds_category(self) -> None:
        record = SubscriptionRecord(
            name="Netflix",
            amount=15.99,
            cycle=BillingCycle.MONTHLY,
            subscribed_date=date(2024, 1, 1),
        )
        enriched = enrich_record(record)
        assert enriched.category == CategoryName.ENTERTAINMENT

    def test_enrich_preserves_existing_category(self) -> None:
        record = SubscriptionRecord(
            name="Netflix",
            amount=15.99,
            cycle=BillingCycle.MONTHLY,
            subscribed_date=date(2024, 1, 1),
            category=CategoryName.BUSINESS,  # Override
        )
        enriched = enrich_record(record)
        assert enriched.category == CategoryName.BUSINESS  # Unchanged


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Result Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineResult:
    def test_success_rate_calculation(self) -> None:
        result = PipelineResult(extractions_attempted=10, extractions_succeeded=7)
        assert result.success_rate == 0.7

    def test_success_rate_zero_division(self) -> None:
        result = PipelineResult()
        assert result.success_rate == 0.0


class TestRedactionAndTokenPermissions:
    def test_redacts_email_and_card_number(self) -> None:
        text = "Bill user@example.com card 4111-1111-1111-1111"
        redacted = redact_sensitive_text(text)
        assert "user@example.com" not in redacted
        assert "4111-1111-1111-1111" not in redacted
        assert "[REDACTED_EMAIL]" in redacted
        assert "[REDACTED_CARD]" in redacted

    def test_token_file_is_owner_read_write_only(self, tmp_path) -> None:
        token_path = tmp_path / "token.json"
        _write_token_file(str(token_path), '{"token":"secret"}')
        assert token_path.read_text() == '{"token":"secret"}'
        assert token_path.stat().st_mode & 0o777 == 0o600

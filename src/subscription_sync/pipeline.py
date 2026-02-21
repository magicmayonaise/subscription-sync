"""ETL pipeline orchestrator.

Demonstrates:
- Pipeline pattern with quality gates between stages
- Error isolation (one email failure doesn't abort the batch)
- Observability via PipelineResult metrics
- Confidence threshold as decision boundary (precision vs. recall tradeoff)
- Idempotent writes via dedup + upsert
"""

from __future__ import annotations

import structlog

from subscription_sync.config import Settings
from subscription_sync.email_parser import (
    extract_subscription_from_email,
    fetch_billing_emails,
)
from subscription_sync.matcher import enrich_record, find_duplicate
from subscription_sync.models import PipelineResult, SubscriptionRecord, SubscriptionStatus
from subscription_sync.notion_client import NotionSubscriptionRepo

logger = structlog.get_logger()


def run_pipeline(
    settings: Settings,
    dry_run: bool = False,
    lookback_days: int | None = None,
) -> PipelineResult:
    """Execute the full Extract → Transform → Load pipeline.

    Stages:
    1. EXTRACT: Fetch billing emails from Gmail
    2. TRANSFORM: Use Claude tool_use to extract structured data
    3. VALIDATE: Apply confidence threshold + fuzzy dedup
    4. LOAD: Upsert to Notion (skipped in dry_run mode)
    """
    result = PipelineResult()
    repo = NotionSubscriptionRepo(settings)

    try:
        # ── Stage 1: Extract ──────────────────────────────────────────
        logger.info("pipeline_stage", stage="extract")
        emails = fetch_billing_emails(settings, lookback_days)

        if not emails:
            logger.info("no_emails_found")
            return result

        # ── Stage 2: Transform ────────────────────────────────────────
        logger.info("pipeline_stage", stage="transform", email_count=len(emails))
        extractions: list[SubscriptionRecord] = []

        for email in emails:
            result.extractions_attempted += 1
            extraction = extract_subscription_from_email(email, settings)

            if extraction is None:
                continue

            # Quality gate: confidence threshold
            if extraction.confidence < settings.confidence_threshold:
                logger.debug(
                    "low_confidence_skip",
                    service=extraction.service_name,
                    confidence=extraction.confidence,
                )
                continue

            # Promote to strict domain model
            record = extraction.to_subscription_record()
            if record is None:
                result.errors.append(
                    f"Failed to promote: {extraction.service_name}"
                )
                continue

            result.extractions_succeeded += 1
            extractions.append(record)

        logger.info(
            "extraction_complete",
            attempted=result.extractions_attempted,
            succeeded=result.extractions_succeeded,
        )

        # ── Stage 3: Validate (Dedup + Enrich) ───────────────────────
        logger.info("pipeline_stage", stage="validate")
        existing_names = repo.get_existing_subscription_names() if not dry_run else []

        records_to_write: list[SubscriptionRecord] = []
        for record in extractions:
            # Enrich with category
            record = enrich_record(record)

            # Check for duplicates
            match = find_duplicate(record, existing_names)
            if match and record.status == SubscriptionStatus.ACTIVE:
                logger.debug("duplicate_skipped", name=record.name, matched=match)
                result.duplicates_skipped += 1
                continue

            records_to_write.append(record)

        # ── Stage 4: Load ─────────────────────────────────────────────
        if dry_run:
            logger.info(
                "dry_run_results",
                would_write=len(records_to_write),
                records=[r.name for r in records_to_write],
            )
            return result

        logger.info("pipeline_stage", stage="load", count=len(records_to_write))
        for record in records_to_write:
            try:
                page_id = repo.upsert_subscription(record)
                result.records_created += 1
                logger.info("record_written", name=record.name, page_id=page_id)
            except Exception as e:
                result.errors.append(f"{record.name}: {e}")
                logger.error("write_error", name=record.name, error=str(e))

    finally:
        repo.close()

    logger.info(
        "pipeline_complete",
        success_rate=result.success_rate,
        created=result.records_created,
        errors=len(result.errors),
    )
    return result


def add_manual_subscription(
    settings: Settings,
    record: SubscriptionRecord,
) -> str:
    """Add a single subscription directly (bypasses email scanning)."""
    repo = NotionSubscriptionRepo(settings)
    try:
        record = enrich_record(record)
        page_id = repo.upsert_subscription(record)
        logger.info("manual_add_complete", name=record.name, page_id=page_id)
        return page_id
    finally:
        repo.close()

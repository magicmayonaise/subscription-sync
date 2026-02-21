"""Fuzzy deduplication and category inference.

Demonstrates:
- Fuzzy string matching with rapidfuzz (Levenshtein-based)
- Token sort ratio for word-reordering tolerance
- Keyword-based heuristic classifier (upgrade path: embeddings)
- Threshold-based decision boundary (analogous to classifier confidence)
"""

from __future__ import annotations

from rapidfuzz import fuzz

from subscription_sync.models import CategoryName, SubscriptionRecord

# Minimum similarity score (0-100) to consider a match
MATCH_THRESHOLD = 85

# ═══════════════════════════════════════════════════════════════════════════
# Fuzzy Duplicate Detection
# ═══════════════════════════════════════════════════════════════════════════


def find_duplicate(
    record: SubscriptionRecord,
    existing_names: list[str],
    threshold: int = MATCH_THRESHOLD,
) -> str | None:
    """Find a fuzzy match for a subscription name in existing entries.

    Uses token_sort_ratio which handles word reordering:
    "Adobe Photoshop" ≈ "Photoshop Adobe" → high score

    Returns the matched existing name, or None if no match above threshold.
    """
    if not existing_names:
        return None

    best_match: str | None = None
    best_score: float = 0.0

    for name in existing_names:
        score = fuzz.token_sort_ratio(
            record.name.lower(),
            name.lower(),
        )
        if score > best_score:
            best_score = score
            best_match = name

    if best_score >= threshold:
        return best_match

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Category Inference (Keyword Heuristic)
# ═══════════════════════════════════════════════════════════════════════════

CATEGORY_KEYWORDS: dict[CategoryName, list[str]] = {
    CategoryName.ENTERTAINMENT: [
        "netflix", "spotify", "hulu", "disney", "hbo", "paramount",
        "peacock", "apple tv", "youtube", "crunchyroll", "audible",
        "kindle", "xbox", "playstation", "nintendo", "steam", "twitch",
    ],
    CategoryName.BUSINESS: [
        "notion", "slack", "github", "gitlab", "jira", "confluence",
        "figma", "canva", "adobe", "photoshop", "dropbox", "zoom",
        "microsoft 365", "office 365", "google workspace", "linear",
        "vercel", "aws", "heroku", "digitalocean", "cloudflare",
        "1password", "lastpass", "grammarly",
    ],
    CategoryName.EDUCATION: [
        "coursera", "udemy", "skillshare", "masterclass", "linkedin learning",
        "pluralsight", "datacamp", "codecademy", "brilliant", "oreilly",
        "safari", "springboard", "deeplearning.ai", "scrimba",
    ],
    CategoryName.HEALTH: [
        "headspace", "calm", "peloton", "strava", "fitbit", "noom",
        "myfitnesspal", "whoop", "oura",
    ],
    CategoryName.DECORATION: [
        "pinterest", "houzz", "wayfair",
    ],
}


def infer_category(service_name: str) -> CategoryName | None:
    """Infer subscription category from service name using keyword matching.

    Simple heuristic classifier. Production upgrade path:
    - Embedding similarity against category exemplars
    - Fine-tuned text classifier
    - LLM-based classification (already available via the extraction step)
    """
    name_lower = service_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in name_lower:
                return category
    return None


def enrich_record(record: SubscriptionRecord) -> SubscriptionRecord:
    """Add inferred category if not already set."""
    if record.category is not None:
        return record

    inferred = infer_category(record.name)
    if inferred is not None:
        return record.model_copy(update={"category": inferred})

    return record

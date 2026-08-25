"""CLI interface using Typer.

Demonstrates:
- Modern CLI with Typer (type-hint driven argument parsing)
- Rich console output for human-readable feedback
- Separation of CLI concerns from business logic
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from subscription_sync.config import get_settings
from subscription_sync.models import (
    BillingCycle,
    CategoryName,
    SubscriptionRecord,
)

app = typer.Typer(
    name="subsync",
    help="Subscription tracker: Gmail → LLM extraction → Notion sync",
    no_args_is_help=True,
)
console = Console()


@app.command()
def scan(
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without writing to Notion"),
    lookback: Optional[int] = typer.Option(None, "--lookback", help="Days to look back in Gmail"),
) -> None:
    """Scan Gmail for subscription emails and sync to Notion."""
    from subscription_sync.pipeline import run_pipeline

    settings = get_settings()
    console.print("[bold blue]📨 Scanning Gmail for subscription emails...[/]")

    result = run_pipeline(settings, dry_run=dry_run, lookback_days=lookback)

    # Display results
    table = Table(title="Pipeline Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Emails processed", str(result.extractions_attempted))
    table.add_row("Extractions succeeded", str(result.extractions_succeeded))
    table.add_row("Success rate", f"{result.success_rate:.0%}")
    table.add_row("Duplicates skipped", str(result.duplicates_skipped))
    table.add_row("Records created", str(result.records_created))
    table.add_row("Records updated", str(result.records_updated))
    table.add_row("Errors", str(len(result.errors)))

    console.print(table)

    if result.errors:
        console.print("\n[bold red]Errors:[/]")
        for error in result.errors:
            console.print(f"  • {error}")


@app.command()
def add(
    name: str = typer.Argument(..., help="Service name"),
    amount: float = typer.Option(..., "--amount", "-a", help="Billing amount in USD"),
    cycle: str = typer.Option("monthly", "--cycle", "-c", help="monthly|quarterly|yearly"),
    category: Optional[str] = typer.Option(None, "--category", help="Category name"),
) -> None:
    """Manually add a subscription to Notion."""
    from subscription_sync.pipeline import add_manual_subscription

    settings = get_settings()

    cycle_map = {
        "monthly": BillingCycle.MONTHLY,
        "quarterly": BillingCycle.QUARTERLY,
        "yearly": BillingCycle.YEARLY,
    }

    cat = None
    if category:
        try:
            cat = CategoryName(category)
        except ValueError:
            console.print(f"[red]Unknown category: {category}[/]")
            console.print(f"Valid: {', '.join(c.value for c in CategoryName)}")
            raise typer.Exit(1)

    record = SubscriptionRecord(
        name=name,
        amount=amount,
        cycle=cycle_map.get(cycle.lower(), BillingCycle.MONTHLY),
        subscribed_date=date.today(),
        category=cat,
    )

    page_id = add_manual_subscription(settings, record)
    console.print(f"[green]✓ Added {name} (${amount}/{cycle}) → {page_id}[/]")


@app.command()
def auth() -> None:
    """Authenticate with Gmail (run once to set up OAuth)."""
    from subscription_sync.email_parser import authenticate_gmail

    settings = get_settings()
    console.print("[bold blue]🔑 Starting Gmail OAuth flow...[/]")
    authenticate_gmail(settings)
    console.print("[green]✓ Gmail authentication complete![/]")


@app.command()
def status() -> None:
    """Check configuration and Notion connection."""
    from subscription_sync.notion_client import NotionSubscriptionRepo

    settings = get_settings()

    console.print("[bold]Configuration:[/]")
    console.print(f"  Notion DB: ...{settings.notion_database_id[-4:]}")
    console.print(f"  Lookback: {settings.lookback_days} days")
    console.print(f"  Confidence threshold: {settings.confidence_threshold}")

    repo = NotionSubscriptionRepo(settings)
    if repo.test_connection():
        names = repo.get_existing_subscription_names()
        console.print(f"  [green]✓ Notion connected ({len(names)} subscriptions)[/]")
    else:
        console.print("  [red]✗ Notion connection failed[/]")
    repo.close()


if __name__ == "__main__":
    app()

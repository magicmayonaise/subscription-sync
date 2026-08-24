# subscription-sync

Gmail → LLM extraction → Notion subscription tracker.

## Secrets

Never commit `.env`, `token.json`, or `credentials.json`. Copy `.env.example` to `.env` and keep those files local.

## Gmail access

The Gmail OAuth scope must stay `gmail.readonly`. Do not request write or broader mail scopes.

## Email bodies

Fetched email bodies are sent to Claude for extraction. Obvious card numbers and email addresses are redacted before that send.

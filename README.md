# Anchor

Anchor is a multi-tenant AI customer-support platform: companies upload their help docs, and Anchor's AI agent answers their customers' tickets. It looks up the right documents, takes safe actions such as an order lookup or a refund with guardrails, and hands the ticket to a human with full context when it should.

You build it, one lesson at a time, in **[CodeClear's AI Engineering course](https://codeclear.dev)**. This repository is the course's reference code.

## Follow along lesson by lesson

Every lesson is one commit with a tag:

```sh
git tag --list 'lesson-*'                  # every lesson, in order
git checkout lesson-007-anchor-s-ai-service-boundary-calling-the-python   # the project exactly as that lesson leaves it
```

To see what one lesson added, compare its tag with the one before it:

```sh
git diff lesson-006-llm-foundations-transformer-architecture-and-tok lesson-007-anchor-s-ai-service-boundary-calling-the-python
```

Or open the same comparison on GitHub: `https://github.com/sandeepmvl/anchor/compare/<earlier-tag>...<later-tag>`.

## What is where

| Folder | What it holds |
|---|---|
| `app/`, `config/`, `db/`, `test/` | The Rails product: tenants, tickets, conversations, admin, the chat API |
| `anchor_ai/` | The Python AI service Rails calls: retrieval, the agent and its tools, guardrails (tests in `anchor_ai/tests/`) |
| `eval/` | Golden datasets, judges and the evaluation gate CI runs |
| `ops/` | Deploy, load tests, backups, runbooks |
| `labs/<lesson>/` | Small exercise scripts a single lesson uses to explore an idea |

## Run it

Everything runs offline by default: model and embedding calls go through a deterministic fake, so you need no API key to follow the course.

```sh
bundle install && bin/rails db:prepare
bin/rails test                                   # the Rails suite
python3 -m pytest -q anchor_ai/tests             # the AI service suite
```

To call a real model, set `ANCHOR_LLM_PROVIDER` and your provider key, as the lessons show.

## License

MIT. See [LICENSE](LICENSE).

# Labeling guide

Context: messages that customers send to the support channel of a SaaS project-management product (Free / Pro / Business / Enterprise plans, billed per seat).

> This is the English translation. The prompts sent to the LLM generator and annotator use the original Vietnamese version, [`LABELING_GUIDE.vi.md`](LABELING_GUIDE.vi.md), which produced the cached labels.

| Label | Definition | Example (Vietnamese data, with translation) |
|---|---|---|
| `pricing` | Questions about prices, plans, quotas, discounts, payment methods, quotes **before buying / upgrading**. | "Gói Pro có bao nhiêu seat?" ("How many seats does the Pro plan have?") |
| `complaint` | Dissatisfaction, reporting a mistake (wrong charge, slow support, an incident that caused damage), demanding a refund / compensation — **without** asking to end the service. | "Sao tháng này bị trừ tiền 2 lần vậy?" ("Why was I charged twice this month?") |
| `cancellation` | Wants to cancel, downgrade, remove seats, pause, turn off auto-renewal, delete / close the account; or asks about the procedure or consequences of cancelling. | "Làm sao để tắt tự động gia hạn?" ("How do I turn off auto-renewal?") |
| `tech_support` | How-to questions, bug reports, login, integrations, API — neutral tone, needs guidance or a fix. | "API trả về lỗi 401 dù token còn hạn" ("The API returns 401 even though the token is valid") |
| `other` | Greetings, thanks, feature suggestions, jobs, partnerships, off-topic, gibberish. | "Cảm ơn nhiều nhé" ("Thanks a lot") |

## Tie-break rules

1. **Any intent to cancel → `cancellation`**, even when combined with a complaint ("Dịch vụ tệ quá, hủy luôn cho tôi" — "The service is awful, just cancel it for me").
2. **Technical problems**: neutral tone, needs guidance → `tech_support`; the focus is frustration / holding someone accountable → `complaint`.
3. **Money**: asking about prices before buying → `pricing`; charged incorrectly / unexpected fee → `complaint`.
4. Feature suggestions and praise → `other`.

# Hourly forecast reliability

## November 10, 2026 audit

Before the repair, GitHub listed 51 executions of `hourly.yml`: 45 scheduled and six manual. Scheduled runs had 39 successes, five failures and one setup-time skip. Manual runs had five successes and one failure. The archive contained 43 unique issue hours; repeated executions within an hour reuse the immutable saved forecast.

The main loss occurred before job creation: scheduled starts were several hours apart despite the configured hourly/half-hourly cron. GitHub documents that scheduled events can be delayed or dropped. Changing the cron minute is not a guarantee of hourly execution.

The November 1–2 and November 8 failures include missing recent observations. Recovery previously began at the latest complete observation, which could skip internal holes. Recovery now scans the model's required history (SARIMA state cutoff and at least seven days), stores received observations before prediction, and retries the job twice after 60 seconds. No missing readings are fabricated. No past forecast issues are backdated or reconstructed.

## Independent clock

`gridcast-scheduler` runs on Cloudflare at minutes 12, 27, 42 and 57 UTC. It reads the saved forecast pointer. If the current UTC hour already has a forecast, it does nothing. Otherwise it checks for a recently queued/running forecasting job and dispatches `hourly.yml` on `main` when needed. GitHub retains inference, its shared single-writer concurrency group and R2 secrets. The archive workflow refreshes after successful inference.

There is no public dispatch endpoint. `/health` reports configuration and the latest timer status; diagnostics are stored at `state/scheduler.json`. Dispatch errors are logged without credentials. GitHub's own cron remains a fallback. This improves trigger reliability but is not an hourly SLA: GitHub runner delays, source outages and model/data validation can still prevent a forecast.

## Required credential

The Worker and timer are deployed, but cannot dispatch until its secret is supplied:

1. Create a fine-grained GitHub personal access token restricted to `tp-duarte/aemo-gridcast`, with repository **Actions: Read and write**. Set an expiration and arrange renewal.
2. Cloudflare → Workers & Pages → `gridcast-scheduler` → Settings → Variables and Secrets → Add **Secret** named `GITHUB_DISPATCH_TOKEN`.
3. Save/deploy. Do not put the token in code, GitHub files, logs or chat.
4. After the next timer tick, verify `/health`, the GitHub workflow run, the new immutable forecast issue, and the dashboard timestamp. `dispatched` only confirms acceptance by GitHub; it is not proof of a saved forecast.

The scheduler needs no paid compute resource. It makes at most 96 checks per day, with fewer dispatches when an hourly forecast is already present.

References: [GitHub schedule limitations](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule), [workflow dispatch permissions](https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event), [Cloudflare Cron Triggers](https://developers.cloudflare.com/workers/configuration/cron-triggers/).


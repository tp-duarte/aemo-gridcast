# Hourly forecasts with Cloudflare and GitHub Actions

Prepared 29 October 2026. No account resources were created, no credentials were requested, and nothing was deployed.

## Architecture prepared locally

1. Cloudflare Pages can serve the existing static `site/` portfolio.
2. GitHub Actions runs `gridcast.hourly hourly` at minute 17 each hour. It downloads recent AEMO observations, loads fitted models, creates a forecast version, evaluates earlier forecasts, and refreshes read-only API files.
3. Cloudflare R2 holds private model files, actuals, immutable forecast/evaluation archives, and replaceable dashboard snapshots. The bucket must remain private.
4. A small Cloudflare Worker serves only whitelisted public dashboard JSON/CSV and a freshness health check. It never trains or runs LightGBM/SARIMA.
5. A separate monthly training workflow fits the already-selected configurations. Both workflows share one concurrency group. There must be no second writer from a different repository or machine.

Training and inference are now separate code paths. LightGBM is stored in its native text format. SARIMA stores fitted coefficients plus its final state and covariance, not a pickle containing years of training arrays. Training-reference observations are a separate compressed Parquet file for drift analysis. A model ID links every saved forecast to its fitted models. The serving format contains no executable pickle.

The experimental hourly bundle includes weighted seasonal, selected LightGBM and selected SARIMA. Ridge remains available in the original runtime and historical dashboard but is not part of this deployment bundle. This tuning experiment does not include weather because its archived runs do not cover the multi-year validation period.

## Cadence and latency

Hourly means a new scheduled issue every hour, not an hourly retrain. The batch starts from the latest complete observation hour, predicts 24 hours, and discards already-elapsed target intervals before saving. Therefore the published future portion is usually about 23–24 hours; it is not the old 48-hour demonstration outlook.

GitHub explicitly says scheduled runs can be delayed or dropped under load. The minute-17 schedule avoids the busiest top-of-hour slot but is not an SLA. Public scheduled workflows can be disabled after 60 days without repository activity. A truly strict hourly deadline would need a scheduler/compute service with stronger guarantees.

The connector requests enough recent files to recover delayed runs, capped at 288 reports (roughly a day if one report arrives each five minutes). Directory retention and availability can shorten that coverage. Missing recent demand or missing SARIMA state updates blocks forecasting rather than silently bridging gaps. Longer outages require archive backfill. The Worker returns HTTP 503 if the latest forecast is more than 90 minutes old. A model older than 35 days is not used.

## Storage behavior

- `models/<model_id>/`: immutable compact models, calibration and training reference. `models/current.json` is updated only after the complete model bundle is uploaded.
- `forecasts/YYYY-MM-DD/<hour-model_id>.parquet`: immutable forecasts; retrying the same hour reuses the saved version.
- `evaluations/YYYY-MM-DD/<received-time>.parquet`: newly scored forecasts and later actual revisions; prior forecasts never change.
- `observations/YYYY-MM-DD/<received-time>.parquet`: received source observations.
- `state/`: replaceable actuals and seven-day forecast/score read snapshots, not the permanent forecast archive.
- `public/snapshots/`: derived dashboard JSON/CSV. A pointer is committed after all six regional/NEM files exist. Only derived snapshots older than four hours are pruned. Forecasts, evaluations, observations and models are not deleted by that cleanup.

The R2 API's overwrite semantics and a single writer are assumed. Forecast immutability is checked in the adapter; it is not a cross-repository distributed lock. Crash recovery reloads missing recent immutable forecast chunks. The public Worker cannot read arbitrary object paths through its URL. Do not enable public bucket access or commit credentials.

FAV uses matched evaluated live forecasts over the recent seven-day read window. It is not silently relabeled as the old historical backtest. Drift compares saved training observations with new observations, including demand, seasonal residual and ramp distributions; expensive density diagnostics refresh every six hours with their actual comparison-end timestamp. Interval widths use past forecast errors by region/model/lead bucket. First deployment needs a calibration file with no future observations; later training uses recent evaluated forecasts. Nominal 90% bands are not a coverage guarantee.

## Free-tier budget

Official documentation checked 29 October 2026:

- [R2 Standard pricing](https://developers.cloudflare.com/r2/pricing/): 10 GB-month included storage, 1 million Class A and 10 million Class B operations, no internet egress charge. This is retained average storage, not 10 new GB added each month. Having a free Cloudflare account does not confirm that R2 is activated or that billing requirements have been satisfied.
- [Workers limits](https://developers.cloudflare.com/workers/platform/limits/): Free CPU limits make the native forecasting pipeline unsuitable for the read Worker. Its task is streaming precomputed objects only.
- [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions): standard public-repository runners are free. GitHub Free private repositories currently have $12/month of shared included compute, equivalent to about 2,000 minutes at the standard Linux $0.006/minute rate. Other account usage draws from that allowance too. Actions artifact and cache storage have separate limits.

For 720 hourly runs/month, two billed minutes/run is $8.64 of included compute before training or other account usage. Three minutes/run is $12.96. These are scenarios, not measured GitHub timings. Dependency installation, downloads, source retries, scoring and drift generation must be included when measuring real runs. The model-only local benchmark excludes those steps.

Three models × six series × up to 96 targets × 720 hours produces up to 1,244,160 forecast rows/month (elapsed targets are omitted in this implementation). Immutable Parquet chunks plus small model files avoid hourly full-database backups. Measure dashboard-cache and revision volume as well as forecast rows before setting retention. No claim of a guaranteed $0 bill is made; set account budgets/alerts before enabling a schedule.

## Activation steps for later review

1. Choose the GitHub repository and whether code will be public or private. The repository root should contain this project's Python package and requirements files, not the surrounding workspace/private research directory.
2. Enable/create a private R2 bucket and a bucket-scoped read/write token. Put `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` in GitHub Actions secrets. Put `R2_ENDPOINT` and `R2_BUCKET` in repository variables. Never put tokens in portfolio JavaScript.
3. Backfill actuals to the present and verify continuity. The research sample ends August 2026 and cannot bootstrap live forecasts as-is. Fit a fresh compact bundle with the selected settings, attach past-only calibration, and use `python -m gridcast.hourly bootstrap --data <fresh-wide.parquet> --models <bundle>` once. It refuses stale data and refuses to overwrite an existing state.
4. Copy the selected-configuration JSON to `deployment/selected-models.json`. Copy the prepared YAML templates from `deployment/github/` into `.github/workflows/`. Both hourly and monthly workflows must use the same concurrency group. Review/approve action versions for your repository policy.
5. Configure the bucket binding and exact portfolio origin in `deployment/cloudflare/wrangler.jsonc`; deploy the read Worker only after account approval. No API token is required by the hourly job to deploy a Worker because it only writes R2 objects.
6. Set the site's public `apiBase` to the Worker URL and `forecastMode` to `live`. Deploy the frontend through the chosen Cloudflare Pages project. Test CORS, health, CSV export and a full hourly cycle before setting `GRIDCAST_ENABLED=true`.

The templates are stored outside `.github/workflows/` and additionally gated by `GRIDCAST_ENABLED`. They are not active merely because these files exist. No repository, bucket, workflow or site was created remotely.

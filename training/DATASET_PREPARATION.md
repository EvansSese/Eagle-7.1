# Chronological feature datasets

Model Lab's preparation step joins every season's fixture table to goals,
points, xG, rho, fair odds, outcome probability, power ratings and H2H tables
using `fixture_id`. Every feature column is retained and named
`<source_table_suffix>__<column>`, with hyphens replaced by underscores.
Fixture identifiers, dates, teams, scores, status and result are retained once
as metadata and targets, not duplicated from feature tables. Scores and result
must never be used as input features. Historical missing feature values remain
NULL and are logged; any imputation must be fitted on training data only.

The latest season with a fixture table in the country database is the current
season, and every fixture in it (played or unplayed) is saved to
`<league>-prediction-dataset`. Earlier seasons contribute only completed
fixtures with valid scores. Excluded counts are logged and recorded in
`<league>-dataset-metadata`. Historical fixtures are sorted by kickoff, then
fixture ID, and split at `floor(0.70 * n)` and `floor(0.85 * n)` into
`<league>-training-dataset`, `<league>-validation-dataset` and
`<league>-test-dataset`. Simultaneous kickoffs can cross a boundary. This can
allow information shared across simultaneous fixtures into both subsets, so
evaluation should consider this limitation.

Fixture dates used for processing and training are normalized to UTC without
a timezone suffix. Legacy timezone-naive timestamps are interpreted as UTC,
matching the stored API fixture dates; timezone-aware timestamps are converted
to UTC. No fixture tables are rewritten by this normalization.

All expected feature tables must exist and have consistent feature schemas
across seasons. Missing tables, duplicate feature IDs and schema mismatches
fail explicitly. A missing feature row remains NULL after the left join and
is logged; feature rows not present in the fixture table are ignored and
logged. Tables are replaced together transactionally in the datasets database,
and fixture IDs are uniquely indexed.

The xG league baselines and rho estimates now exclude earlier-round fixtures
whose kickoff is simultaneous with or later than the target fixture. Existing
xG, rho and fair-odds tables created before this change must be reprocessed,
then the datasets re-prepared, before training. This applies to every league,
not just England.

Model Lab's **Train market models** action consumes these splits. The legacy
training and prediction pipelines use a different schema and remain blocked
for this dataset version. See `training/MARKET_MODELS.md`.

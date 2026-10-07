# Data provenance and dictionary

## Source and usage

Source: [dunnhumby Source Files — Breakfast at the Frat](https://www.dunnhumby.com/source-files/).
The download contains a workbook and a user guide. The publisher describes it as a representation of retail sales and promotion data designed for classroom projects and case studies, not untouched transaction logs. The accompanying guide states © 2023 dunnhumby / All rights reserved. This project does not assign an open-source license to the dataset. Obtain the source from the publisher and review its terms before redistribution or a different use. Raw data is excluded from version control.

The workbook dates, rather than the broad glossary description, define the period: 2009-01-14 to 2012-01-04, 156 weekly endings. The next simulated week ends 2012-01-11. Currency is source dollars; no conversion is performed.

The guide describes five products from three brands across four categories. The actual lookup contains 58 UPCs; 55 appear in valid observations. The store lookup contains 79 rows representing 77 stores. Source observations total 524,950.

## Normalized observations

Grain: one product × store × week. Duplicate keys are rejected rather than summed.

| Field | Meaning and treatment |
|---|---|
| `product_id` | UPC preserved as an identifier string |
| `store_id` | Source store number, string |
| `week` | Source week-ending date, not week-start |
| `units` | Nonnegative integer units sold |
| `revenue` | Source `SPEND`; observed dollars, not a model prediction |
| `price` | Actual shelf price; finite and positive |
| `base_price` | Reference base price; finite and positive |
| `feature` | Binary in-store circular indicator |
| `display` | Binary in-store display indicator |
| `tpr_only` | Binary temporary reduction with shelf tag only; exclusive of feature/display |
| `visits`, `hhs` | Retained in processed data for exploration, excluded from predictive features |

The last two fields describe contemporaneous purchases and would leak demand information if used for prediction. Base price is used for discount calculations, not alongside redundant discount variables in the regression.

Product metadata provides description, manufacturer, category and package size. Store metadata provides name, location and segment. Stores 4503 and 17627 each appear with two different segment labels; both are labeled `CONFLICTING SOURCE VALUES`. Segment is not used to fit the model.

## Validation policy

- Missing keys, nonfinite values, nonpositive prices, negative units/spend, fractional units and malformed support flags are quarantined. Missing required columns and duplicate observation keys stop ingestion.
- The current source has 24 invalid shelf-price rows and 185 invalid base-price rows, excluding 209 rows in total. No valid zero-sales rows remain in this release. Zero-sales fixtures are tested and retained when prices are valid, but excluded from log fitting.
- Missing product/store weeks are counted over the full calendar for observed pairs, not imputed as zero. The count includes changing assortment and unobserved availability, not solely reporting failures.
- Price above base price is reported but not automatically removed. The source contains such observations.
- Unusually large valid purchases remain; no outcome-driven outlier removal is performed. Explore units per visit and visits per household in notebook 01.
- Source spend is retained even if a future release differs from rounded shelf price × units.

`reports/data_quality.json` records counts, exceptions and the workbook SHA-256 prefix. Rejected records, including reasons, are stored separately in `data/processed/rejected.parquet`. The original archive and workbook remain unchanged.

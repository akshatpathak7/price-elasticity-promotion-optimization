# Five-minute interview demonstration

Start the API and dashboard using the README. Run `uv run --no-editable retail-pricing demo` after setup to create a reproducible example. `reports/demo_result.json` contains the actual inputs, predictions, model version and saved scenario ID. See `docs/results.md` for the product/store from the measured run.

## Minute 1: Business problem

“Retailers need to decide whether a markdown brings enough additional demand to justify its lower price. I built a tool that estimates historical price sensitivity and compares supported price and promotion scenarios for one product in one store.”

Open **Sales overview**. Show the sales history, reference versus shelf price, and source dates. Explain that absent observations are not automatically zero sales. Revenue is observed spend, not fitted revenue.

## Minute 2: Explain elasticity

Open **Elasticity & promotions**. Explain the sign and confidence interval of the selected product's coefficient. A coefficient of -1.5 means a 1% price increase is associated with approximately 1.5% lower units under the model. Store intercepts capture persistent store differences, while calendar and promotion terms adjust for observable conditions.

Be ready to explain why `exp(predicted log sales)` needs a smearing correction. The log-scale fit does not directly estimate average sales on the original scale.

## Minute 3: Compare a scenario

Open **Scenario simulator**. Compare the baseline with a supported price and promotion from `reports/demo_result.json`. Show units, revenue, assumptions and the curve. The baseline uses the latest observed price/support but predicts the same future week as the proposed scenario.

Explain the separate support-lift figure: it changes promotion support while holding price fixed. It is still an observational model contrast, not a controlled experiment. Export the CSV and show that it includes assumptions and the model version.

## Minute 4: Recommend responsibly

Evaluate candidate prices. Explain the 80–110% grid, observed price range, discount cap, and optional margin constraint. Enter a deliberately high cost to show the infeasible state. Select a product with an uncertain estimate to show why the application can decline to recommend.

“The selected price maximizes predicted revenue among feasible tested candidates. A constant-elasticity model often chooses a boundary, so I do not call this the globally optimal price.”

## Minute 5: Show evidence and limitations

Open **Model validation**. Explain 104/26/26 chronological splitting, train-only preprocessing, and the rolling baseline. On the measured run, the interpretable regression outperforms XGBoost on the final test period. The challenger remains in the report rather than being hidden.

“Good prediction at historical prices does not prove a pricing intervention will work. Unobserved inventory and promotion timing can bias elasticity. Before applying this commercially, I would propose a controlled pricing experiment.”

## Questions to practice

- **Why not a random split?** It mixes time periods and can leak future patterns into training.
- **Why not use spend or households as predictors?** They reflect contemporaneous purchases, which are unavailable when making the decision.
- **Why regression for scenarios?** Its elasticity is interpretable and its behavior under a price change is explicit. Predictive performance alone does not justify counterfactual XGBoost predictions.
- **Why PostgreSQL and an API?** They separate durable data/scenario records from the interface and let the same decision engine serve notebooks and a dashboard.
- **Why no integer-programming library?** One product has only a small finite candidate list. Enumerating it is sufficient and easier to audit.
- **What would you improve next?** Better inventory controls, richer seasonality, jointly supported price/promotion cells, and a prospective evaluation of policy impact.

Walk through `data.py`, `modeling.py`, and `service.py` in that order when discussing the implementation. Focus on the assumptions and tests, not memorizing library syntax.

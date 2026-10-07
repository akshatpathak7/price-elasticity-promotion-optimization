# Methodology

## 1. Explain the business decision

The objective is expected revenue for one product/store in the next historical week. Optional cost assumptions make contribution profit and margin visible. The model cannot infer cost, stock availability, competitor response or cannibalization from fields it does not have.

## 2. Fit an interpretable demand curve

For each product, fit:

`log(units) = store intercept + elasticity × log(price) + promotion terms + annual sine/cosine + linear trend + error`

The store intercept adjusts for persistent store demand differences. Annual sine/cosine terms use elapsed calendar weeks and a 52.18-week period. The trend is elapsed weeks / 52. Each product shares one elasticity across its stores. There is no price-by-store interaction in this version.

Only positive sales enter the regression. The training residual smearing factor, `mean(exp(residual))`, corrects the simple exponentiation of fitted log demand. In sparse series this remains a conditional-positive-sales model rather than a model of whether any sale occurs. No valid zero-sales rows remain in the supplied cleaned release, but the code reports exclusions for other inputs.

Constant promotion columns are excluded, and rank deficiency is reported. Standard errors are clustered by store to account for within-store dependence. They do not address every common weekly shock, endogenous pricing or omitted confounder. Scenario intervals exponentiate the linear predictor's approximate 95% interval and apply the fitted smearing factor. They are conditional-mean intervals, not future-sales prediction intervals; smearing-factor uncertainty is omitted.

At fixed price, support lift is `100 × (Q_with_support / Q_without_support - 1)`. The simulator compares the proposed combined price/support change with the latest observed price/support under identical next-week conditions. This avoids comparing a future prediction directly with a past observed sale and calling the difference causal lift.

## 3. Evaluate without temporal leakage

- Train: first 104 calendar weeks, ending 2011-01-05.
- Validate: next 26 weeks, 2011-01-12 through 2011-07-06.
- Test: final 26 weeks, 2011-07-13 through 2012-01-04.

Splits follow the calendar even if a week is absent. No random row splitting. Feature encoders, coefficients and the smearing correction fit only training rows. XGBoost uses log price, support flags, deterministic calendar features and one-hot product/store IDs. It excludes units, spend, households and visits. Its fixed modest parameters are not tuned on the test period.

The baseline predicts the mean of up to four *previous observed* product/store sales, shifted before rolling. It updates as each week becomes available: this is one-step evaluation, not a 26-week forecast made at the beginning of the holdout. This difference is intentional and documented. All three models' metrics use the same rows; rows lacking a baseline or a trained product/store statistical model are counted as excluded.

MAE is average absolute error in units. WAPE is summed absolute error / summed actual units (undefined when the denominator is zero). Bias is mean predicted minus actual units. Compare both aggregate and product metrics; aggregate results emphasize high-volume products.

## 4. Recommend within explicit limits

Enumerate 1% baseline-price increments from 80% through 110%, rounded to cents, plus the exact unchanged baseline. Restrict changed prices to the selected store/product's historical 5th–95th percentile range. This range uses all available history at the simulated decision date; it is never used to improve held-out model scores.

Promotional combinations must be observed for the product. Automatic search additionally requires the combination to have appeared during training. Tag-only support requires a price below the latest base price. Prices and promotions may still be weakly supported jointly: these guards prevent gross extrapolation, not causal identification.

Discount is measured against the latest source base price for automatic search, or an explicit reference price for a manual scenario. A higher shelf price can imply a negative discount. The discount cap removes excessive markdowns; it does not prohibit increases.

Eligibility requires at least 78 training weeks, five distinct training prices, log-price standard deviation at least 0.03, a full-rank design, a wholly negative elasticity confidence interval, and lower validation MAE than the rolling baseline. The selected store additionally needs 78 observed weeks, five distinct prices, and an observation within two weeks of the dataset end. These are transparent project heuristics, not universal statistical standards.

Contribution profit is `(price - assumed unit cost) × demand - assumed weekly promotion cost`. Contribution margin is profit / revenue. The unchanged baseline uses its separate baseline cost assumption; changed promoted candidates use the supplied promotion cost, and changed unsupported candidates use zero promotional cost. Minimum margin requires a supplied unit cost. All costs default to explicitly documented zero/no-cost-analysis assumptions, not observed business costs.

Candidates rank by expected revenue. Ties favor the unchanged scenario, then lower price and deterministic support order. If evidence fails, the API returns exploratory candidates with no recommendation. If constraints eliminate every candidate, it returns `infeasible`. The unchanged baseline is always considered but can fail the user's constraints.

With constant elasticity, revenue is proportional to `price^(elasticity + 1)`. Therefore the best price often sits on a range boundary. A bounded candidate recommendation is honest; claiming an unconstrained interior optimum would not be.

## 5. Limits and potential extensions

This is observational estimation. Promotion timing may respond to expected demand, prices can be endogenous, and missing weeks may reflect stockouts. Conditional confidence intervals understate total business uncertainty. Product substitution, repeat-purchase effects and promotion carryover are not modeled. Test accuracy does not establish policy value. A prospective controlled pricing experiment would be the appropriate next step before business deployment.

# Infosys Valuation & Modeling Project

End-to-end valuation project for Infosys using FY2022-FY2025 actual data only. The project keeps the technical setup simple and puts the emphasis on corporate finance: DCF valuation, 3-statement forecasting, LBO debt capacity, sponsor return sensitivity, and an investment memo.

## Project Objective

Build a Python-automated valuation model for Infosys that answers:

- What is Infosys worth under a normalized unlevered DCF?
- How sensitive is equity value to WACC and terminal growth?
- Could a financial sponsor generate attractive returns in a 5-year LBO?
- How do leverage and exit multiple assumptions affect sponsor IRR?

## Files

- `data/historical_financials.csv` - FY2022-FY2025 actual financials in INR crore.
- `data/assumptions.csv` - DCF and LBO assumptions.
- `src/infosys_valuation_model.py` - Single Python script that runs the full model.
- `index.html` - Static frontend dashboard for reviewing the valuation outputs.
- `assets/styles.css` - Frontend visual system and responsive layout.
- `assets/app.js` - Lightweight tabs, section reveal, and active navigation behavior.
- `assets/infosys-logo.svg` - Local Infosys logo asset used by the dashboard.
- `render.yaml` - Render Blueprint for hosting the dashboard as a static site.
- `public/` - Static deploy folder used by Render.
- `data/fy2026_actuals.csv` - Current FY2026 actuals used for prediction validation.
- `src/validate_fy2026_actuals.py` - Forecast-vs-actual back-test script.
- `outputs/infosys_valuation_model.xlsx` - Excel workbook with actuals, assumptions, DCF, LBO, and sensitivities.
- `outputs/fy2026_prediction_validation.xlsx` - FY2026 validation workbook.
- `outputs/fy2026_validation_memo.md` - Short model back-test memo.
- `outputs/valuation_summary.csv` - Headline output metrics.
- `outputs/investment_memo.md` - Short finance memo.
- `outputs/lbo_irr_sensitivity.png` - LBO IRR sensitivity heatmap.

## How To Run

```powershell
pip install -r requirements.txt
python src\infosys_valuation_model.py
python src\validate_fy2026_actuals.py
```

The scripts refresh the model and validation files in `outputs/`.

To view the frontend:

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8000/index.html`.

## Render Deployment

The project includes a static-site Blueprint:

```yaml
services:
  - type: web
    name: infosys-valuation-workbench
    runtime: static
    buildCommand: "true"
    staticPublishPath: public
    autoDeployTrigger: commit
```

Render requires a Git-backed source. Once the repository is pushed and the Render CLI token is refreshed, create the service from this `render.yaml` or connect the repo in the Render Dashboard.

## Model Structure

### 1. Historical Actuals

The model uses FY2022-FY2025 only as actual data. FY2025 is the base year for valuation.

Key FY2025 inputs:

- Revenue: INR 162,990 crore
- Operating profit: INR 34,424 crore
- Net profit: INR 26,750 crore
- Cash flow from operations: INR 35,694 crore
- Free cash flow: INR 34,549 crore
- Cash and current investments less debt: INR 35,975 crore

### 2. 3-Statement Forecast

The forecast is intentionally simple:

- Revenue grows from FY2026-FY2030 using explicit annual assumptions.
- EBIT is driven by forecast operating margin.
- Cash taxes convert EBIT to NOPAT.
- D&A, capex, and net working capital are modeled as percentages of revenue.
- Free cash flow is calculated as NOPAT plus D&A, less capex, less change in NWC.

### 3. DCF Valuation

The DCF uses:

- 5-year explicit unlevered FCF forecast
- Gordon-growth terminal value
- Base WACC of 10.0%
- Base terminal growth of 3.0%
- Net cash added to enterprise value
- Share count in crore to calculate INR/share value

Base-case output:

- Enterprise value: INR 430,959 crore
- Equity value: INR 466,934 crore
- Implied value per share: INR 1,126
- Implied downside to illustrative INR 1,500 price: 25.0%

### 4. LBO Analysis

The LBO is a practical sponsor-return test, not a recommendation that Infosys is a likely take-private target.

The model includes:

- Entry valuation based on FY2025 EBITDA multiple
- Opening debt based on Debt / EBITDA
- Transaction fees
- Cash interest
- Cash sweep debt repayment
- Exit valuation after 5 years
- MOIC and IRR

Base-case output:

- FY2025 EBITDA: INR 39,284 crore
- Entry EV: INR 471,408 crore
- Opening acquisition debt: INR 78,568 crore
- Sponsor equity check: INR 402,268 crore
- MOIC: 1.45x
- Sponsor IRR: 7.7%

## Investment View

Infosys screens as a high-quality, cash-generative IT services company with a strong balance sheet. The DCF is more cautious than the illustrative market price because revenue growth is modeled conservatively and the company already trades like a mature, high-quality compounder. The LBO case is structurally constrained: cash flow is strong, but large-cap entry valuation and limited prudent leverage keep sponsor IRR below typical buyout return hurdles.

## FY2026 Validation

Infosys has now reported FY2026 actuals, so the project includes a separate forecast-vs-actual check. The FY2025-base model was too conservative on revenue and cash generation: FY2026 revenue beat the forecast by 6.4%, CFO beat by 11.6%, and company-reported FCF beat by 16.3%. Reported operating margin missed by 50 bps, but adjusted operating margin beat by 20 bps after excluding the Labour Codes provision.

## Data Sources

- Infosys P&L overview: https://www.infosys.com/investors/reports-filings/financials/profit-loss-data.html
- Infosys balance sheet overview: https://www.infosys.com/investors/reports-filings/financials/balance-sheet-data.html
- Infosys FY2025 annual report: https://www.infosys.com/investors/reports-filings/annual-report/annual-reports/ar-2024-25.html
- Infosys FY2026 data sheet: https://www.infosys.com/investors/reports-filings/financials/data-sheet.html
- Infosys Q4 FY2026 IFRS INR press release: https://www.infosys.com/investors/reports-filings/quarterly-results/2025-2026/q4/documents/ifrs-inr-press-release.pdf

## Notes

- Figures are in INR crore unless stated otherwise.
- FY2026 and later are forecast years only.
- Market price is illustrative and should be updated manually in `data/assumptions.csv` for live use.
- This is an educational valuation model, not investment advice.

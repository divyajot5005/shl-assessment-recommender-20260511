from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
LAST_ACTUAL_YEAR = "FY2025"
FORECAST_START_YEAR = 2026
FORECAST_END_YEAR = 2030
MODEL_INPUT_FILES = ("historical_financials.csv", "assumptions.csv")
EXCLUDED_MODEL_FILES = ("fy2026_actuals.csv",)


def fiscal_year_number(year):
    return int(str(year).upper().replace("FY", "").strip())


def validate_model_data_scope(historical, assumptions):
    years = historical["Fiscal Year"].map(fiscal_year_number)
    leaked_years = historical.loc[years >= FORECAST_START_YEAR, "Fiscal Year"].tolist()
    if leaked_years:
        raise ValueError(
            "Model historical actuals must stop at "
            f"{LAST_ACTUAL_YEAR}; found forecast/validation years: {', '.join(leaked_years)}."
        )

    last_actual = str(historical.iloc[-1]["Fiscal Year"])
    if last_actual != LAST_ACTUAL_YEAR:
        raise ValueError(f"Model base year must be {LAST_ACTUAL_YEAR}; found {last_actual}.")

    assumption_text = assumptions.astype(str).apply(lambda row: " ".join(row).lower(), axis=1)
    blocked_terms = (
        "fy2026 actual",
        "fy26 actual",
        "reported fy2026",
        "reported fy26",
        "fy2026 reported",
        "fy26 reported",
    )
    tuned_rows = assumptions[assumption_text.apply(lambda text: any(term in text for term in blocked_terms))]
    if not tuned_rows.empty:
        names = ", ".join(tuned_rows["Assumption"].astype(str).tolist())
        raise ValueError(
            "Model assumptions cannot reference FY2026 reported results or actuals. "
            f"Review: {names}."
        )


def load_inputs():
    historical = pd.read_csv(DATA / "historical_financials.csv")
    assumptions = pd.read_csv(DATA / "assumptions.csv")
    validate_model_data_scope(historical, assumptions)
    assumption_map = {
        row["Assumption"]: float(row["Value"])
        for _, row in assumptions.iterrows()
        if str(row["Value"]).replace(".", "", 1).replace("-", "", 1).isdigit()
    }
    return historical, assumptions, assumption_map


def forecast_three_statement(historical, a):
    last = historical.iloc[-1]
    years = [f"FY{y}" for y in range(FORECAST_START_YEAR, FORECAST_END_YEAR + 1)]
    rows = []
    prior_revenue = last["Revenue"]
    prior_nwc = last["NWC"]

    for year in years:
        revenue_growth = a[f"Revenue Growth {year}"]
        ebit_margin = a[f"EBIT Margin {year}"]
        revenue = prior_revenue * (1 + revenue_growth)
        ebit = revenue * ebit_margin
        taxes = ebit * a["Cash Tax Rate"]
        nopat = ebit - taxes
        da = revenue * a["D&A % Revenue"]
        capex = revenue * a["Capex % Revenue"]
        nwc = revenue * a["NWC % Revenue"]
        change_nwc = nwc - prior_nwc
        fcf = nopat + da - capex - change_nwc
        net_income = nopat
        cfo = net_income + da - change_nwc

        rows.append(
            {
                "Fiscal Year": year,
                "Revenue": revenue,
                "Revenue Growth": revenue_growth,
                "EBIT": ebit,
                "EBIT Margin": ebit_margin,
                "Cash Taxes": taxes,
                "NOPAT": nopat,
                "D&A": da,
                "Capex": capex,
                "NWC": nwc,
                "Change in NWC": change_nwc,
                "CFO": cfo,
                "FCF": fcf,
            }
        )

        prior_revenue = revenue
        prior_nwc = nwc

    return pd.DataFrame(rows)


def run_dcf(forecast, historical, a):
    wacc = a["WACC"]
    terminal_growth = a["Terminal Growth"]
    pv_rows = []

    for idx, row in forecast.iterrows():
        year_number = idx + 1
        pv_fcf = row["FCF"] / ((1 + wacc) ** year_number)
        pv_rows.append(
            {
                "Fiscal Year": row["Fiscal Year"],
                "FCF": row["FCF"],
                "Discount Factor": 1 / ((1 + wacc) ** year_number),
                "PV of FCF": pv_fcf,
            }
        )

    terminal_fcf = forecast.iloc[-1]["FCF"] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal_value = terminal_value / ((1 + wacc) ** len(forecast))
    enterprise_value = sum(row["PV of FCF"] for row in pv_rows) + pv_terminal_value
    net_cash = (
        historical.iloc[-1]["Cash and Equivalents"]
        + historical.iloc[-1]["Current Investments"]
        - historical.iloc[-1]["Total Debt"]
    )
    equity_value = enterprise_value + net_cash
    price_per_share = equity_value / a["Shares Outstanding Crore"]

    dcf_summary = pd.DataFrame(
        [
            ("PV of forecast FCF", sum(row["PV of FCF"] for row in pv_rows)),
            ("Terminal value", terminal_value),
            ("PV of terminal value", pv_terminal_value),
            ("Enterprise value", enterprise_value),
            ("Cash + current investments - debt", net_cash),
            ("Equity value", equity_value),
            ("Shares outstanding crore", a["Shares Outstanding Crore"]),
            ("Implied value / share INR", price_per_share),
            ("Illustrative market price INR", a["Market Price INR"]),
            ("Upside / (downside)", price_per_share / a["Market Price INR"] - 1),
        ],
        columns=["Metric", "Value"],
    )

    return pd.DataFrame(pv_rows), dcf_summary


def dcf_sensitivity(forecast, historical, a):
    waccs = [0.085, 0.09, 0.095, 0.10, 0.105, 0.11, 0.115]
    terminal_growths = [0.020, 0.025, 0.030, 0.035, 0.040]
    rows = []
    original_wacc = a["WACC"]
    original_growth = a["Terminal Growth"]

    for growth in terminal_growths:
        row = {"Terminal Growth": growth}
        for wacc in waccs:
            a["WACC"] = wacc
            a["Terminal Growth"] = growth
            _, summary = run_dcf(forecast, historical, a)
            row[f"{wacc:.1%} WACC"] = summary.loc[
                summary["Metric"] == "Implied value / share INR", "Value"
            ].iloc[0]
        rows.append(row)

    a["WACC"] = original_wacc
    a["Terminal Growth"] = original_growth
    return pd.DataFrame(rows)


def run_lbo(forecast, historical, a):
    last = historical.iloc[-1]
    fy25_ebitda = last["Operating Profit"] + last["D&A"]
    entry_ev = fy25_ebitda * a["Entry EBITDA Multiple"]
    fees = entry_ev * a["Sponsor Transaction Fee %"]
    opening_debt = fy25_ebitda * a["Debt / EBITDA at Close"]
    sponsor_equity = entry_ev + fees - opening_debt
    debt = opening_debt
    rows = []

    for _, row in forecast.iterrows():
        ebitda = row["EBIT"] + row["D&A"]
        interest = debt * a["Interest Rate"]
        cash_after_interest = max(row["FCF"] - interest, 0)
        repayment = min(debt, cash_after_interest * a["Cash Sweep %"])
        ending_debt = debt - repayment
        rows.append(
            {
                "Fiscal Year": row["Fiscal Year"],
                "Revenue": row["Revenue"],
                "EBITDA": ebitda,
                "FCF before debt service": row["FCF"],
                "Beginning Debt": debt,
                "Cash Interest": interest,
                "Debt Repayment": repayment,
                "Ending Debt": ending_debt,
                "Net Debt / EBITDA": ending_debt / ebitda,
            }
        )
        debt = ending_debt

    lbo = pd.DataFrame(rows)
    exit_ebitda = lbo.iloc[-1]["EBITDA"]
    exit_ev = exit_ebitda * a["Exit EBITDA Multiple"]
    exit_equity = exit_ev - lbo.iloc[-1]["Ending Debt"]
    moic = exit_equity / sponsor_equity
    irr = moic ** (1 / len(forecast)) - 1

    summary = pd.DataFrame(
        [
            ("FY2025 EBITDA", fy25_ebitda),
            ("Entry EV", entry_ev),
            ("Transaction fees", fees),
            ("Opening acquisition debt", opening_debt),
            ("Sponsor equity check", sponsor_equity),
            ("Exit EV", exit_ev),
            ("Exit equity value", exit_equity),
            ("MOIC", moic),
            ("Sponsor IRR", irr),
        ],
        columns=["Metric", "Value"],
    )
    return lbo, summary


def lbo_sensitivity(forecast, historical, a):
    leverage_cases = [1.0, 1.5, 2.0, 2.5, 3.0]
    exit_multiples = [10.0, 11.0, 12.0, 13.0, 14.0]
    rows = []
    original_leverage = a["Debt / EBITDA at Close"]
    original_exit = a["Exit EBITDA Multiple"]

    for leverage in leverage_cases:
        row = {"Debt / EBITDA": leverage}
        for multiple in exit_multiples:
            a["Debt / EBITDA at Close"] = leverage
            a["Exit EBITDA Multiple"] = multiple
            _, summary = run_lbo(forecast, historical, a)
            row[f"{multiple:.1f}x Exit"] = summary.loc[
                summary["Metric"] == "Sponsor IRR", "Value"
            ].iloc[0]
        rows.append(row)

    a["Debt / EBITDA at Close"] = original_leverage
    a["Exit EBITDA Multiple"] = original_exit
    return pd.DataFrame(rows)


def model_data_scope():
    return pd.DataFrame(
        [
            ("Allowed valuation input", ", ".join(MODEL_INPUT_FILES)),
            ("Last reported actual used by valuation model", LAST_ACTUAL_YEAR),
            ("Forecast period", f"FY{FORECAST_START_YEAR}-FY{FORECAST_END_YEAR}"),
            ("Excluded from valuation inputs and tuning", ", ".join(EXCLUDED_MODEL_FILES)),
            (
                "FY2026 actuals usage",
                "Back-test validation only; not used to set, fit, or tune model assumptions.",
            ),
        ],
        columns=["Item", "Scope"],
    )


def make_heatmap(lbo_sens):
    OUTPUTS.mkdir(exist_ok=True)
    values = lbo_sens.drop(columns=["Debt / EBITDA"]).values
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(values.shape[1]))
    ax.set_xticklabels(lbo_sens.columns[1:])
    ax.set_yticks(range(values.shape[0]))
    ax.set_yticklabels([f"{x:.1f}x" for x in lbo_sens["Debt / EBITDA"]])
    ax.set_xlabel("Exit Multiple")
    ax.set_ylabel("Opening Leverage")
    ax.set_title("Infosys LBO IRR Sensitivity")
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            ax.text(x, y, f"{values[y, x]:.1%}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, format="%.0f")
    fig.tight_layout()
    path = OUTPUTS / "lbo_irr_sensitivity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_excel(historical, assumptions, forecast, dcf, dcf_summary, dcf_sens, lbo, lbo_summary, lbo_sens):
    OUTPUTS.mkdir(exist_ok=True)
    workbook_path = OUTPUTS / "infosys_valuation_model.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        model_data_scope().to_excel(writer, sheet_name="Model Data Scope", index=False)
        historical.to_excel(writer, sheet_name="Historical Actuals", index=False)
        assumptions.to_excel(writer, sheet_name="Assumptions", index=False)
        forecast.to_excel(writer, sheet_name="3 Statement Forecast", index=False)
        dcf.to_excel(writer, sheet_name="DCF Build", index=False)
        dcf_summary.to_excel(writer, sheet_name="DCF Summary", index=False)
        dcf_sens.to_excel(writer, sheet_name="DCF Sensitivity", index=False)
        lbo.to_excel(writer, sheet_name="LBO Debt Schedule", index=False)
        lbo_summary.to_excel(writer, sheet_name="LBO Summary", index=False)
        lbo_sens.to_excel(writer, sheet_name="LBO IRR Sensitivity", index=False)

    wb = load_workbook(workbook_path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
        for col in range(1, ws.max_column + 1):
            letter = get_column_letter(col)
            ws.column_dimensions[letter].width = min(
                max(len(str(ws.cell(row=row, column=col).value or "")) for row in range(1, ws.max_row + 1)) + 2,
                32,
            )
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "0.0%" if abs(cell.value) < 1 else "#,##0.0"

    ws = wb["3 Statement Forecast"]
    chart = LineChart()
    chart.title = "Revenue and FCF Forecast"
    chart.y_axis.title = "INR crore"
    chart.x_axis.title = "Fiscal Year"
    data = Reference(ws, min_col=2, max_col=13, min_row=1, max_row=ws.max_row)
    cats = Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    ws.add_chart(chart, "P2")

    wb.save(workbook_path)
    return workbook_path


def write_memo(dcf_summary, lbo_summary, lbo_sens, heatmap_path):
    price = dcf_summary.loc[dcf_summary["Metric"] == "Implied value / share INR", "Value"].iloc[0]
    upside = dcf_summary.loc[dcf_summary["Metric"] == "Upside / (downside)", "Value"].iloc[0]
    irr = lbo_summary.loc[lbo_summary["Metric"] == "Sponsor IRR", "Value"].iloc[0]
    debt = lbo_summary.loc[lbo_summary["Metric"] == "Opening acquisition debt", "Value"].iloc[0]
    equity = lbo_summary.loc[lbo_summary["Metric"] == "Sponsor equity check", "Value"].iloc[0]

    memo = f"""# Infosys Valuation & Modeling Memo

## Recommendation

Base-case DCF value is INR {price:,.0f} per share, implying {upside:.1%} versus the illustrative market price in the assumptions file. The LBO base case generates a {irr:.1%} sponsor IRR, which is modest because Infosys is already large, cash-rich, mature, and difficult to lever aggressively without changing its conservative capital structure.

## What the model does

- Uses FY2022-FY2025 reported actuals only; FY2026-FY2030 are forecast years.
- Excludes FY2026 reported results from valuation inputs and assumption tuning; FY2026 data is only for the separate back-test.
- Builds a simple 3-statement forecast around revenue, EBIT margin, cash taxes, D&A, capex, and working capital.
- Values Infosys using an unlevered DCF with terminal-value sensitivity across WACC and terminal growth.
- Tests a 5-year LBO with acquisition debt, cash interest, cash sweep repayment, MOIC, and IRR sensitivity across leverage and exit multiples.

## Key finance takeaways

- Infosys is a high-quality but mature IT services asset: FY2025 revenue of INR 162,990 crore, operating profit of INR 34,424 crore, and FCF of INR 34,549 crore.
- The DCF is terminal-value heavy, so WACC and terminal growth matter more than small annual forecast changes.
- The LBO works operationally because cash conversion is strong, but sponsor returns are constrained by entry valuation and limited prudent leverage.
- Base transaction funding uses INR {debt:,.0f} crore of acquisition debt and INR {equity:,.0f} crore of sponsor equity.

## Main risks

- Client discretionary spend remains weak and delays the revenue recovery case.
- Generative AI reduces pricing power faster than automation improves delivery cost.
- INR/USD movement affects translated growth and offshore margin optics.
- Large-cap entry multiples compress at exit, which has the largest direct impact on LBO IRR.

## Source Notes

- Infosys investor financial overview: https://www.infosys.com/investors/reports-filings/financials/profit-loss-data.html
- Infosys balance sheet data: https://www.infosys.com/investors/reports-filings/financials/balance-sheet-data.html
- Infosys FY2025 annual report page: https://www.infosys.com/investors/reports-filings/annual-report/annual-reports/ar-2024-25.html

![LBO IRR sensitivity]({heatmap_path.as_posix()})
"""
    path = OUTPUTS / "investment_memo.md"
    path.write_text(memo, encoding="utf-8")
    return path


def main():
    historical, assumptions, a = load_inputs()
    forecast = forecast_three_statement(historical, a)
    dcf, dcf_summary = run_dcf(forecast, historical, a)
    dcf_sens = dcf_sensitivity(forecast, historical, a)
    lbo, lbo_summary = run_lbo(forecast, historical, a)
    lbo_sens = lbo_sensitivity(forecast, historical, a)
    heatmap_path = make_heatmap(lbo_sens)
    workbook_path = write_excel(
        historical,
        assumptions,
        forecast,
        dcf,
        dcf_summary,
        dcf_sens,
        lbo,
        lbo_summary,
        lbo_sens,
    )
    memo_path = write_memo(dcf_summary, lbo_summary, lbo_sens, heatmap_path)

    summary = pd.concat(
        [
            dcf_summary.assign(Model="DCF"),
            lbo_summary.assign(Model="LBO"),
        ],
        ignore_index=True,
    )[["Model", "Metric", "Value"]]
    summary.to_csv(OUTPUTS / "valuation_summary.csv", index=False)

    print(f"Built {workbook_path}")
    print(f"Built {memo_path}")
    print(f"Built {heatmap_path}")


if __name__ == "__main__":
    main()

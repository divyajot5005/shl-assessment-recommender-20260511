from pathlib import Path
from unittest.mock import patch

import pandas as pd

import infosys_valuation_model as model


FY2026_ACTUALS = (model.DATA / "fy2026_actuals.csv").resolve()
ALLOWED_MODEL_INPUTS = {
    (model.DATA / filename).resolve()
    for filename in model.MODEL_INPUT_FILES
}
ASSUMPTION_ACTUAL_CHECKS = {
    "Revenue Growth FY2026": ("Revenue Growth",),
    "EBIT Margin FY2026": ("Operating Margin", "Adjusted Operating Margin"),
}


def resolve_path(path):
    return Path(path).resolve()


def assert_historical_actuals_stop_at_fy2025():
    historical = pd.read_csv(model.DATA / "historical_financials.csv")
    years = historical["Fiscal Year"].map(model.fiscal_year_number)
    leaked_years = historical.loc[years >= model.FORECAST_START_YEAR, "Fiscal Year"].tolist()
    if leaked_years:
        raise AssertionError(
            "Historical actuals include FY2026 or later: "
            f"{', '.join(leaked_years)}."
        )

    last_actual = str(historical.iloc[-1]["Fiscal Year"])
    if last_actual != model.LAST_ACTUAL_YEAR:
        raise AssertionError(f"Last model actual must be {model.LAST_ACTUAL_YEAR}; found {last_actual}.")


def assert_assumptions_not_backfit_to_fy2026_actuals():
    assumptions = pd.read_csv(model.DATA / "assumptions.csv")
    actuals = pd.read_csv(FY2026_ACTUALS).set_index("Metric")
    assumption_values = assumptions.set_index("Assumption")["Value"].astype(float)

    matches = []
    for assumption_name, actual_metric_names in ASSUMPTION_ACTUAL_CHECKS.items():
        if assumption_name not in assumption_values:
            continue
        assumption_value = float(assumption_values.loc[assumption_name])
        for metric_name in actual_metric_names:
            actual_value = float(actuals.loc[metric_name, "Actual"])
            if abs(assumption_value - actual_value) <= 1e-9:
                matches.append(f"{assumption_name} matches FY2026 actual {metric_name}")

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
        matches.extend(
            f"{row['Assumption']} references FY2026 reported results"
            for _, row in tuned_rows.iterrows()
        )

    if matches:
        raise AssertionError(
            "FY2026 result leakage detected in model assumptions: "
            + "; ".join(matches)
            + "."
        )


def assert_valuation_model_does_not_read_fy2026_actuals():
    read_files = []
    original_read_csv = model.pd.read_csv

    def tracking_read_csv(path, *args, **kwargs):
        resolved = resolve_path(path)
        read_files.append(resolved)
        if resolved == FY2026_ACTUALS:
            raise AssertionError(f"Valuation model attempted to read excluded file: {FY2026_ACTUALS}")
        if resolved not in ALLOWED_MODEL_INPUTS:
            raise AssertionError(f"Valuation model attempted to read unexpected input file: {resolved}")
        return original_read_csv(path, *args, **kwargs)

    with patch.object(model.pd, "read_csv", side_effect=tracking_read_csv):
        historical, _, assumptions = model.load_inputs()
        forecast = model.forecast_three_statement(historical, assumptions)
        model.run_dcf(forecast, historical, assumptions)
        model.dcf_sensitivity(forecast, historical, assumptions)
        model.run_lbo(forecast, historical, assumptions)
        model.lbo_sensitivity(forecast, historical, assumptions)

    return read_files


def main():
    checks = (
        ("Historical actuals stop at FY2025", assert_historical_actuals_stop_at_fy2025),
        ("Assumptions are not backfit to FY2026 actuals", assert_assumptions_not_backfit_to_fy2026_actuals),
        ("Valuation model does not read FY2026 actuals", assert_valuation_model_does_not_read_fy2026_actuals),
    )

    for label, check in checks:
        check()
        print(f"PASS: {label}")


if __name__ == "__main__":
    main()

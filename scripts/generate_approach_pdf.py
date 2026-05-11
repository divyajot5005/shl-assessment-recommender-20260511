from __future__ import annotations

import json
from pathlib import Path
import sys

import fitz
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings


def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def format_metric(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_preview_images(pdf_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf_path)
    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pixmap.save(output_dir / f"page-{page_index + 1}.png")
    document.close()


def main() -> None:
    settings = get_settings()
    eval_dir = settings.artifacts_dir / "evaluation"
    submission_dir = settings.artifacts_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    trace_metrics = load_metrics(eval_dir / "public_trace_metrics.json")
    behavior_metrics = load_metrics(eval_dir / "behavior_probe_metrics.json")
    live_trace_metrics = load_metrics(eval_dir / "public_trace_metrics_live.json")
    live_behavior_metrics = load_metrics(eval_dir / "behavior_probe_metrics_live.json")

    pdf_path = submission_dir / "approach_document.pdf"
    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F3D56"),
        spaceAfter=12,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=13,
        textColor=colors.HexColor("#0F3D56"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=12,
        spaceAfter=6,
    )

    story = [
        Paragraph("SHL Conversational Assessment Recommender", title),
        Paragraph(
            "Design goal: convert a vague hiring brief into grounded SHL assessment recommendations through a stateless API that can clarify, refine, compare, and refuse off-scope requests without hallucinating products or URLs.",
            body,
        ),
        Paragraph("Architecture", heading),
        Paragraph(
            "The service is a FastAPI app with a preloaded local catalog index. Startup downloads or loads the provided SHL JSON catalog, normalizes 377 products, derives test-type codes, aliases, duration metadata, language coverage, job-level sets, and a report-vs-assessment flag. Retrieval stays local and deterministic: word-level TF-IDF, character n-gram TF-IDF, and dense SVD vectors are combined into a single score, then re-ranked with business rules for language, level, use case, duration, and shortlist continuity.",
            body,
        ),
        Paragraph(
            "The conversation layer is stateless over the full message history. It applies guardrails first, then classifies the turn into clarify, recommend, refine, or compare. Clarification is capped at two questions and is driven by explicit missing-slot checks such as role scope, contact-center language variant, full-stack orientation, and leadership use case. Shortlist-driving state remains deterministic. When hosted credentials are present, an OpenAI-compatible LLM can rewrite the final prose reply, but it does not control product selection by default.",
            body,
        ),
        Paragraph("Retrieval and ranking choices", heading),
        Paragraph(
            "Pure keyword matching was not enough for mixed briefs such as job descriptions, role pivots, or product comparisons. The final ranker combines semantic recall with explicit boosts for known catalog behaviors: language-sensitive SVAR/contact-center flows, leadership instrument-plus-report bundles, development stacks such as GSA plus development reports, safety-focused DSI cases, and technical batteries that combine stack tests with Verify G+ and, when appropriate, OPQ32r. Report products are penalized by default so they only surface when the user is clearly asking for feedback, benchmarking, or development outputs.",
            body,
        ),
        Paragraph("Evaluation", heading),
    ]

    mean_recall = trace_metrics.get("mean_recall_at_10", 0.0)
    mean_candidate_hit = trace_metrics.get("mean_candidate_hit_rate")
    pass_rate = behavior_metrics.get("pass_rate", 0.0)
    trace_count = trace_metrics.get("trace_count", 0)

    metrics_table = Table(
        [
            ["Metric", "Result"],
            ["Public trace count", str(trace_count)],
            ["Mean Recall@10", format_metric(mean_recall)],
            ["Mean candidate hit rate", format_metric(mean_candidate_hit)],
            ["Behavior probe pass rate", format_metric(pass_rate)],
        ],
        colWidths=[2.8 * inch, 2.2 * inch],
    )
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F3D56")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B7C9D3")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F8FA")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    live_metrics_table = None
    if live_trace_metrics or live_behavior_metrics:
        live_metrics_table = Table(
            [
                ["Live endpoint metric", "Result"],
                ["Public trace count", format_metric(live_trace_metrics.get("trace_count"))],
                ["Mean Recall@10", format_metric(live_trace_metrics.get("mean_recall_at_10"))],
                ["Mean candidate hit rate", format_metric(live_trace_metrics.get("mean_candidate_hit_rate"))],
                ["Behavior probe pass rate", format_metric(live_behavior_metrics.get("pass_rate"))],
            ],
            colWidths=[2.8 * inch, 2.2 * inch],
        )
        live_metrics_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#415A66")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B7C9D3")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F8FA")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
    story.extend(
        [
            metrics_table,
            Spacer(1, 0.15 * inch),
            Paragraph(
                "Evaluation is split into two layers. First, a replay harness walks the 10 public traces turn-by-turn, validates schema compliance, verifies every returned URL is catalog-backed, and computes Recall@10 plus candidate hit rate when internal candidate rankings are available. Second, targeted probes check behaviors the replay traces under-sample: vague-query clarification, shortlist refinement, grounded product comparison, legal refusal, prompt-injection refusal, language constraints, sales-development grounding, and variant disambiguation. The same scripts can target either the local service or a deployed endpoint, which keeps reported metrics aligned with the runtime being submitted.",
                body,
            ),
        ]
    )
    if live_metrics_table is not None:
        story.extend(
            [
                live_metrics_table,
                Spacer(1, 0.12 * inch),
            ]
        )
    story.extend(
        [
            Paragraph("What did not work", heading),
            Paragraph(
                "Three weaker approaches were discarded. A pure nearest-neighbor retriever over the catalog over-selected report products and missed shortlist composition. A pure rules engine was brittle on long job descriptions and unexpected wording. Live web scraping at request time added avoidable latency and fragility. The final design keeps retrieval local, uses the supplied JSON feed as runtime truth, and limits model usage to optional assistance rather than core correctness.",
                body,
            ),
            Paragraph("Improvement loop", heading),
            Paragraph(
                "Each iteration was judged against groundedness first, then utility. I used the public-trace replay to identify missed shortlist items, added rules only when they represented genuine catalog behavior rather than narrow wording, and checked that the same changes improved candidate hit rate without breaking refusal or comparison probes. The same evaluation flow can now be rerun against the deployed endpoint before submission, which makes the reported metrics auditable rather than aspirational.",
                body,
            ),
        ]
    )

    document.build(story)
    render_preview_images(pdf_path, submission_dir / "approach_preview")
    print(f"Generated {pdf_path}")


if __name__ == "__main__":
    main()

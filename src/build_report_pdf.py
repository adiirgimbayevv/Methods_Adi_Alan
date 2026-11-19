"""Build a concise PDF report from the ProgFeed summary and figures."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "data/processed/student_outcomes_summary.csv"
BENCHMARK_SUMMARY = ROOT / "data/processed/metrics_summary.csv"
FIGURES = ROOT / "reports/figures"
OUTPUT = ROOT / "output/pdf/Task3_Report_ProgFeed.pdf"
ARM_LABELS = {
    "no_feedback": "No feedback",
    "tc": "Test-case hint",
    "nl": "Natural-language hint",
}


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def add_page_chrome(canvas, doc) -> None:
    canvas.saveState()
    width, height = letter
    canvas.setStrokeColor(colors.HexColor("#C9D4DF"))
    canvas.line(0.72 * inch, height - 0.55 * inch, width - 0.72 * inch, height - 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#526579"))
    canvas.drawString(0.72 * inch, height - 0.42 * inch, "ProgFeed classroom analysis | Task 3")
    canvas.drawRightString(width - 0.72 * inch, 0.38 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build() -> Path:
    summary = pd.read_csv(SUMMARY).set_index("feedback_type")
    benchmark = pd.read_csv(BENCHMARK_SUMMARY).sort_values(["load_level", "variant"])
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=23, leading=28, textColor=colors.HexColor("#17324D"),
        alignment=TA_LEFT, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="Deck", parent=styles["Normal"], fontSize=11, leading=15,
        textColor=colors.HexColor("#40566D"), spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="Section", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=13, leading=16, textColor=colors.HexColor("#17324D"),
        spaceBefore=8, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="BodyCompact", parent=styles["BodyText"], fontSize=9.2,
        leading=13, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="Caption", parent=styles["BodyText"], fontSize=8.5,
        leading=11, textColor=colors.HexColor("#40566D"), alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="TableHeader", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=7.0, leading=8.2, textColor=colors.white, alignment=TA_CENTER,
    ))

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter,
        rightMargin=0.72 * inch, leftMargin=0.72 * inch,
        topMargin=0.78 * inch, bottomMargin=0.62 * inch,
        title="AI-Generated Feedback and Programming Outcomes",
        author="Adi Irgimbayev; Alan Shmyrin",
    )
    story = [
        Spacer(1, 0.18 * inch),
        Paragraph("AI-Generated Feedback and Programming Outcomes", styles["ReportTitle"]),
        Paragraph("An analysis of anonymized classroom data from an introductory Python course", styles["Deck"]),
        Paragraph("Adi Irgimbayev and Alan Shmyrin | Task 3 research report", styles["BodyCompact"]),
        Paragraph("Research question", styles["Section"]),
        Paragraph(
            "Among students with a failed first submission in randomized intervention labs, "
            "how do test-case hints, natural-language hints, and no feedback relate to retries, "
            "later scores, and eventual full-credit submissions?",
            styles["BodyCompact"],
        ),
        Paragraph("Data and method", styles["Section"]),
        Paragraph(
            "ProgFeed contains 17,385 graded-test rows from 215 consenting students, grouped "
            "into 6,693 submissions. We use labs 05, 06, 07, and 09, where feedback assignment "
            "was randomized. Test rows are aggregated to submission scores and then to student-lab "
            "trajectories. The descriptive outcome is the score change from the first to the final "
            "observed submission among units whose first submission failed. Confidence intervals "
            "resample students, retaining each selected student’s lab records together (5,000 draws; seed 42).",
            styles["BodyCompact"],
        ),
        Paragraph("Descriptive results", styles["Section"]),
    ]

    data = [[
        "Assigned feedback", "Units", "Failed first", "Retry rate",
        "Full credit", "Mean gain (pp)", "Cluster 95% CI",
    ]]
    for arm in ("no_feedback", "tc", "nl"):
        row = summary.loc[arm]
        data.append([
            ARM_LABELS[arm], str(int(row["student_lab_units"])),
            str(int(row["first_attempt_failed"])),
            pct(row["retry_rate_after_first_failure"]),
            pct(row["eventual_full_pass_rate_after_first_failure"]),
            f"{row['mean_score_gain_pp_after_first_failure']:.2f}",
            f"[{row['score_gain_cluster_ci95_low']:.2f}, {row['score_gain_cluster_ci95_high']:.2f}]",
        ])
    table = Table(data, colWidths=[1.28*inch, 0.48*inch, 0.65*inch, 0.62*inch,
                                   0.64*inch, 0.78*inch, 1.12*inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.4),
        ("LEADING", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F9")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D2DCE5")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([
        table, Spacer(1, 9),
        Paragraph(
            "The table is descriptive, not a causal estimate. Feedback is delivered only after "
            "errors, and repeated submissions are clustered within students. ProgFeed contains no "
            "validated critical-thinking measure; no critical-thinking conclusion is made.",
            styles["BodyCompact"],
        ),
        PageBreak(),
        Paragraph("Score gain after an initial failure", styles["Section"]),
        Image(str(FIGURES / "fig3_feedback_score_gain.png"), width=6.65*inch, height=4.16*inch),
        Spacer(1, 8),
        Paragraph(
            "Mean change from first to final observed score for student-lab units whose first "
            "submission failed. Error bars show 95% student-cluster bootstrap intervals.",
            styles["Caption"],
        ),
        PageBreak(),
        Paragraph("Final submission scores and study limitations", styles["Section"]),
        Image(str(FIGURES / "fig4_final_score_distribution.png"), width=6.65*inch, height=4.16*inch),
        Spacer(1, 8),
        Paragraph(
            "Final autograder score as a percentage of available points, by assigned feedback arm. "
            "Boxes show quartiles and the median; diamonds mark means.",
            styles["Caption"],
        ),
        Spacer(1, 12),
        Paragraph("Limitations", styles["Section"]),
        Paragraph(
            "Only four randomized labs are included. Assigned feedback and feedback actually "
            "delivered differ because the tool responds to failed attempts. Final observed scores "
            "may be incomplete. Results from one introductory Python course may not generalize "
            "to other courses or kinds of AI assistance.",
            styles["BodyCompact"],
        ),
        Paragraph("Source and attribution", styles["Section"]),
        Paragraph(
            "Heickal, H. &amp; Lan, A. (2026). A Classroom Study of LLM-generated Feedback "
            "Intervention in Introductory Programming. Proceedings of the Impactful and Responsible "
            "AI Systems for Education Workshop, PMLR 339, 12-21. Dataset: ProgFeed public repository, "
            "CC BY 4.0. https://proceedings.mlr.press/v339/heickal26a.html",
            styles["BodyCompact"],
        ),
        PageBreak(),
        Paragraph("Local ETL benchmark", styles["Section"]),
        Paragraph(
            "The benchmark times a Python row loop and a pandas groupby while aggregating "
            "actual ProgFeed rows. It includes 5 warm-up and 30 measured runs per condition. "
            "Input size is measured in graded-test rows; throughput is rows per second. These are "
            "local data-processing timings, not AI service latency.",
            styles["BodyCompact"],
        ),
    ])

    bench_table_data = [[
        Paragraph("Input<br/>rows", styles["TableHeader"]),
        Paragraph("Baseline<br/>median (ms)", styles["TableHeader"]),
        Paragraph("Pandas<br/>median (ms)", styles["TableHeader"]),
        Paragraph("Baseline<br/>p95 (ms)", styles["TableHeader"]),
        Paragraph("Pandas<br/>p95 (ms)", styles["TableHeader"]),
        Paragraph("Proposed<br/>speedup", styles["TableHeader"]),
        Paragraph("Error<br/>count", styles["TableHeader"]),
    ]]
    baseline_name = "Baseline (Python row loop)"
    proposed_name = "Proposed (pandas groupby)"
    for load_rows in sorted(benchmark["load_level"].unique()):
        pair = benchmark[benchmark["load_level"] == load_rows].set_index("variant")
        bench_table_data.append([
            f"{int(load_rows):,}",
            f"{pair.loc[baseline_name, 'median']:.3f}",
            f"{pair.loc[proposed_name, 'median']:.3f}",
            f"{pair.loc[baseline_name, 'p95']:.3f}",
            f"{pair.loc[proposed_name, 'p95']:.3f}",
            f"{pair.loc[proposed_name, 'speedup_vs_baseline']:.2f}x",
            f"{int(pair['error_count'].sum())}",
        ])
    bench_table = Table(
        bench_table_data,
        colWidths=[0.62*inch, 1.17*inch, 1.17*inch, 1.15*inch, 1.15*inch, 0.72*inch, 0.54*inch],
        repeatRows=1,
    )
    bench_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("LEADING", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F9")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D2DCE5")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([
        bench_table,
        Spacer(1, 10),
        Paragraph(
            "The pandas groupby has a fixed setup cost and is slower for small input sizes. "
            "It becomes faster than the Python loop at 2,500 rows in this local run. "
            "The speedup is the baseline median divided by the pandas median.",
            styles["BodyCompact"],
        ),
        PageBreak(),
        Paragraph("Measured latency by input size", styles["Section"]),
        Image(str(FIGURES / "fig1_latency_vs_load.png"), width=6.65*inch, height=4.08*inch),
        Spacer(1, 8),
        Paragraph(
            "Mean aggregation latency by number of actual graded-test rows processed. "
            "Shaded bands show 95% bootstrap confidence intervals.",
            styles["Caption"],
        ),
        PageBreak(),
        Paragraph("Latency distribution", styles["Section"]),
        Image(str(FIGURES / "fig2_distribution_boxplot.png"), width=6.65*inch, height=3.75*inch),
        Spacer(1, 8),
        Paragraph(
            "Latency distributions by input size and implementation. Notches indicate median "
            "uncertainty; diamonds show means.",
            styles["Caption"],
        ),
    ])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=add_page_chrome, onLaterPages=add_page_chrome)
    return OUTPUT


if __name__ == "__main__":
    print(build())

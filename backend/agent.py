import json
import anthropic
from analysis.eda import get_eda_summary
from analysis.pca_analysis import run_pca
from analysis.outlier_detection import run_outlier_detection

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "get_eda_summary",
        "description": "Get EDA summary: label distribution, feature means per label, binary feature rates.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_outlier_detection_results",
        "description": "Run Isolation Forest + LOF. Returns outlier rates by label and authenticity score stats.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_pca_summary",
        "description": "Run PCA and return explained variance and feature loadings for PC1 and PC2.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


def _run_tool(name, _input):
    if name == "get_eda_summary":
        r = get_eda_summary()
        return json.dumps({
            'total_records': r['total_records'],
            'label_counts': r['label_counts'],
            'feature_means_by_label': r['feature_means_by_label'],
            'binary_rates_by_label': r['binary_rates_by_label'],
        })
    if name == "get_outlier_detection_results":
        r = run_outlier_detection()
        return json.dumps({
            'iso_outlier_by_label': r['iso_outlier_by_label'],
            'lof_outlier_by_label': r['lof_outlier_by_label'],
            'score_stats_by_label': r['score_stats_by_label'],
        })
    if name == "get_pca_summary":
        r = run_pca()
        return json.dumps({
            'explained_variance': r['explained_variance'],
            'loadings': r['loadings'],
        })
    return json.dumps({'error': 'Unknown tool'})


SYSTEM_PROMPT = """You are an AI forensic investigator specializing in social media fraud detection.
You analyze Instagram account data to detect bots, scammers, and spammers using unsupervised ML.
Be analytical, precise, and cite specific numbers from the data."""

USER_PROMPT = """Conduct a full forensic analysis of our Instagram account dataset.

Steps:
1. Call get_eda_summary to understand feature distributions across account types
2. Call get_outlier_detection_results to analyze Isolation Forest vs LOF performance
3. Call get_pca_summary to understand which features drive separation

Then write a structured forensic report with these sections:
## Executive Summary
## Key EDA Findings
## Outlier Detection Analysis (Isolation Forest vs LOF)
## PCA Insights
## Precision-Recall Trade-offs in Unsupervised Detection
## Authenticity Confidence Score Methodology
## Recommendations

Use specific numbers. Highlight precision-recall trade-offs inherent to unsupervised detection."""


def generate_forensic_report():
    messages = [{"role": "user", "content": USER_PROMPT}]

    while True:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, 'text'):
                    return block.text
            return "Report generation completed."

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = _run_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        if not tool_results:
            break

        messages.append({"role": "user", "content": tool_results})

    return "Report generation completed."

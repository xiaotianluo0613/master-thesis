#!/usr/bin/env python3
"""
Create a supervisor-facing PDF showing improved query generation examples.

For each selected group shows:
  1. Source text excerpt (chunk segments)
  2. Full prompt (system + user with few-shot examples)
  3. Generated queries with English translations
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import requests
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, HRFlowable


# ── helpers ──────────────────────────────────────────────────────────────────

def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace("{", "&#123;")
         .replace("}", "&#125;")
    )


def _get_github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ""


def _translate(texts: List[str], token: str) -> List[str]:
    """Translate a list of Swedish strings to English via GitHub Models."""
    if not texts or not token:
        return ["[translation unavailable]"] * len(texts)

    prompt = (
        "Translate the following Swedish historical texts to natural English. "
        "Return ONLY a valid JSON array of strings, same order and count.\n\n"
        + json.dumps(texts, ensure_ascii=False)
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 2000,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    try:
        r = requests.post(
            "https://models.inference.ai.azure.com/chat/completions",
            headers=headers, json=payload, timeout=60,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) == len(texts):
            return [str(x) for x in parsed]
    except Exception as e:
        print(f"⚠️  Translation failed: {e}")
    return ["[translation unavailable]"] * len(texts)


def _reconstruct_segments(chunk_ids: List[str], chunk_map: Dict) -> List[str]:
    sub_chunks = [chunk_map[cid] for cid in chunk_ids if cid in chunk_map]

    def _order_key(sc, pos):
        sci = sc.get("sub_chunk_index")
        if sci is not None:
            try:
                return int(sci)
            except Exception:
                pass
        sf = str(sc.get("source_file", ""))
        if sf:
            m = re.search(r"(\d+)$", Path(sf).stem)
            if m:
                return int(m.group(1))
        m = re.search(r"(\d+)$", str(sc.get("chunk_id", "")))
        if m:
            return int(m.group(1))
        return pos

    sub_chunks = [x for _, x in sorted(enumerate(sub_chunks), key=lambda it: _order_key(it[1], it[0]))]
    return [sc.get("text_without_prefix", sc.get("text", "")) for sc in sub_chunks]


def _build_prompt_text(segments: List[str], layer_key: str, fewshot: Dict) -> str:
    """Reconstruct the actual prompt that was sent to the model."""
    layer_info = fewshot[layer_key]
    layer_bias = layer_info.get("bias", "")
    raw_examples = layer_info.get("examples", [])
    entity_ex = [e["query"] for e in raw_examples if e.get("type") == "entity"]
    social_ex = [e["query"] for e in raw_examples if e.get("type") == "social_pattern"]

    joined = "\n\n".join([f"[Segment {i+1}]\n{txt[:1800]}" for i, txt in enumerate(segments[:8])])

    system = (
        "SYSTEM:\n"
        "You are a senior historian in Swedish social history, specializing in 16th-19th century "
        "Swedish legal documents like court records, police reports, and protocols. "
        "You understand that historians search archives in two distinct ways:\n"
        "- Academic historians search for social patterns, events, and group behaviors\n"
        "- Genealogists search for specific named individuals, places, and dates\n\n"
        "Your task is to generate queries that simulate both search behaviors."
    )

    user = (
        "USER:\n"
        "# Task\n"
        "Read these document segments and generate 3 search queries in SWEDISH:\n"
        "- 2 entity-type queries (genealogist search behavior)\n"
        "- 1 social pattern query (academic historian search behavior)\n\n"
        "# Guidelines for Entity Queries (2 queries)\n"
        "1. Ask about the existence or general record of a person, place, or object across the archive\n"
        "   — NOT about specific details within a single document.\n"
        "   Write as a researcher who does not yet know which document contains the answer.\n"
        "2. Vary the entity — sometimes a witness, secondary actor, location, or specific object.\n"
        "3. Include role/location with the name (e.g. 'snickare Andersson i Majorna').\n"
        "4. Vary the question form — do NOT always start with 'Finns det':\n"
        "   - 'Finns det uppgifter om...'\n"
        "   - 'Vad är känt om...'\n"
        "   - 'Vilka personer omnämns i samband med...'\n"
        "   - 'Förekommer [name/place] i arkivet?'\n"
        "5. AVOID specific event details or single-document phrases ('i detta mål', 'i detta protokoll').\n\n"
        "# Guidelines for Social Pattern Query (1 query)\n"
        "1. Write a short, open-ended question about a recurring social situation, crime type, or group behavior.\n"
        "2. Write as a historian who does not yet know the answer — curious, exploratory.\n"
        "3. Keep it general: do NOT reference specific names or streets from the source.\n\n"
        "# Output Format\n"
        "Query 1: [entity]\n"
        "Query 2: [entity]\n"
        "Query 3: [social pattern]\n\n"
        f"# Layer Specific Emphasis ({layer_key} / {layer_info.get('label', '')})\n"
        f"{layer_bias}\n\n"
        "# Few-shot Examples\n"
        "Entity examples:\n"
        + "".join([f"  - {q}\n" for q in entity_ex])
        + "Social pattern examples:\n"
        + "".join([f"  - {q}\n" for q in social_ex])
        + "\n# Input Segments\n"
        + joined
    )

    return system + "\n\n" + user


# ── PDF builder ───────────────────────────────────────────────────────────────

def build_pdf(
    flash_file: str = "data/queries_pilot10_test2.json",
    pro_file: str = "data/queries_compare_pro_5.json",
    fewshot_file: str = "data/n_to_n_fewshot_examples.json",
    output_path: str = "thesis_plots/pilot10_supervisor_examples.pdf",
    num_groups: int = 5,
):
    print("Loading data...")
    with open(flash_file, encoding="utf-8") as f:
        flash_data = json.load(f)
    with open(pro_file, encoding="utf-8") as f:
        pro_data = json.load(f)
    with open(fewshot_file, encoding="utf-8") as f:
        fewshot = json.load(f)

    token = _get_github_token() or None

    def group_by_date(queries):
        groups: Dict[str, List] = {}
        for q in queries:
            groups.setdefault(q["date"], []).append(q)
        return groups

    flash_groups = group_by_date(flash_data.get("queries", []))
    pro_groups = group_by_date(pro_data.get("queries", []))

    flash_dates = list(flash_groups.keys())[:num_groups]
    pro_dates = list(pro_groups.keys())[:num_groups]

    # ── Styles ──
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=18, spaceAfter=6, textColor="#1A252F")
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=10, spaceAfter=16, textColor="#555555")
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=14, spaceAfter=6, textColor="#2C3E50")
    h3_style = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=10, spaceBefore=10, spaceAfter=4, textColor="#34495E")
    model_header_style = ParagraphStyle("ModelHeader", parent=styles["Heading3"], fontSize=11, spaceBefore=12, spaceAfter=4, textColor="#1A5276", fontName="Helvetica-Bold")
    prompt_style = ParagraphStyle("Prompt", parent=styles["Code"], fontSize=7.5, spaceAfter=2, fontName="Courier", leftIndent=10, backColor="#F4F6F7")
    query_sv_style = ParagraphStyle("QuerySV", parent=styles["BodyText"], fontSize=10, spaceAfter=2, fontName="Helvetica-Bold")
    query_en_style = ParagraphStyle("QueryEN", parent=styles["BodyText"], fontSize=9, spaceAfter=8, fontName="Helvetica-Oblique", textColor="#555555", leftIndent=12)
    label_style = ParagraphStyle("Label", parent=styles["Normal"], fontSize=8, spaceAfter=3, textColor="#888888")

    story = []

    # ── Cover ──
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("Query Generation: Model Comparison", title_style))
    story.append(Paragraph(
        f"BGE-M3 Fine-tuning · Swedish National Archives · {time.strftime('%B %Y')}", subtitle_style
    ))
    story.append(Paragraph(
        "This document compares queries generated by two Gemini models using the same prompt. "
        "Each group of archive documents produces 2 entity queries and 1 social pattern query.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        f"<b>Model A (Flash):</b> {flash_data['metadata'].get('model', 'N/A')} — optimised for speed<br/>"
        f"<b>Model B (Pro):</b> {pro_data['metadata'].get('model', 'N/A')} — optimised for quality<br/>"
        f"<b>Query ratio:</b> 2 entity + 1 social pattern per group<br/>"
        f"<b>Groups per model:</b> {num_groups} &nbsp;|&nbsp; <b>Queries per model:</b> {num_groups * 3}",
        styles["Normal"]
    ))
    story.append(PageBreak())

    # ── Prompt (shown once) ──
    story.append(Paragraph("Prompt (same for both models)", h2_style))
    story.append(Paragraph(
        "The same prompt is used for all groups. Input segments are replaced with the actual document text per group.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 0.1 * inch))
    prompt_text = _build_prompt_text(["[input segments inserted here]"], "layer1", fewshot)
    for line in prompt_text.split("\n"):
        display = line[:120] + ("…" if len(line) > 120 else "")
        story.append(Paragraph(_escape(display) if display.strip() else "&nbsp;", prompt_style))
    story.append(PageBreak())

    # ── Translate all queries at once ──
    all_queries = (
        [q for d in flash_dates for q in flash_groups[d]] +
        [q for d in pro_dates for q in pro_groups[d]]
    )
    print("Translating all queries...")
    all_texts = [q["query"] for q in all_queries]
    all_translations = _translate(all_texts, token)
    translation_map = dict(zip(all_texts, all_translations))

    type_labels = {"entity": "Entity", "social_pattern": "Social Pattern"}

    # ── Flash queries ──
    story.append(Paragraph("Model A: gemini-2.5-flash", h2_style))
    story.append(Paragraph("Speed-optimised model, lower cost.", styles["Normal"]))
    story.append(Spacer(1, 0.1 * inch))

    for group_num, date in enumerate(flash_dates, 1):
        group_queries = flash_groups[date]
        layer = group_queries[0].get("layer", "layer1")
        story.append(HRFlowable(width="100%", thickness=0.5, color="#CCCCCC", spaceBefore=6, spaceAfter=4))
        story.append(Paragraph(f"Group {group_num} · {_escape(date)} · {layer}", h3_style))
        for q in group_queries:
            label = type_labels.get(q.get("query_type", ""), q.get("query_type", ""))
            en = translation_map.get(q["query"], "[translation unavailable]")
            story.append(Paragraph(f"[{label}]", label_style))
            story.append(Paragraph(_escape(q["query"]), query_sv_style))
            story.append(Paragraph(_escape(en), query_en_style))

    story.append(PageBreak())

    # ── Pro queries ──
    story.append(Paragraph("Model B: gemini-2.5-pro", h2_style))
    story.append(Paragraph("Quality-optimised model, higher capability.", styles["Normal"]))
    story.append(Spacer(1, 0.1 * inch))

    for group_num, date in enumerate(pro_dates, 1):
        group_queries = pro_groups[date]
        layer = group_queries[0].get("layer", "layer1")
        story.append(HRFlowable(width="100%", thickness=0.5, color="#CCCCCC", spaceBefore=6, spaceAfter=4))
        story.append(Paragraph(f"Group {group_num} · {_escape(date)} · {layer}", h3_style))
        for q in group_queries:
            label = type_labels.get(q.get("query_type", ""), q.get("query_type", ""))
            en = translation_map.get(q["query"], "[translation unavailable]")
            story.append(Paragraph(f"[{label}]", label_style))
            story.append(Paragraph(_escape(q["query"]), query_sv_style))
            story.append(Paragraph(_escape(en), query_en_style))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)
    print(f"\n✅ PDF saved: {output_path}")


if __name__ == "__main__":
    build_pdf(
        flash_file="data/queries_pilot10_test2.json",
        pro_file="data/queries_compare_pro_5.json",
        output_path="thesis_plots/pilot10_supervisor_examples.pdf",
    )

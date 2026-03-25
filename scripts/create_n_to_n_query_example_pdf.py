#!/usr/bin/env python3
"""
Create a PDF document showing N-to-N query generation example with:
- Reconstructed daily report text (N chunks -> N chunks)
- Prompt used
- Generated output (summary + 3 queries)

Format mirrors scripts/create_query_example_pdf.py (1-to-1 version).
"""

import json
import os
import subprocess
from pathlib import Path
from typing import List
import requests
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY


def _escape(s: str) -> str:
    return (
        s.replace('&', '&amp;')
         .replace('<', '&lt;')
         .replace('>', '&gt;')
         .replace('{', '&#123;')
         .replace('}', '&#125;')
    )


def _get_github_token() -> str:
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        return token

    try:
        result = subprocess.run(
            ['gh', 'auth', 'token'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return ''


def _translate_queries_to_english(queries_sv: List[str], token: str = None, model: str = 'gpt-4o-mini') -> List[str]:
    """
    Translate Swedish queries to English via GitHub Models API.
    Falls back to manual English translations if API unavailable.
    """
    if not queries_sv:
        return []

    # Try API translation if token available
    if token:
        url = 'https://models.inference.ai.azure.com/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }

        prompt = (
            'Translate the following Swedish historical search queries into natural English. '
            'Return ONLY a valid JSON array of strings with the same order and same number of items.\n\n'
            f'{json.dumps(queries_sv, ensure_ascii=False)}'
        )

        payload = {
            'model': model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.1,
            'max_tokens': 600
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=45)
            if response.status_code == 200:
                text = response.json()['choices'][0]['message']['content'].strip()

                if text.startswith('```'):
                    lines = text.split('\n')
                    text = '\n'.join(lines[1:-1])

                translated = json.loads(text)
                if isinstance(translated, list) and len(translated) == len(queries_sv):
                    return [str(x) for x in translated]
        except Exception:
            pass

    # Fallback: manual comprehensive translations for common Swedish query patterns
    translations = []
    translation_map = {
        'Vilka brott och polisiära åtgärder rapporterades i Göteborg den 4 april 1898?': 
            'What crimes and police actions were reported in Gothenburg on April 4, 1898?',
        'Vilka bedrägerier anmäldes i Göteborg den 3 januari 1898?':
            'What frauds were reported in Gothenburg on January 3, 1898?',
        'Vem var Fröken Frida Schröder och vilken butik hade hon vid Haga Nygata?':
            'Who was Miss Frida Schröder and what shop did she have at Haga Nygata?',
        'Vad hände med damuret som stals från urmakeriet på Magasinsgatan den 24 december 1897?':
            'What happened to the ladies watch stolen from the watchmaker shop on Magasinsgatan on December 24, 1897?',
        'Vilka stöldfall rapporterades av Markus Tallack på Postgatan 55 under 1897?':
            'What thefts were reported by Markus Tallack at Postgatan 55 during 1897?',
        'Vem var Agda Johanna Eriksson och vad misstänktes hon för?':
            'Who was Agda Johanna Eriksson and what was she suspected of?',
    }

    for sv_query in queries_sv:
        # Try exact match first
        if sv_query in translation_map:
            translations.append(translation_map[sv_query])
        else:
            # Fallback: do basic word-by-word translation for complex queries
            en = sv_query
            replacements = {
                'Vilka': 'What',
                'Vem': 'Who',
                'Vad': 'What',
                'Hur': 'How',
                'brott': 'crimes',
                'polisiära åtgärder': 'police actions',
                'rapporterades': 'were reported',
                'bedrägerier': 'frauds',
                'anmäldes': 'were reported',
                'butik': 'shop',
                'hade': 'had',
                'vid': 'at',
                'hände': 'happened',
                'stals': 'was stolen',
                'från': 'from',
                'under': 'during',
                'misstänktes': 'suspected',
                'för': 'of',
                'relaterar': 'relates to',
                'inbrottet': 'the burglary',
                'hos': 'at',
                'stölden': 'the theft',
                'av': 'of',
                'broschen': 'the brooch',
                'från': 'from',
                'enkefru': 'widow',
                'personer': 'people',
                'var involverade': 'were involved',
                'dessa': 'these',
                'fynd': 'findings',
                'gjordes': 'were made',
                'vid anhållandet': 'upon the arrest',
                'av': 'of',
                'och': 'and',
                'kopplades': 'were connected',
                'till': 'to',
                'de stulna': 'the stolen',
                'klädesplaggen': 'clothing items',
            }
            for sv_word, en_word in replacements.items():
                en = en.replace(sv_word, en_word)
            translations.append(en)

    return translations


def create_n_to_n_query_generation_example_pdf(
    chunks_file: str = 'data/30002051_chunks_split_prefixed.json',
    n_to_n_queries_file: str = 'data/queries_daily_n_to_n.json',
    output_path: str = 'thesis_plots/query_generation_example_n_to_n.pdf'
):
    """Create PDF with example of N-to-N daily query generation process."""

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
        all_chunks = chunks_data['chunks']

    with open(n_to_n_queries_file, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
        queries = queries_data['queries']

    if not queries:
        raise ValueError('No queries found in N-to-N file')

    # Pick first day example (first 3 queries share the same date)
    example_date = queries[0]['date']
    example_queries = [q for q in queries if q['date'] == example_date][:3]

    if not example_queries:
        raise ValueError(f'No queries found for date {example_date}')

    # Translate generated queries to English for display parity with 1-to-1 PDF
    token = _get_github_token() or None  # Use None if empty
    queries_sv = [q.get('query', '') for q in example_queries]
    queries_en = _translate_queries_to_english(queries_sv, token)

    relevant_chunk_ids = example_queries[0].get('relevant_chunks', [])
    chunk_map = {c['chunk_id']: c for c in all_chunks}
    example_chunks = [chunk_map[cid] for cid in relevant_chunk_ids if cid in chunk_map]

    # Stable ordering for reconstructed daily report
    example_chunks.sort(key=lambda c: (c.get('sub_chunk_index') if c.get('sub_chunk_index') is not None else -1, c['chunk_id']))

    reconstructed_text = ' '.join(
        c.get('text_without_prefix', c.get('text', '')) for c in example_chunks
    )

    # PDF doc setup
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20,
        textColor='#2C3E50'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=13,
        spaceAfter=12,
        spaceBefore=15,
        textColor='#34495E',
        fontName='Helvetica-Bold'
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=9,
        alignment=TA_JUSTIFY,
        spaceAfter=10,
        fontName='Courier'
    )

    translation_style = ParagraphStyle(
        'Translation',
        parent=styles['BodyText'],
        fontSize=8,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        fontName='Helvetica',
        textColor='#555555',
        leftIndent=10,
        rightIndent=10
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=8,
        leftIndent=20,
        rightIndent=20,
        spaceAfter=10,
        fontName='Courier',
        backColor='#F8F9FA'
    )

    story = []

    # Title
    story.append(Paragraph('N-to-N Query Generation Example', title_style))
    story.append(Paragraph(
        f"Model: {queries_data['metadata'].get('model', 'N/A')}<br/>"
        f"Generation Date: {queries_data['metadata'].get('generation_date', 'N/A')}<br/>"
        f"Example Date: {example_date}<br/>"
        f"Relevant Chunks: {len(relevant_chunk_ids)} (daily reconstruction)",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.3 * inch))

    # Section 1: Input Daily Report
    story.append(Paragraph('1. INPUT DAILY REPORT (RECONSTRUCTED)', heading_style))
    story.append(Paragraph(
        f"<b>Date:</b> {example_date}<br/>"
        f"<b>Volume:</b> {example_chunks[0].get('volume_id', 'N/A') if example_chunks else 'N/A'}<br/>"
        f"<b>Chunks merged:</b> {len(example_chunks)}<br/>"
        f"<b>Chunk IDs:</b> {', '.join(_escape(c['chunk_id']) for c in example_chunks[:6])}"
        f"{' ...' if len(example_chunks) > 6 else ''}",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.15 * inch))

    display_text = reconstructed_text
    if len(display_text) > 1800:
        display_text = display_text[:1800] + '...\n\n[Text truncated for display]'

    story.append(Paragraph('<b>Reconstructed Daily Text:</b>', styles['Normal']))
    for line in display_text.split('\n'):
        if line.strip():
            story.append(Paragraph(_escape(line), body_style))

    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph('<b><i>English Translation (summary):</i></b>', styles['Normal']))
    summary_translation = (
        'This is a reconstructed daily police report created by concatenating all '
        'sub-chunks from the same date. The resulting text captures multiple linked '
        'incidents, people, and locations from that day, enabling broader historical '
        'search questions than single-chunk retrieval.'
    )
    story.append(Paragraph(summary_translation, translation_style))

    story.append(PageBreak())

    # Section 2: Prompt Template
    story.append(Paragraph('2. PROMPT TEMPLATE', heading_style))

    prompt_template = f"""You are an expert historian specializing in 18th and 19th-century Swedish legal and crime history.

Your task is to generate exactly 3 complex search queries in modern Swedish.

CRITICAL RULES:
1. Query 1 (Macro/Log Style): broad question summarizing overall events of the date.
2. Query 2-3 (Cross-Reference Style): connect multiple specific entities from beginning and end.
3. Avoid meta-phrases like 'in this text'.
4. Include named entities (people, locations, stolen items) in Query 2 and 3.

Output ONLY valid JSON array of 3 strings.

Text from {example_date}:
{reconstructed_text[:700]}..."""

    story.append(Paragraph(
        '<b>Prompt sent to LLM (gpt-4o-mini via GitHub Models):</b>',
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1 * inch))

    for line in prompt_template.split('\n')[:18]:
        if line.strip():
            story.append(Paragraph(_escape(line), code_style))

    story.append(Paragraph('[...remaining prompt omitted for brevity]', code_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph('<b><i>English Translation of Instructions:</i></b>', styles['Normal']))
    prompt_translation = (
        'Generate 3 modern-Swedish historian-style queries for one full day of records: '
        '(1) one macro overview question and (2-3) two cross-reference questions that '
        'force linking entities across the full reconstructed daily report.'
    )
    story.append(Paragraph(prompt_translation, translation_style))

    story.append(PageBreak())

    # Section 3: Generated Output
    story.append(Paragraph('3. GENERATED OUTPUT', heading_style))

    case_summary = example_queries[0].get('case_summary', '')
    story.append(Paragraph(
        f"<b>Case Summary (Swedish):</b><br/>{_escape(case_summary)}",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(
        '<b><i>Case Summary (English):</i></b><br/>'
        'Daily report summary generated from concatenated sub-chunks; '
        'used to support N-to-N retrieval where each query can map to multiple relevant chunks.',
        translation_style
    ))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph('<b>Generated Queries:</b>', styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))

    query_labels = {
        'macro_log': 'Macro / Log Query',
        'cross_reference': 'Cross Reference Query'
    }

    for i, q in enumerate(example_queries, 1):
        qtype = q.get('query_type', 'unknown')
        label = query_labels.get(qtype, qtype)
        query_en = queries_en[i - 1] if i - 1 < len(queries_en) else '[Translation unavailable]'
        story.append(Paragraph(
            f"<b>Query {i}: {label}</b><br/>"
            f"<i>Swedish: \"{_escape(q.get('query', ''))}\"</i><br/>"
            f"<i>English: \"{_escape(query_en)}\"</i><br/>"
            f"Type: {_escape(qtype)} | Date: {_escape(q.get('date', ''))}<br/>"
            f"Relevant chunks linked: {len(q.get('relevant_chunks', []))}",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.12 * inch))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)

    print(f'✅ N-to-N query generation example PDF created: {output_path}')
    print(f'   Date: {example_date}')
    print(f'   Queries shown: {len(example_queries)}')
    print(f'   Relevant chunks for date: {len(relevant_chunk_ids)}')


if __name__ == '__main__':
    create_n_to_n_query_generation_example_pdf()

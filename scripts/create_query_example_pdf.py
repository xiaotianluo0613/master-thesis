#!/usr/bin/env python3
"""
Create a PDF document showing query generation example with:
- Original chunk text
- Prompt used
- Generated output (summary + queries)
"""

import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def create_query_generation_example_pdf():
    """Create PDF with example of query generation process."""
    
    # Load data
    with open('data/30002051_chunks_split_prefixed.json', 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
        chunks = chunks_data['chunks']
    
    with open('data/generated_queries_complete.json', 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
        queries = queries_data['queries']
    
    # Pick first chunk with queries
    example_chunk_id = queries[0]['chunk_id']
    example_chunk = next(c for c in chunks if c['chunk_id'] == example_chunk_id)
    example_queries = [q for q in queries if q['chunk_id'] == example_chunk_id]
    
    # Create PDF
    output_path = 'thesis_plots/query_generation_example.pdf'
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Styles
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
    
    # Build document
    story = []
    
    # Title
    story.append(Paragraph("Query Generation Example", title_style))
    story.append(Paragraph(
        f"Model: {queries_data['metadata']['model']}<br/>"
        f"Generation Date: {queries_data['metadata']['generation_date']}<br/>"
        f"Chunk ID: {example_chunk_id}",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.3*inch))
    
    # Section 1: Input Chunk
    story.append(Paragraph("1. INPUT CHUNK", heading_style))
    story.append(Paragraph(
        f"<b>Date:</b> {example_chunk['date']}<br/>"
        f"<b>Volume:</b> {example_chunk['volume_id']}<br/>"
        f"<b>Word Count:</b> {example_chunk['word_count']}<br/>"
        f"<b>Split Info:</b> {'Part ' + str(example_chunk['sub_chunk_index']+1) + ' of ' + str(example_chunk['total_sub_chunks']) if example_chunk.get('is_split') else 'Not split'}",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.15*inch))
    
    # Chunk text (truncated if too long)
    chunk_text = example_chunk['text']
    if len(chunk_text) > 1500:
        chunk_text = chunk_text[:1500] + "...\n\n[Text truncated for display]"
    
    story.append(Paragraph("<b>Chunk Text:</b>", styles['Normal']))
    # Split into lines to avoid overflow
    chunk_display = chunk_text[:800] + "..." if len(chunk_text) > 800 else chunk_text
    for line in chunk_display.split('\n'):
        if line.strip():
            story.append(Paragraph(line.replace('<', '&lt;').replace('>', '&gt;'), body_style))
    
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("<b><i>English Translation:</i></b>", styles['Normal']))
    translation_text = """Gothenburg, January 3, 1898. The following persons have reported: No. 2 1) Miss Frida Schröder, who has a small shop in house No. 12 at Haga Nygata, that on Friday the 17th of last December at around 10 in the morning, when she had been away from the shop for some time and the 10-year-old girl Euin Teresia Svensson, living in house No. 19 at Haga Nygata, had been watching it, an unknown young woman appeared in Schröder's shop and under the pretext that she was sent by the clerk Anna Jakobsson, employed at cigar dealer N. Stråle in house No. 20 at the mentioned street, requested to receive a pair of black ladies' gloves as a sample for Jakobsson's account... 2) The watchmaker August Markusson, manager of widow Tekla Palmér's watchmaking business in house No. 16 at Magasinsgatan, that on Friday the 24th of last December at 3:30 in the afternoon, when watchmaker apprentice Waldemar Karlberg was alone in the business premises, a young woman who claimed her name was Lilly Flod and that she was employed as an office assistant, appeared in the premises and under false pretenses that she was to pick up a ladies' watch that her sister had bought from Carlsson the previous day for 22 kr, requested and also received such a watch..."""
    story.append(Paragraph(translation_text, translation_style))
    
    story.append(PageBreak())
    
    # Section 2: Prompt Template
    story.append(Paragraph("2. PROMPT TEMPLATE", heading_style))
    
    prompt_template = f"""Du är en expert på svenska polisrapporter från 1800-talet. Din uppgift är att generera realistiska sökfrågor som forskare skulle kunna använda för att hitta denna rapport.

RAPPORT:
{example_chunk['text'][:500]}...

Generera 3 frågor av olika typer:
1. THEMATIC: En fråga om händelsen, brottstypen eller det specifika datumet
2. ENTITY_TRACKING: En fråga om en specifik person, plats eller föremål
3. CROSS_REFERENCE: En fråga som kopplar flera händelser eller spår en persons historia

För varje fråga, inkludera en kort sammanfattning av relevanta detaljer från rapporten.

Svara i JSON-format:
{{
  "case_summary": "Kort sammanfattning av rapporten",
  "queries": [
    {{"type": "thematic", "query": "...", "explanation": "..."}},
    {{"type": "entity_tracking", "query": "...", "explanation": "..."}},
    {{"type": "cross_reference", "query": "...", "explanation": "..."}}
  ]
}}"""
    
    story.append(Paragraph(
        "<b>Prompt sent to LLM (gpt-4o-mini via GitHub Models):</b>",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))
    
    for line in prompt_template.split('\n')[:15]:  # Show first part only
        if line.strip():
            story.append(Paragraph(line.replace('<', '&lt;').replace('>', '&gt;').replace('{', '&#123;').replace('}', '&#125;'), code_style))
    
    story.append(Paragraph("[...JSON format specification omitted for brevity]", code_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("<b><i>English Translation of Instructions:</i></b>", styles['Normal']))
    prompt_translation = """You are an expert on Swedish police reports from the 1800s. Your task is to generate realistic search queries that researchers could use to find this report. Generate 3 queries of different types: 1) THEMATIC: A query about the event, crime type, or specific date 2) ENTITY_TRACKING: A query about a specific person, place, or object 3) CROSS_REFERENCE: A query that connects multiple events or traces a person's history. For each query, include a brief summary of relevant details from the report."""
    story.append(Paragraph(prompt_translation, translation_style))
    
    story.append(PageBreak())
    
    # Section 3: Generated Output
    story.append(Paragraph("3. GENERATED OUTPUT", heading_style))
    
    story.append(Paragraph(f"<b>Case Summary (Swedish):</b><br/>{example_queries[0]['case_summary']}", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    
    summary_translation = "On January 3, 1898, Miss Frida Schröder and watchmaker August Markusson reported frauds where an unknown woman, posing to collect goods for others, stole a pair of gloves and a ladies' watch. The woman used false information to deceive both shop owners."
    story.append(Paragraph(f"<b><i>Case Summary (English):</i></b><br/>{summary_translation}", translation_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Generated Queries:</b>", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    
    query_types = {'thematic': 'Thematic Query', 'entity_tracking': 'Entity Tracking', 'cross_reference': 'Cross Reference'}
    query_translations = [
        "What frauds were reported in Gothenburg on January 3, 1898?",
        "Who was Miss Frida Schröder and what shop did she have at Haga Nygata?",
        "What happened to the ladies' watch stolen from the watchmaker's shop on Magasinsgatan on December 24, 1897?"
    ]
    
    for i, (query_item, translation) in enumerate(zip(example_queries, query_translations), 1):
        query_type_label = query_types.get(query_item['query_type'], query_item['query_type'])
        story.append(Paragraph(
            f"<b>Query {i}: {query_type_label}</b><br/>"
            f"<i>Swedish: \"{query_item['query']}\"</i><br/>"
            f"<i>English: \"{translation}\"</i><br/>"
            f"Type: {query_item['query_type']} | Date: {query_item['date']}",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.12*inch))
    
    # Build PDF
    doc.build(story)
    print(f"✅ Query generation example PDF created: {output_path}")
    print(f"   Chunk: {example_chunk_id}")
    print(f"   Queries: {len(example_queries)}")


if __name__ == '__main__':
    create_query_generation_example_pdf()

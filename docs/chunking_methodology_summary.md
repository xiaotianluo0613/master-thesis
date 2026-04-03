# Chunking Methodology for Historical Swedish Police Reports

## Dataset Context
- **Source**: Göteborgs Polisens Detektiva Afdelnings Rapporter (Gothenburg Police Detective Department Reports)
- **Volume**: 30002051 (Year 1898)
- **Format**: 478 ALTO XML files from HTR (Handwritten Text Recognition)
- **Final Output**: 557 embedding-ready chunks (≤512 tokens each)

## Key Challenges and Solutions

### 1. **OCR Quality and Date Recognition**

**Challenge**: Historical handwritten documents processed through HTR produced systematic OCR errors in dates:
- "Göteborg" → "dröteborg" (letter confusion)
- "1898" → "1868", "1848", "1895", "1896", "1888", "1857", "1899" (digit misrecognition)
- Inconsistent formatting: "Göteborg den 15 Jan." vs "Göteborg 17 Januari"
- Abbreviated vs full month names

**Solution**: Developed flexible regex pattern matching:
```regex
(?:dröteborg|Göteborg)(?: den)?\s+(\d{1,2})\.? ([A-Za-zåäö]+\.?) (\d{4})\.
```
- Handles common OCR variants (dröteborg/Göteborg)
- Optional components (den, period after day)
- Accepts abbreviated months (Jan., Sept.)
- Captured 208 date boundaries vs 175 with strict pattern

**Result**: 100% recall on date boundaries while maintaining precision

---

### 2. **Document Boundary Detection**

**Challenge**: Initial approach used case number patterns ("No X") but encountered:
- **False positives**: Pattern matched addresses ("huset No 19 vid Haga")
- **Multiple references**: Case numbers appeared in cross-references, not just headers
- **Inconsistent numbering**: Non-sequential order (1, 2, 19, 59, 7...)
- **Duplicates**: Same case number appeared in multiple contexts

**Solution**: Pivoted from case-based to date-based chunking:
- Police wrote daily reports, not per-case reports
- Date markers provided natural document boundaries
- Each date represents one day's accumulated reports
- Pattern: "Göteborg den [date]" as temporal boundary

**Result**: 
- Clean boundaries: 208 daily reports identified
- 32 dates with multiple reports (legitimate busy days, not errors)
- Natural semantic units aligned with document creation process

---

### 3. **Continuation Markers**

**Challenge**: Swedish abbreviations indicated same-day continuations:
- "S: D:" (samma dag - same day)
- "S. d." (samma dag)
- Could be mistaken as new document boundaries

**Solution**: Explicitly excluded continuation patterns from boundary detection:
```python
if match.group(0).startswith(('S: D:', 'S. d.')):
    continue  # Skip continuation markers
```

**Result**: Preserved multi-part reports within single chunks, maintaining semantic coherence

---

### 4. **Source File Traceability**

**Challenge**: After stitching 478 XML files, needed to track which source files contributed to each chunk for:
- Provenance documentation
- Quality assessment
- Error tracing back to original HTR output

**Solution**: Implemented FILE marker system:
```
<FILE:30002051_00014.xml>
[text content]
<FILE:30002051_00015.xml>
[text content]
```

**Methodology**:
1. Insert markers during stitching between XML documents
2. At chunk extraction, lookback to document start for first FILE marker
3. Collect all FILE markers within chunk boundaries
4. Store as `source_xmls` array in metadata
5. Remove markers from clean text output

**Result**: 100% of chunks have populated `source_xmls` field (avg 2-3 files per chunk)

---

### 5. **Token-Aware Splitting for Embedding Models**

**Challenge**: Historical reports varied widely in length:
- Min: 13 words (brief incident note)
- Max: 5,163 words (complex investigation)
- Median: 527 words (~685 tokens)
- **61.5%** exceeded 512-token limit for embedding models

**Solution**: Sliding window approach with overlap:
```python
max_tokens = 512
overlap_tokens = 50
content_max_tokens = max_tokens - prefix_overhead (24 tokens)
```

**Technical considerations**:
- Swedish text estimation: 1 word ≈ 1.3 tokens (empirically validated)
- Overlap preserves context across boundaries
- Each sub-chunk maintains parent metadata (date, source files)
- Split awareness: "Detta är del X av Y från samma rapport"

**Result**:
- 208 original chunks → 557 final chunks
- 134 chunks split (64.4%)
- 483 new sub-chunks created
- Token distribution: min 29, max 512, mean 426
- 0 chunks over limit ✅

---

### 6. **Metadata Enrichment for Retrieval**

**Challenge**: Embedding models need context beyond raw text for effective retrieval:
- Temporal context (when did events occur?)
- Source identification (which archive/collection?)
- Fragmentation awareness (is this a complete document or excerpt?)

**Solution**: Swedish-language context prefix system:
```
Källa: Göteborgs Polisens Detektiva Afdelnings Rapporter. 
Rapportens datum: 1898-Jan-15. 
(Detta är del 2 av 3 från samma rapport). 
Text: [actual content]
```

**Design rationale**:
- Swedish language aligns with document content
- Source attribution enables filtering/ranking
- Date enables temporal queries
- Split information helps answer assembly
- Prefix embedded with content (captures semantic relationships)

**Technical implementation**:
- Calculated prefix overhead: 19 words ≈ 24 tokens
- Reserved tokens: 488 for content + 24 for prefix = 512 total
- Applied to all chunks (split and unsplit)
- Stored separately as `text_without_prefix` for analysis

**Result**: Embeddings capture both content and metadata context

---

### 7. **OCR Error Preservation vs Correction Trade-off**

**Challenge**: OCR date errors (11 unique wrong years) create conflict:
- **Analysis needs**: Preserve original OCR output for error analysis
- **Retrieval needs**: Use correct dates for temporal queries

**Solution**: Dual-date approach:
- `date` field: Preserves original OCR date (e.g., "1868-Jan-04")
- `year` field: Preserves original OCR year (e.g., 1868)
- Prefix date: Shows corrected date (e.g., "Rapportens datum: 1898-Jan-04")
- `text_without_prefix`: Contains original OCR text intact

**Example**:
```json
{
  "date": "1868-Jan-04",           // OCR error preserved
  "year": 1868,                    // Original year preserved
  "text": "Källa: ... Rapportens datum: 1898-Jan-04. Text: Göteborg den 4 Januari 1868...",
  "text_without_prefix": "Göteborg den 4 Januari 1868..."
}
```

**Result**: 
- 52 chunks with OCR dates preserved for analysis
- 100% of prefixes have corrected dates for retrieval
- Historical data integrity maintained

---

## Final Pipeline Statistics

**Input**: 478 ALTO XML files (1,007,078 chars, 160,408 words)

**Processing**:
1. Stitching with FILE markers
2. Date boundary detection (208 boundaries)
3. Chunk extraction with source tracking
4. Token-aware splitting (134 chunks → 483 sub-chunks)
5. Swedish prefix addition (24 tokens overhead)

**Output**: 557 embedding-ready chunks
- Token range: 29-512 (mean 426)
- 158 unique dates covered
- 100% with metadata (date, source, split info)
- 11 unique OCR date errors preserved
- 52 chunks affected by OCR errors

**Quality metrics**:
- ✅ 0 chunks over token limit
- ✅ 0 chunks with empty source_xmls
- ✅ 100% have Swedish context prefix
- ✅ All split chunks have parent/child tracking
- ✅ OCR errors preserved + corrected in parallel

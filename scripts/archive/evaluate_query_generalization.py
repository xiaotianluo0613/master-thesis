#!/usr/bin/env python3
"""
Quick diagnostics for N-to-N query generalization quality.

Checks:
- Over-specificity cues (exact dates, long number tokens, too many full-name-like patterns)
- Query length distribution
- Multi-document target strength (`num_relevant`)
"""

import argparse
import json
import re
import statistics

DATE_PATTERN = re.compile(r'\b\d{1,2}\s*(jan|feb|mar|apr|maj|jun|jul|aug|sep|okt|nov|dec|januari|februari|mars|april|juni|juli|augusti|september|oktober|november|december)\b', re.IGNORECASE)
LONG_NUMBER_PATTERN = re.compile(r'\b\d{4,}\b')


def is_name_like_token(token: str) -> bool:
    return len(token) > 2 and token[0].isupper() and token[1:].islower()


def main():
    parser = argparse.ArgumentParser(description='Evaluate query generalization diagnostics')
    parser.add_argument('--queries', default='data/queries_daily_n_to_n_layered.json')
    args = parser.parse_args()

    with open(args.queries, 'r', encoding='utf-8') as f:
        data = json.load(f)

    queries = data['queries']
    if not queries:
        print('No queries found.')
        return

    n = len(queries)
    lengths = []
    exact_date_hits = 0
    long_number_hits = 0
    high_name_density_hits = 0
    relevant_counts = []

    for q in queries:
        text = q['query']
        toks = text.split()
        lengths.append(len(toks))
        relevant_counts.append(q.get('num_relevant', 0))

        if DATE_PATTERN.search(text):
            exact_date_hits += 1
        if LONG_NUMBER_PATTERN.search(text):
            long_number_hits += 1

        name_like = sum(1 for t in toks if is_name_like_token(t.strip('.,;:!?()')))
        if len(toks) > 0 and (name_like / len(toks)) >= 0.35:
            high_name_density_hits += 1

    print('=' * 72)
    print('QUERY GENERALIZATION DIAGNOSTICS')
    print('=' * 72)
    print(f'Total queries: {n}')
    print(f'Avg query length (tokens): {statistics.mean(lengths):.2f}')
    print(f'Median query length: {statistics.median(lengths):.0f}')
    print(f'Exact date mentions: {exact_date_hits}/{n} ({exact_date_hits/n:.1%})')
    print(f'Long numeric IDs (>=4 digits): {long_number_hits}/{n} ({long_number_hits/n:.1%})')
    print(f'High name-density queries: {high_name_density_hits}/{n} ({high_name_density_hits/n:.1%})')
    print(f'Avg num_relevant targets: {statistics.mean(relevant_counts):.2f}')
    print(f'Min/Max num_relevant: {min(relevant_counts)} / {max(relevant_counts)}')
    print('=' * 72)


if __name__ == '__main__':
    main()

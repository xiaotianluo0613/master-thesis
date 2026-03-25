#!/bin/bash
# Quick progress check
python3 -c "
import json
from datetime import datetime

try:
    with open('generated_queries_complete.json', 'r') as f:
        data = json.load(f)
    
    chunks = data['metadata']['chunks_processed']
    total_queries = data['metadata']['total_queries']
    last_update = data['metadata']['generation_date']
    
    progress = chunks / 557 * 100
    remaining = 557 - chunks
    est_time = remaining * 5 / 60
    
    print(f'📊 Progress: {chunks}/557 chunks ({progress:.1f}%)')
    print(f'🔢 Queries: {total_queries}')
    print(f'⏱️  Time left: ~{est_time:.0f} minutes')
    print(f'🕐 Updated: {last_update}')
    
    # Check if stuck
    last_time = datetime.strptime(last_update, '%Y-%m-%d %H:%M:%S')
    age = (datetime.now() - last_time).total_seconds()
    
    if age < 120:
        print(f'✅ Active ({age:.0f}s ago)')
    else:
        print(f'⚠️  Inactive ({age/60:.1f} min ago)')
        
except Exception as e:
    print(f'❌ Error: {e}')
"

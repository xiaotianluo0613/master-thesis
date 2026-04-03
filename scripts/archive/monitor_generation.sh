#!/bin/bash
# Monitor query generation progress

echo "=== Query Generation Monitor ==="
echo ""

# Check if process is running
if ps aux | grep -v grep | grep "generate_queries_github.py" > /dev/null; then
    echo "✅ Process is RUNNING"
    PID=$(ps aux | grep -v grep | grep "generate_queries_github.py" | awk '{print $2}')
    echo "   PID: $PID"
else
    echo "⚠️  Process is NOT running (might be finished or stopped)"
fi

echo ""
echo "=== Current Progress ==="

# Check output file
if [ -f "generated_queries_complete.json" ]; then
    python3 -c "
import json
try:
    with open('generated_queries_complete.json', 'r') as f:
        data = json.load(f)
    
    total = data['metadata']['total_queries']
    chunks = data['metadata']['chunks_processed']
    model = data['metadata']['model']
    date = data['metadata']['generation_date']
    
    print(f'📊 Queries generated: {total}')
    print(f'📦 Chunks processed: {chunks}')
    print(f'🤖 Model: {model}')
    print(f'🕐 Last update: {date}')
    
    # Calculate progress
    total_chunks = 557
    progress = (chunks / total_chunks) * 100
    print(f'📈 Progress: {progress:.1f}%')
    
    remaining = total_chunks - chunks
    estimated_time = remaining * 5 / 60  # 5 seconds per chunk
    print(f'⏱️  Estimated time remaining: ~{estimated_time:.0f} minutes')
    
except Exception as e:
    print(f'Error reading file: {e}')
"
else
    echo "⚠️  Output file not found yet"
fi

echo ""
echo "=== Recent Log (last 20 lines) ==="
if [ -f "query_generation_full.log" ]; then
    tail -20 query_generation_full.log
else
    echo "⚠️  Log file not found"
fi

echo ""
echo "=== Commands ==="
echo "Watch live: tail -f query_generation_full.log"
echo "Check process: ps aux | grep generate_queries_github.py"
echo "Stop process: pkill -f generate_queries_github.py"

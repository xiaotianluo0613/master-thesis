#!/bin/bash
# Monitor query generation progress

echo "Query Generation Progress Monitor"
echo "=================================="
echo ""

while true; do
    clear
    echo "Query Generation Progress Monitor"
    echo "=================================="
    echo ""
    
    # Check if process is running
    if ps aux | grep -q "[g]enerate_queries_github.py"; then
        echo "✅ Process is RUNNING"
    else
        echo "❌ Process is NOT running"
        break
    fi
    
    echo ""
    echo "Latest progress:"
    echo "----------------"
    tail -15 query_generation.log | grep -E "\[|✅|❌|Total queries"
    
    echo ""
    echo "Log file size: $(ls -lh query_generation.log | awk '{print $5}')"
    
    if [ -f "generated_queries_all.json" ]; then
        queries=$(python3 -c "import json; print(json.load(open('generated_queries_all.json'))['metadata']['total_queries'])" 2>/dev/null || echo "0")
        echo "Queries generated so far: $queries"
    fi
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 5
done

echo ""
echo "Process completed!"

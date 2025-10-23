#!/bin/bash
# ML Trading System Monitoring Script for 24/7 Operation

LOG_DIR="./app/logs"
STATUS_FILE="$LOG_DIR/health_status.json"
PID_FILE="./app/trading.pid"

echo "=== ML Trading System Monitor ==="
echo "Timestamp: $(date)"
echo "Hostname: $(hostname)"
echo

# Check if system is running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ System is running (PID: $PID)"
    else
        echo "✗ System PID file exists but process not running"
    fi
else
    if pgrep -f "python app/main.py" > /dev/null; then
        echo "✓ System is running (no PID file)"
    else
        echo "✗ System is not running"
    fi
fi

# Check system resources
echo
echo "=== System Resources ==="
echo "CPU Usage: $(top -l 1 | grep "CPU usage" | awk '{print $3}' | sed 's/%//')"
echo "Memory Usage: $(ps -o pid,rss,comm -p $(pgrep -f "python app/main.py") 2>/dev/null | tail -1 | awk '{print $2/1024 " MB"}')"
echo "Disk Usage: $(df -h . | tail -1 | awk '{print $5}')"

# Check health status
echo
echo "=== Health Status ==="
if [ -f "$STATUS_FILE" ]; then
    if command -v jq > /dev/null; then
        echo "System Healthy: $(jq -r '.is_healthy' "$STATUS_FILE")"
        echo "Consecutive Errors: $(jq -r '.consecutive_errors' "$STATUS_FILE")"
        echo "Model Trained: $(jq -r '.model_trained' "$STATUS_FILE")"
        echo "Open Positions: $(jq -r '.open_positions' "$STATUS_FILE")"
        
        ISSUES=$(jq -r '.issues[]?' "$STATUS_FILE" 2>/dev/null)
        if [ ! -z "$ISSUES" ]; then
            echo "Issues:"
            echo "$ISSUES" | sed 's/^/  - /'
        fi
    else
        echo "Health file exists but jq not available for parsing"
        echo "Raw health status:"
        cat "$STATUS_FILE"
    fi
else
    echo "✗ No health status file found"
fi

# Check recent logs
echo
echo "=== Recent Activity ==="
TODAY=$(date +%Y-%m-%d)
if [ -f "$LOG_DIR/trading_$TODAY.log" ]; then
    echo "Recent Trading Activity:"
    tail -5 "$LOG_DIR/trading_$TODAY.log" | sed 's/^/  /'
else
    echo "No trading logs found for today"
fi

echo
echo "=== Recent Errors ==="
if [ -f "$LOG_DIR/errors_$TODAY.log" ]; then
    ERROR_COUNT=$(wc -l < "$LOG_DIR/errors_$TODAY.log")
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "Errors today: $ERROR_COUNT"
        echo "Recent errors:"
        tail -3 "$LOG_DIR/errors_$TODAY.log" | sed 's/^/  /'
    else
        echo "✓ No errors today"
    fi
else
    echo "No error logs found for today"
fi

# Check system uptime
echo
echo "=== System Uptime ==="
if [ -f "$LOG_DIR/trading_$TODAY.log" ]; then
    FIRST_LOG=$(head -1 "$LOG_DIR/trading_$TODAY.log" | cut -d' ' -f1-2)
    if [ ! -z "$FIRST_LOG" ]; then
        echo "System started: $FIRST_LOG"
    fi
fi

# Performance summary
echo
echo "=== Performance Summary ==="
if [ -f "./app/reports/summary.json" ]; then
    if command -v jq > /dev/null; then
        TOTAL_RETURN=$(jq -r '.total_return' "./app/reports/summary.json" 2>/dev/null)
        WIN_RATE=$(jq -r '.win_rate' "./app/reports/summary.json" 2>/dev/null)
        NUM_TRADES=$(jq -r '.num_trades' "./app/reports/summary.json" 2>/dev/null)
        
        if [ "$TOTAL_RETURN" != "null" ]; then
            echo "Total Return: $TOTAL_RETURN"
        fi
        if [ "$WIN_RATE" != "null" ]; then
            echo "Win Rate: $WIN_RATE"
        fi
        if [ "$NUM_TRADES" != "null" ]; then
            echo "Total Trades: $NUM_TRADES"
        fi
    else
        echo "Performance data available but jq not installed"
    fi
else
    echo "No performance data available"
fi

echo
echo "=== Recommendations ==="
if [ -f "$STATUS_FILE" ] && command -v jq > /dev/null; then
    HEALTHY=$(jq -r '.is_healthy' "$STATUS_FILE")
    ERRORS=$(jq -r '.consecutive_errors' "$STATUS_FILE")
    
    if [ "$HEALTHY" = "false" ]; then
        echo "⚠️  System health issues detected - check logs"
    fi
    
    if [ "$ERRORS" -gt 3 ]; then
        echo "⚠️  High error count - consider restarting system"
    fi
    
    if [ "$ERRORS" -eq 0 ] && [ "$HEALTHY" = "true" ]; then
        echo "✓ System running smoothly"
    fi
fi

echo
echo "=== Quick Commands ==="
echo "View live logs: tail -f $LOG_DIR/trading_$TODAY.log"
echo "Check status: uv run python app/main.py --status"
echo "Restart system: uv run python app/main.py --restart"
echo "Stop system: pkill -f 'python app/main.py'"
echo "Start system: uv run python app/main.py --mode paper"

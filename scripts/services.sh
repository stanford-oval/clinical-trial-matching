#!/usr/bin/env bash
# ============================================================================
# SatIR Services Manager
# ============================================================================
# Manages Elasticsearch and Snowstorm SNOMED-CT server lifecycle.
#
# Usage:
#   ./scripts/services.sh start       Start both services in background
#   ./scripts/services.sh stop        Stop both services
#   ./scripts/services.sh status      Check if services are running
#   ./scripts/services.sh install     Download and install both services
#   ./scripts/services.sh import      Import SNOMED-CT data into Snowstorm
#   ./scripts/services.sh check       Health check (verify data is loaded)
#
# Configuration (via satir.toml, env vars, or defaults):
#   ELASTIC_HOME     ~/elastic/elasticsearch-7.17.15
#   SNOWSTORM_JAR    Auto-detected in parent dirs or set explicitly
#   SNOWSTORM_HEAP   4g (default)
#   SNOMED_ZIP       Path to SnomedCT RF2 release zip for import
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Configuration ────────────────────────────────────────────────────────

ELASTIC_HOME="${ELASTIC_HOME:-$HOME/elastic/elasticsearch-7.17.15}"
ELASTIC_URL="${ELASTICSEARCH_URL:-http://localhost:9200}"
ELASTIC_PID="/tmp/satir-elasticsearch.pid"

SNOWSTORM_URL="${SNOWSTORM_BASE:-http://localhost:8080}"
SNOWSTORM_HEAP="${SNOWSTORM_HEAP:-4g}"
SNOWSTORM_PID="/tmp/satir-snowstorm.pid"
SNOWSTORM_LOG="/tmp/satir-snowstorm.log"

# Auto-detect snowstorm jar
_find_snowstorm_jar() {
    local candidates=(
        "${SNOWSTORM_JAR:-}"
        "$PROJECT_ROOT/snowstorm-10.7.0.jar"
        "$PROJECT_ROOT/../snowstorm-10.7.0.jar"
        "$PROJECT_ROOT/../../snowstorm-10.7.0.jar"
        "$HOME/snowstorm-10.7.0.jar"
    )
    for jar in "${candidates[@]}"; do
        if [[ -n "$jar" && -f "$jar" ]]; then
            echo "$jar"
            return 0
        fi
    done
    echo ""
}

SNOWSTORM_JAR_PATH="$(_find_snowstorm_jar)"

# ── Colors ───────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }

# ── Helpers ──────────────────────────────────────────────────────────────

_is_running() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

_wait_for_url() {
    local url="$1"
    local name="$2"
    local max_wait="${3:-60}"
    local i=0
    while ! curl -s --connect-timeout 2 "$url" >/dev/null 2>&1; do
        i=$((i + 1))
        if [[ $i -ge $max_wait ]]; then
            fail "$name did not start within ${max_wait}s"
            return 1
        fi
        sleep 1
    done
    ok "$name is ready ($url)"
}

_ensure_java17() {
    if command -v java >/dev/null 2>&1; then
        local ver
        ver=$(java -version 2>&1 | head -1 | sed 's/.*"\([0-9]*\).*/\1/')
        if [[ "$ver" -ge 17 ]]; then
            return 0
        fi
    fi
    # Try to find Java 17
    if [[ -x /usr/libexec/java_home ]]; then
        local jh
        jh=$(/usr/libexec/java_home -v17 2>/dev/null || true)
        if [[ -n "$jh" && -d "$jh" ]]; then
            export JAVA_HOME="$jh"
            export PATH="$JAVA_HOME/bin:$PATH"
            return 0
        fi
    fi
    fail "Java 17+ required. Install: brew install openjdk@17"
    return 1
}

# ── Commands ─────────────────────────────────────────────────────────────

cmd_install() {
    echo "Installing SatIR services..."
    echo ""

    # Elasticsearch
    echo "[Elasticsearch]"
    if [[ -d "$ELASTIC_HOME" ]]; then
        ok "Already installed at $ELASTIC_HOME"
    else
        echo "  Downloading Elasticsearch 7.17.15..."
        mkdir -p "$(dirname "$ELASTIC_HOME")"
        local arch
        arch=$(uname -m)
        local suffix="linux-x86_64"
        if [[ "$(uname)" == "Darwin" ]]; then
            if [[ "$arch" == "arm64" ]]; then
                suffix="darwin-aarch64"
            else
                suffix="darwin-x86_64"
            fi
        fi
        local url="https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.15-${suffix}.tar.gz"
        local tarball="/tmp/elasticsearch-7.17.15.tar.gz"
        curl -L -o "$tarball" "$url"
        tar -xzf "$tarball" -C "$(dirname "$ELASTIC_HOME")"
        rm -f "$tarball"
        ok "Installed to $ELASTIC_HOME"
    fi
    echo ""

    # Snowstorm
    echo "[Snowstorm]"
    if [[ -n "$SNOWSTORM_JAR_PATH" && -f "$SNOWSTORM_JAR_PATH" ]]; then
        ok "Found at $SNOWSTORM_JAR_PATH"
    else
        warn "snowstorm-10.7.0.jar not found"
        echo "  Download from: https://drive.google.com/drive/folders/16v16aopzmJuQdZtkjCi2uJ_pNUuPNGbX"
        echo "  Place in: $PROJECT_ROOT/ or set SNOWSTORM_JAR=/path/to/snowstorm-10.7.0.jar"
    fi
    echo ""

    # Java
    echo "[Java 17]"
    if _ensure_java17; then
        ok "Java 17+ available: $(java -version 2>&1 | head -1)"
    fi
}

cmd_start() {
    echo "Starting SatIR services..."
    echo ""

    # Elasticsearch
    echo "[Elasticsearch]"
    if _is_running "$ELASTIC_PID"; then
        ok "Already running (pid $(cat "$ELASTIC_PID"))"
    elif [[ ! -d "$ELASTIC_HOME" ]]; then
        fail "Not installed. Run: ./scripts/services.sh install"
    else
        "$ELASTIC_HOME/bin/elasticsearch" -d -p "$ELASTIC_PID" 2>/dev/null
        echo "  Starting... (pid file: $ELASTIC_PID)"
        _wait_for_url "$ELASTIC_URL" "Elasticsearch" 30
    fi
    echo ""

    # Snowstorm
    echo "[Snowstorm]"
    if _is_running "$SNOWSTORM_PID"; then
        ok "Already running (pid $(cat "$SNOWSTORM_PID"))"
    elif [[ -z "$SNOWSTORM_JAR_PATH" || ! -f "$SNOWSTORM_JAR_PATH" ]]; then
        fail "snowstorm-10.7.0.jar not found. Run: ./scripts/services.sh install"
    else
        _ensure_java17 || return 1
        nohup java -Xms"$SNOWSTORM_HEAP" -Xmx"$SNOWSTORM_HEAP" \
            -jar "$SNOWSTORM_JAR_PATH" \
            --snowstorm.rest-api.readonly=true \
            > "$SNOWSTORM_LOG" 2>&1 &
        echo $! > "$SNOWSTORM_PID"
        echo "  Starting... (pid $(cat "$SNOWSTORM_PID"), log: $SNOWSTORM_LOG)"
        _wait_for_url "$SNOWSTORM_URL" "Snowstorm" 300
    fi
}

cmd_stop() {
    echo "Stopping SatIR services..."
    echo ""

    for name_pid in "Elasticsearch:$ELASTIC_PID" "Snowstorm:$SNOWSTORM_PID"; do
        local name="${name_pid%%:*}"
        local pid_file="${name_pid##*:}"
        echo "[$name]"
        if _is_running "$pid_file"; then
            local pid
            pid=$(cat "$pid_file")
            kill "$pid" 2>/dev/null || true
            rm -f "$pid_file"
            ok "Stopped (pid $pid)"
        else
            ok "Not running"
        fi
    done
}

cmd_status() {
    echo "SatIR Services Status"
    echo ""

    echo "[Elasticsearch]"
    if _is_running "$ELASTIC_PID"; then
        ok "Running (pid $(cat "$ELASTIC_PID"))"
    elif curl -s --connect-timeout 2 "$ELASTIC_URL" >/dev/null 2>&1; then
        ok "Running (external, $ELASTIC_URL)"
    else
        fail "Not running"
    fi

    echo "[Snowstorm]"
    if _is_running "$SNOWSTORM_PID"; then
        ok "Running (pid $(cat "$SNOWSTORM_PID"))"
    elif curl -s --connect-timeout 2 "$SNOWSTORM_URL" >/dev/null 2>&1; then
        ok "Running (external, $SNOWSTORM_URL)"
    else
        fail "Not running"
    fi
}

cmd_check() {
    echo "SatIR Services Health Check"
    echo ""

    echo "[Elasticsearch]"
    if curl -s --connect-timeout 2 "$ELASTIC_URL" >/dev/null 2>&1; then
        local health
        health=$(curl -s "$ELASTIC_URL/_cluster/health" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")
        ok "Cluster health: $health"
        local indices
        indices=$(curl -s "$ELASTIC_URL/_cat/indices?h=index" 2>/dev/null | wc -l | tr -d ' ')
        ok "$indices indices loaded"
    else
        fail "Not reachable at $ELASTIC_URL"
    fi
    echo ""

    echo "[Snowstorm]"
    if curl -s --connect-timeout 2 "$SNOWSTORM_URL" >/dev/null 2>&1; then
        ok "Reachable at $SNOWSTORM_URL"
        # Test SNOMED query
        local result
        result=$(curl -s "${SNOWSTORM_URL}/browser/MAIN/descriptions?term=osteomalacia&active=true&limit=1" 2>/dev/null)
        if echo "$result" | python3 -c "import sys,json; items=json.load(sys.stdin).get('items',[]); sys.exit(0 if items else 1)" 2>/dev/null; then
            ok "SNOMED-CT data loaded (test query: osteomalacia)"
        else
            warn "SNOMED-CT data may not be loaded. Run: ./scripts/services.sh import"
        fi
    else
        fail "Not reachable at $SNOWSTORM_URL"
    fi
}

cmd_import() {
    local snomed_zip="${SNOMED_ZIP:-}"

    if [[ -z "$snomed_zip" ]]; then
        # Auto-detect
        for candidate in \
            "$PROJECT_ROOT/../SnomedCT_InternationalRF2_PRODUCTION_"*.zip \
            "$PROJECT_ROOT/SnomedCT_InternationalRF2_PRODUCTION_"*.zip \
            "$HOME/SnomedCT_InternationalRF2_PRODUCTION_"*.zip; do
            if [[ -f "$candidate" ]]; then
                snomed_zip="$candidate"
                break
            fi
        done
    fi

    if [[ -z "$snomed_zip" || ! -f "$snomed_zip" ]]; then
        fail "SNOMED-CT RF2 release zip not found."
        echo "  Set SNOMED_ZIP=/path/to/SnomedCT_InternationalRF2_PRODUCTION_*.zip"
        echo "  Download from: https://drive.google.com/drive/folders/16v16aopzmJuQdZtkjCi2uJ_pNUuPNGbX"
        return 1
    fi

    if [[ -z "$SNOWSTORM_JAR_PATH" || ! -f "$SNOWSTORM_JAR_PATH" ]]; then
        fail "snowstorm-10.7.0.jar not found"
        return 1
    fi

    _ensure_java17 || return 1

    echo "Importing SNOMED-CT data..."
    echo "  Jar:  $SNOWSTORM_JAR_PATH"
    echo "  Data: $snomed_zip"
    echo "  This may take 30-60 minutes."
    echo ""

    java -jar "$SNOWSTORM_JAR_PATH" import \
        --path "$snomed_zip" \
        --branch MAIN \
        --code-system-short-name SNOMEDCT \
        --release-type FULL \
        --create-code-system-version true \
        --exit

    ok "SNOMED-CT import complete"
}

# ── Main ─────────────────────────────────────────────────────────────────

case "${1:-help}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    install) cmd_install ;;
    import)  cmd_import ;;
    check)   cmd_check ;;
    help|*)
        echo "Usage: $0 {start|stop|status|install|import|check}"
        echo ""
        echo "Commands:"
        echo "  start    Start Elasticsearch + Snowstorm in background"
        echo "  stop     Stop both services"
        echo "  status   Check if services are running"
        echo "  install  Download and install Elasticsearch + Snowstorm"
        echo "  import   Import SNOMED-CT data into Snowstorm"
        echo "  check    Health check (verify services + data)"
        echo ""
        echo "Environment:"
        echo "  ELASTIC_HOME     Elasticsearch install dir (default: ~/elastic/elasticsearch-7.17.15)"
        echo "  SNOWSTORM_JAR    Path to snowstorm-10.7.0.jar (auto-detected)"
        echo "  SNOWSTORM_HEAP   JVM heap size (default: 4g)"
        echo "  SNOMED_ZIP       Path to SNOMED-CT RF2 zip (for import)"
        ;;
esac

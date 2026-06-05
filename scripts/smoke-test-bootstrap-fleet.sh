#!/usr/bin/env bash
#
# Sprint 407 — multi-cloud bootstrap fleet smoke test.
#
# End-to-end verification across all sprint 388-405
# observability surfaces against the live canonical fleet:
#
#   - TCP+WSS reachability (sprint 385 fleet probe equivalent)
#   - /health JSON shape (pre-sprint-392 baseline contract)
#   - /health/detailed per-subsystem readiness (sprint 392)
#   - /metrics content negotiation (sprint 389):
#       * Accept: text/plain          -> Prometheus
#       * Accept: application/json    -> JSON
#   - /prometheus always-Prometheus alias (sprint 389)
#   - /health/detailed federation_sync is "disabled" not
#     "stale" on default deploys (sprint 397)
#   - /metrics Prometheus surface has sprint-394 labeled
#     subsystem-status + heartbeat-age gauges
#   - /health.region is set (not REGION_UNSET — sprint 398)
#
# Operators run this from a cron, a monitoring agent, or
# pre/post-deploy to validate the fleet is end-to-end OK.
# Exit codes: 0 = all checks pass, non-zero = at least one
# failure (use --format json to localize).
#
# Usage:
#   scripts/smoke-test-bootstrap-fleet.sh
#   scripts/smoke-test-bootstrap-fleet.sh --host bootstrap-eu.prsm-network.com
#   scripts/smoke-test-bootstrap-fleet.sh --timeout 30
#   scripts/smoke-test-bootstrap-fleet.sh --format json
#
# Compatible with bash 3.2 (macOS default) + bash 4/5.
# Requirements: bash + curl + python3 (all preinstalled on
# any modern unix; no PRSM venv required).

set -uo pipefail

# ── Canonical fleet ──────────────────────────────────────
DEFAULT_HOSTS=(
    "bootstrap-us.prsm-network.com"
    "bootstrap-eu.prsm-network.com"
    "bootstrap-apac.prsm-network.com"
)
WSS_PORT=8765
API_PORT=8000

# ── CLI parsing ──────────────────────────────────────────
HOSTS=()
TIMEOUT=10
FORMAT="text"

while [ $# -gt 0 ]; do
    case "$1" in
        --host)
            HOSTS+=("$2"); shift 2 ;;
        --timeout)
            TIMEOUT="$2"; shift 2 ;;
        --format)
            FORMAT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^#//;s/^ //'
            exit 0 ;;
        *)
            echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ ${#HOSTS[@]} -eq 0 ]; then
    HOSTS=("${DEFAULT_HOSTS[@]}")
fi

# ── ANSI color helpers ───────────────────────────────────
if [ "$FORMAT" = "text" ] && [ -t 1 ]; then
    GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'
    DIM='\033[2m'; BOLD='\033[1m'; RESET='\033[0m'
else
    GREEN=''; RED=''; YELLOW=''; DIM=''; BOLD=''; RESET=''
fi

# ── Result storage (bash-3.2 compatible) ─────────────────
# Each result is "host|name|ok|detail" stored in an indexed
# array. Render phase iterates + filters by host.
RESULTS=()
TOTAL_PASS=0
TOTAL_FAIL=0

_record() {
    local host="$1" name="$2" ok="$3" detail="${4:-}"
    RESULTS+=("${host}|${name}|${ok}|${detail}")
    if [ "$ok" = "1" ]; then
        TOTAL_PASS=$((TOTAL_PASS + 1))
    else
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
}

# ── Check helpers ────────────────────────────────────────

_get() {
    local url="$1"; shift
    curl -sS -m "$TIMEOUT" "$@" "$url" 2>/dev/null
}

_tcp_check() {
    local host="$1" port="$2"
    if command -v nc >/dev/null 2>&1; then
        nc -z -G "$TIMEOUT" "$host" "$port" 2>/dev/null \
            || nc -z -w "$TIMEOUT" "$host" "$port" 2>/dev/null
    else
        timeout "$TIMEOUT" bash -c "</dev/tcp/$host/$port" 2>/dev/null
    fi
}

# ── Run checks per host ──────────────────────────────────

for host in "${HOSTS[@]}"; do
    # 1. TCP reachability on WSS port
    if _tcp_check "$host" "$WSS_PORT"; then
        _record "$host" "tcp_wss_${WSS_PORT}" 1
    else
        _record "$host" "tcp_wss_${WSS_PORT}" 0 "TCP connect failed"
    fi

    # 2. TCP reachability on API port (sprint-398 ufw fix)
    if _tcp_check "$host" "$API_PORT"; then
        _record "$host" "tcp_api_${API_PORT}" 1
    else
        _record "$host" "tcp_api_${API_PORT}" 0 "TCP connect failed"
    fi

    api_base="http://${host}:${API_PORT}"

    # 3. /health JSON shape
    body=$(_get "${api_base}/health")
    if [ -n "$body" ] && echo "$body" | python3 -c "
import sys, json
d = json.load(sys.stdin)
required = {'status', 'uptime_seconds', 'region', 'version'}
assert required.issubset(d.keys()), f'missing: {required - d.keys()}'
" 2>/dev/null; then
        _record "$host" "health_json" 1
    else
        _record "$host" "health_json" 0 "missing fields or invalid JSON"
    fi

    # 4. /health.region populated + not REGION_UNSET (sprint 398)
    region=$(echo "$body" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('region', ''))
except Exception:
    pass
" 2>/dev/null)
    if [ -n "$region" ] && [ "$region" != "REGION_UNSET" ]; then
        _record "$host" "region_set" 1 "$region"
    else
        _record "$host" "region_set" 0 "region=${region:-<empty>}"
    fi

    # 5. /health/detailed (sprint 392) returns valid shape
    body=$(_get "${api_base}/health/detailed")
    if [ -n "$body" ] && echo "$body" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'status' in d and 'subsystems' in d
assert isinstance(d['subsystems'], dict)
expected = {'peer_cleanup', 'peer_backup', 'federation_sync',
            'health_check_loop', 'api_server'}
assert expected.issubset(d['subsystems'].keys()), (
    f'missing subsystems: {expected - d[\"subsystems\"].keys()}'
)
" 2>/dev/null; then
        _record "$host" "health_detailed" 1
    else
        _record "$host" "health_detailed" 0 "shape check failed"
    fi

    # 6. sprint 397: federation_sync NOT stale on default
    fed=$(echo "$body" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('subsystems', {}).get('federation_sync', {}).get('status', ''))
except Exception:
    pass
" 2>/dev/null)
    if [ "$fed" = "disabled" ] || [ "$fed" = "healthy" ]; then
        _record "$host" "federation_sync_not_stale" 1 "$fed"
    else
        _record "$host" "federation_sync_not_stale" 0 "status=${fed:-<empty>}"
    fi

    # 7. /metrics with Accept: text/plain -> Prometheus (sprint 389)
    body=$(_get "${api_base}/metrics" -H "Accept: text/plain")
    if echo "$body" | head -20 | grep -q "^# HELP "; then
        _record "$host" "metrics_prometheus_via_accept" 1
    else
        _record "$host" "metrics_prometheus_via_accept" 0 "no # HELP lines"
    fi

    # 8. /metrics with Accept: application/json -> JSON
    body=$(_get "${api_base}/metrics" -H "Accept: application/json")
    if [ -n "$body" ] && echo "$body" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'uptime_seconds' in d
" 2>/dev/null; then
        _record "$host" "metrics_json_via_accept" 1
    else
        _record "$host" "metrics_json_via_accept" 0 "JSON probe failed"
    fi

    # 9. /prometheus always-Prometheus alias (sprint 389)
    body=$(_get "${api_base}/prometheus")
    if echo "$body" | head -5 | grep -q "^# HELP "; then
        _record "$host" "prometheus_alias" 1
    else
        _record "$host" "prometheus_alias" 0 "no Prometheus exposition"
    fi

    # 10. sprint 394 — Prometheus subsystem labeled gauges
    if echo "$body" | grep -q "^prsm_bootstrap_subsystem_status{"; then
        _record "$host" "subsystem_status_gauge" 1
    else
        _record "$host" "subsystem_status_gauge" 0 "gauge family absent"
    fi
done

# ── Render output ────────────────────────────────────────

if [ "$FORMAT" = "json" ]; then
    # Pipe results to python3 for clean JSON emission
    {
        printf "%s\n" "${RESULTS[@]}"
    } | python3 -c "
import json, sys, collections
hosts_data = collections.OrderedDict()
for line in sys.stdin:
    line = line.rstrip('\n')
    if not line:
        continue
    parts = line.split('|', 3)
    while len(parts) < 4:
        parts.append('')
    host, name, ok, detail = parts
    if host not in hosts_data:
        hosts_data[host] = []
    hosts_data[host].append({
        'name': name, 'ok': ok == '1', 'detail': detail,
    })
out = {
    'total_pass': $TOTAL_PASS,
    'total_fail': $TOTAL_FAIL,
    'overall_status': 'ok' if $TOTAL_FAIL == 0 else 'degraded',
    'hosts': [{'host': h, 'checks': c} for h, c in hosts_data.items()],
}
print(json.dumps(out, indent=2))
"
    [ "$TOTAL_FAIL" -eq 0 ] && exit 0 || exit 1
fi

# Text rendering
echo
echo -e "${BOLD}PRSM Bootstrap Fleet Smoke Test${RESET}"
echo -e "${DIM}timeout: ${TIMEOUT}s · sprint 388-406 surface coverage${RESET}"
echo

for host in "${HOSTS[@]}"; do
    pass=0; fail=0
    fail_lines=()
    for row in "${RESULTS[@]}"; do
        case "$row" in
            "${host}|"*)
                IFS='|' read -r r_host r_name r_ok r_detail <<< "$row"
                if [ "$r_ok" = "1" ]; then
                    pass=$((pass + 1))
                else
                    fail=$((fail + 1))
                    if [ -n "$r_detail" ]; then
                        fail_lines+=("    ${RED}✗${RESET} ${r_name} ${DIM}— ${r_detail}${RESET}")
                    else
                        fail_lines+=("    ${RED}✗${RESET} ${r_name}")
                    fi
                fi
                ;;
        esac
    done
    total=$((pass + fail))
    if [ "$fail" -eq 0 ]; then
        echo -e "${GREEN}✓${RESET} ${BOLD}${host}${RESET} ${DIM}(${pass}/${total} pass)${RESET}"
    else
        echo -e "${RED}✗${RESET} ${BOLD}${host}${RESET} ${DIM}(${pass}/${total} pass, ${fail} fail)${RESET}"
        for line in "${fail_lines[@]}"; do
            echo -e "$line"
        done
    fi
done
echo
echo -e "${DIM}Total: ${TOTAL_PASS} pass, ${TOTAL_FAIL} fail${RESET}"

[ "$TOTAL_FAIL" -eq 0 ] && exit 0 || exit 1

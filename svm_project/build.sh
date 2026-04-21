#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# build.sh — Compile and run the SVM project (no build tools required)
# Usage:
#   ./build.sh          → compile + run main demo
#   ./build.sh test     → compile + run unit tests
#   ./build.sh all      → compile + run demo + run tests
# ─────────────────────────────────────────────────────────────────────────────
set -e

SRC_MAIN="src/main/java"
SRC_TEST="src/test/java"
OUT="out"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Compiling SVM project..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
mkdir -p "$OUT"

# Compile main sources
find "$SRC_MAIN" -name "*.java" | xargs javac -d "$OUT" -encoding UTF-8
echo "  Main sources compiled ✓"

# Compile test sources
find "$SRC_TEST" -name "*.java" | xargs javac -cp "$OUT" -d "$OUT" -encoding UTF-8
echo "  Test sources compiled ✓"
echo ""

MODE="${1:-main}"

if [[ "$MODE" == "main" || "$MODE" == "all" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running Main Demo"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    java -cp "$OUT" svm.Main
    echo ""
fi

if [[ "$MODE" == "test" || "$MODE" == "all" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running Unit Tests"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    java -cp "$OUT" svm.SVMTest
fi

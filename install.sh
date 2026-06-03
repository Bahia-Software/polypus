#!/usr/bin/env bash
# install.sh — Build and install polypus (Rust extension + Python wrapper),
#              then optionally run the test suite and benchmark to verify the installation.
#
# Usage:
#   ./install.sh              # interactive prompts (when run from a terminal)
#   ./install.sh --yes        # accept all defaults, no prompts  (CI / scripts)
#   ./install.sh --no-tests   # accept all defaults, skip tests  (CI / scripts)
#   ./install.sh --benchmark  # accept all defaults + run quick benchmark

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${BOLD}[polypus]${NC} $*"; }
success() { echo -e "${GREEN}[polypus]${NC} $*"; }
warn()    { echo -e "${YELLOW}[polypus]${NC} $*"; }
error()   { echo -e "${RED}[polypus]${NC} $*" >&2; exit 1; }

# ── Interactive helpers ───────────────────────────────────────────────────────
# ask_yn VAR "Question?" "y|n"   — sets VAR to y or n
ask_yn() {
    local var="$1" question="$2" default="$3"
    local hint; [[ "$default" == "y" ]] && hint="${BOLD}Y${NC}/n" || hint="y/${BOLD}N${NC}"
    while true; do
        read -r -p "$(echo -e "  ${CYAN}?${NC} ${question} [${hint}] ")" ans
        ans="${ans:-$default}"
        case "${ans,,}" in
            y|yes) eval "$var='y'"; return ;;
            n|no)  eval "$var='n'"; return ;;
            *) echo "    Please answer y or n." ;;
        esac
    done
}

# ask_build VAR default    — sets VAR to "release" or "dev"
ask_build() {
    local var="$1" default="$2"
    while true; do
        read -r -p "$(echo -e "  ${CYAN}?${NC} Build mode (release/dev) [${BOLD}${default}${NC}] ")" ans
        ans="${ans:-$default}"
        case "${ans,,}" in
            release|dev) eval "$var='$ans'"; return ;;
            *) echo "    Please answer: release or dev." ;;
        esac
    done
}

# ask_tests VAR default    — sets VAR to "all", "smoke", or "none"
ask_tests() {
    local var="$1" default="$2"
    while true; do
        read -r -p "$(echo -e "  ${CYAN}?${NC} Run tests after install? (all/smoke/none) [${BOLD}${default}${NC}] ")" ans
        ans="${ans:-$default}"
        case "${ans,,}" in
            all|smoke|none) eval "$var='$ans'"; return ;;
            *) echo "    Please answer: all, smoke, or none." ;;
        esac
    done
}

# ── Version (read from Cargo.toml) ──────────────────────────────────────────
VERSION=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)"/\1/')

# ── Banner ────────────────────────────────────────────────────────────────────
print_banner() {
    echo ""
    printf "${CYAN}"
    echo '  ██████╗   ██████╗  ██╗      ██╗   ██╗ ██████╗  ██╗   ██╗ ███████╗'
    echo '  ██╔══██╗ ██╔═══██╗ ██║     ╚██╗ ██╔╝ ██╔══██╗ ██║   ██║ ██╔════╝'
    echo '  ██████╔╝ ██║   ██║ ██║      ╚████╔╝  ██████╔╝ ██║   ██║ ███████╗'
    echo '  ██╔═══╝  ██║   ██║ ██║       ╚██╔╝   ██╔═══╝  ██║   ██║ ╚════██║'
    echo '  ██║      ╚██████╔╝ ███████╗   ██║    ██║      ╚██████╔╝  ███████║'
    echo '  ╚═╝       ╚═════╝  ╚══════╝   ╚═╝    ╚═╝       ╚═════╝   ╚══════╝'
    printf "${NC}\n"
    echo -e "\t${BOLD}A Distributed Quantum Computing Library${NC}  ·  v${VERSION}"
    echo ""
}

# ── Args ──────────────────────────────────────────────────────────────────────
YES=0
EXPLICIT_NO_TESTS=0
EXPLICIT_BENCHMARK=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y)       YES=1                ;;
        --no-tests)     EXPLICIT_NO_TESTS=1  ;;
        --benchmark)    EXPLICIT_BENCHMARK=1 ;;
        *) error "Unknown argument: '$arg'. Use --yes, --no-tests or --benchmark." ;;
    esac
done
# --no-tests implies non-interactive (backward compat)
[[ $EXPLICIT_NO_TESTS == 1 ]] && YES=1
# --benchmark implies non-interactive
[[ $EXPLICIT_BENCHMARK == 1 ]] && YES=1

# ── Defaults ──────────────────────────────────────────────────────────────────
INSTALL_DEV="y"
INSTALL_EXAMPLES="n"
BUILD_MODE="release"
RUN_TESTS="all"
RUN_BENCHMARK="n"
[[ $EXPLICIT_NO_TESTS == 1 ]] && RUN_TESTS="none"
[[ $EXPLICIT_BENCHMARK == 1 ]] && RUN_BENCHMARK="y"

# ── Locate script root ────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
print_banner

# ── Interactive configuration ─────────────────────────────────────────────────
if [[ $YES == 0 ]] && [[ -t 0 ]]; then
    echo ""
    echo -e "${BOLD}polypus installer${NC} — press Enter to accept each default."
    echo ""
    ask_yn    INSTALL_DEV      "Install dev dependencies (maturin, pytest)?"                   "y"
    ask_yn    INSTALL_EXAMPLES "Install example dependencies (numpy, scipy, matplotlib, networkx)?" "n"
    ask_build BUILD_MODE       "release"
    ask_tests RUN_TESTS        "all"
    ask_yn    RUN_BENCHMARK    "Run system benchmark after install? (time + RAM sweep)"            "n"

    echo ""
    echo -e "  ${BOLD}Summary:${NC}"
    echo "    Dev deps:      $INSTALL_DEV"
    echo "    Example deps:  $INSTALL_EXAMPLES"
    echo "    Build mode:    $BUILD_MODE"
    echo "    Tests:         $RUN_TESTS"
    echo "    Benchmark:     $RUN_BENCHMARK"
    echo ""
    read -r -p "$(echo -e "  ${CYAN}?${NC} Proceed? [${BOLD}Y${NC}/n] ")" confirm
    confirm="${confirm:-y}"
    [[ "${confirm,,}" == "n" || "${confirm,,}" == "no" ]] && { echo "Aborted."; exit 0; }
    echo ""
else
    echo ""
    info "Build plan (non-interactive):"
    echo "    Dev deps:      $INSTALL_DEV"
    echo "    Example deps:  $INSTALL_EXAMPLES"
    echo "    Build mode:    $BUILD_MODE"
    echo "    Tests:         $RUN_TESTS"
    echo "    Benchmark:     $RUN_BENCHMARK"
    echo ""
fi


# ── 1. Python sanity check ───────────────────────────────────────────────────
info "Checking Python..."
python --version || error "python not found. Activate the right environment first."
PY_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python ${PY_VERSION} at $(which python)"

# ── 2. Install dev dependencies ──────────────────────────────────────────────
if [[ "$INSTALL_DEV" == "y" ]]; then
    info "Installing dev dependencies (requirements-dev.txt)..."
    pip install --quiet -r requirements-dev.txt \
        || error "Failed to install dev dependencies."
    success "Dev dependencies installed."
else
    info "Skipping dev dependencies."
fi

# ── 3. Ensure maturin is installed ───────────────────────────────────────────
info "Checking maturin..."
if ! python -m maturin --version &>/dev/null; then
    warn "maturin not found — installing..."
    pip install "maturin>=1.8,<2.0" || error "Failed to install maturin."
fi
success "maturin $(python -m maturin --version)"

# ── 4. Install example dependencies ──────────────────────────────────────────
if [[ "$INSTALL_EXAMPLES" == "y" ]]; then
    info "Installing example dependencies (requirements-examples.txt)..."
    pip install --quiet -r requirements-examples.txt \
        || error "Failed to install example dependencies."
    success "Example dependencies installed."
fi

# ── 5. Build polypus_python wheel ────────────────────────────────────────────
info "Building polypus_python wheel..."
python -m build packages/polypus_python/ --wheel --no-isolation \
    || error "Failed to build polypus_python wheel."

WHEEL=$(ls -t packages/polypus_python/dist/polypus_python-*.whl | head -1)
[[ -f "$WHEEL" ]] || error "Wheel not found after build."
success "Built: $(basename "$WHEEL")"

# ── 6. Install polypus_python ────────────────────────────────────────────────
info "Installing polypus_python..."
pip install --quiet --force-reinstall "$WHEEL" \
    || error "Failed to install polypus_python."
success "polypus_python installed."

# ── 7. Build and install polypus Rust extension ──────────────────────────────
MATURIN_FLAGS="--features extension-module"
[[ "$BUILD_MODE" == "release" ]] && MATURIN_FLAGS="--release $MATURIN_FLAGS"
info "Building polypus Rust extension (maturin develop ${BUILD_MODE})..."
python -m maturin develop $MATURIN_FLAGS \
    || error "maturin develop failed."
success "polypus Rust extension installed."

# ── 8. Verify the Python environment sees both packages ──────────────────────
info "Verifying installed packages..."
python -c "
import polypus, polypus_python
required = ['run_quantum_circuit', 'train', 'DE', 'PSO', 'QNG']
missing = [s for s in required if not hasattr(polypus, s)]
if missing:
    raise ImportError(f'Missing symbols in polypus: {missing}')
print('  polypus:', ', '.join(required))
print('  polypus_python: OK')
" || error "Installation verification failed."
success "All symbols present."

# ── 9. Tests ─────────────────────────────────────────────────────────────────
if [[ "$RUN_TESTS" == "none" ]]; then
    echo ""
    success "Installation complete."
    exit 0
fi

# Ensure pytest is available
if ! python -m pytest --version &>/dev/null; then
    warn "pytest not found — installing..."
    pip install pytest --quiet || error "Failed to install pytest."
fi

echo ""
if [[ "$RUN_TESTS" == "smoke" ]]; then
    info "Running smoke tests (no hardware required)..."
    python -m pytest tests/python/ -m "not integration" -v
elif [[ "$RUN_TESTS" == "all" ]]; then
    info "Running full Python test suite..."
    python -m pytest tests/python/ -v
    echo ""
    info "Running Rust tests..."
    PYTHON_LIBDIR=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
    export LD_LIBRARY_PATH="${PYTHON_LIBDIR}:${LD_LIBRARY_PATH:-}"
    cargo test
fi

# ── 10. Benchmark ────────────────────────────────────────────────────────────
if [[ "$RUN_BENCHMARK" == "y" ]]; then
    echo ""
    info "Running system benchmark (--quick mode)..."
    python benchmarks/run_benchmarks.py --quick \
        || warn "Benchmark finished with errors — results may be incomplete."
fi

echo ""
info "Re-run options:"
echo "   ./install.sh --yes        # non-interactive with defaults"
echo "   ./install.sh --no-tests   # non-interactive, skip tests"
echo "   ./install.sh --benchmark  # non-interactive + quick benchmark"

echo ""
success "All done."

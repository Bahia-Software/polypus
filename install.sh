#!/usr/bin/env bash
# install.sh — Build and install polypus (Rust extension + Python wrapper),
#              then optionally run the test suite to verify the installation.
#
# Usage:
#   ./install.sh              # install + run full test suite (default)
#   ./install.sh --no-tests   # install only, skip tests

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${BOLD}[polypus]${NC} $*"; }
success() { echo -e "${GREEN}[polypus]${NC} $*"; }
warn()    { echo -e "${YELLOW}[polypus]${NC} $*"; }
error()   { echo -e "${RED}[polypus]${NC} $*" >&2; exit 1; }

# ── Args ──────────────────────────────────────────────────────────────────────
RUN_TESTS="all"
for arg in "$@"; do
    case "$arg" in
        --no-tests) RUN_TESTS=""     ;;
        *) error "Unknown argument: $arg. Use --no-tests to skip tests." ;;
    esac
done

# ── Locate script root ────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── 1. Python sanity check ────────────────────────────────────────────────────
info "Checking Python..."
python --version || error "python not found. Activate the right environment first."
PY_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python ${PY_VERSION} at $(which python)"

# ── 2. Ensure maturin is installed ───────────────────────────────────────────
info "Checking maturin..."
if ! python -m maturin --version &>/dev/null; then
    warn "maturin not found — installing..."
    pip install "maturin>=1.8,<2.0" || error "Failed to install maturin."
fi
success "maturin $(python -m maturin --version)"

# ── 3. Build polypus_python wheel ────────────────────────────────────────────
info "Building polypus_python wheel..."
python -m build packages/polypus_python/ --wheel --no-isolation \
    || error "Failed to build polypus_python wheel."

WHEEL=$(ls -t packages/polypus_python/dist/polypus_python-*.whl | head -1)
[[ -f "$WHEEL" ]] || error "Wheel not found after build."
success "Built: $(basename "$WHEEL")"

# ── 4. Install polypus_python ────────────────────────────────────────────────
info "Installing polypus_python..."
pip install --quiet --force-reinstall "$WHEEL" \
    || error "Failed to install polypus_python."
success "polypus_python installed."

# ── 5. Build and install polypus Rust extension ──────────────────────────────
info "Building polypus Rust extension (maturin develop --release)..."
python -m maturin develop --release --features extension-module \
    || error "maturin develop failed."
success "polypus Rust extension installed."

# ── 6. Verify the Python environment sees both packages ──────────────────────
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

# ── 7. Tests ─────────────────────────────────────────────────────────────────
if [[ -z "$RUN_TESTS" ]]; then
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

echo ""
info "Re-run options:"
echo "   ./install.sh --no-tests   # skip tests"

echo ""
success "All done."

#!/bin/bash

# scripts/install_submodules.sh
# Stage 1 of three-stage installation: Install submodules and their dependencies
# This ensures torch is available before pip install -e . runs

set -e  # Exit on any error

# Color output functions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Environment variables for selective installation
INSTALL_VERL=${INSTALL_VERL:-0}
INSTALL_WEBSHOP=${INSTALL_WEBSHOP:-0}
INSTALL_TUNIX=${INSTALL_TUNIX:-0}
INSTALL_ALL_SUBMODULES=${INSTALL_ALL_SUBMODULES:-0}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --verl)
                INSTALL_VERL=1
                shift
                ;;
            --webshop)
                INSTALL_WEBSHOP=1
                shift
                ;;
            --tunix)
                INSTALL_TUNIX=1
                shift
                ;;
            --all)
                INSTALL_ALL_SUBMODULES=1
                shift
                ;;
            -h|--help)
                echo "Submodule Installation Script for grl"
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --verl      Install verl submodule and dependencies"
                echo "  --webshop   Install webshop submodule and dependencies"
                echo "  --tunix     Install tunix submodule and dependencies"
                echo "  --all       Install all submodules and dependencies"
                echo "  -h, --help  Show this help message"
                echo ""
                echo "Environment variables:"
                echo "  INSTALL_VERL=1              Install verl submodule"
                echo "  INSTALL_WEBSHOP=1           Install webshop submodule"
                echo "  INSTALL_TUNIX=1             Install tunix submodule"
                echo "  INSTALL_ALL_SUBMODULES=1    Install all submodules"
                echo ""
                echo "Examples:"
                echo "  $0 --verl --tunix          # Install verl and tunix only"
                echo "  $0 --all                   # Install all submodules"
                echo "  INSTALL_VERL=1 $0          # Install verl using env var"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Set flags for --all
    if [[ "$INSTALL_ALL_SUBMODULES" == 1 ]]; then
        INSTALL_VERL=1
        INSTALL_WEBSHOP=1
        INSTALL_TUNIX=1
    fi

    # If no specific submodules requested, default to all (backward compatibility)
    if [[ "$INSTALL_VERL" == 0 && "$INSTALL_WEBSHOP" == 0 && "$INSTALL_TUNIX" == 0 ]]; then
        print_warning "No specific submodules requested, installing all for backward compatibility"
        INSTALL_VERL=1
        INSTALL_WEBSHOP=1
        INSTALL_TUNIX=1
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository"
        exit 1
    fi
    
    # Check git
    if ! command_exists git; then
        print_error "git is not installed"
        exit 1
    fi
    
    # Check pip
    if ! command_exists pip; then
        print_error "pip is not installed"
        exit 1
    fi
    
    # Check conda (optional but recommended)
    if command_exists conda; then
        print_success "conda found"
    else
        print_warning "conda not found - some webshop prerequisites may need manual installation"
    fi
    
    print_success "Prerequisites check completed"
}

# Setup git submodules
setup_submodules() {
    print_step "Setting up git submodules..."
    
    # Initialize and update submodules
    print_step "Initializing git submodules..."
    git submodule init
    
    print_step "Updating git submodules..."
    git submodule update --recursive
    
    # Verify submodules based on what was requested
    if [[ "$INSTALL_VERL" == 1 ]]; then
        if [ -d "verl" ] && [ "$(ls -A verl)" ]; then
            print_success "verl submodule successfully downloaded"
        else
            print_error "Failed to download verl submodule"
            exit 1
        fi
    fi
    
    if [[ "$INSTALL_WEBSHOP" == 1 ]]; then
        if [ -d "external/webshop-minimal" ] && [ "$(ls -A external/webshop-minimal)" ]; then
            print_success "webshop-minimal submodule successfully downloaded"
        else
            print_warning "webshop-minimal submodule not found or empty"
        fi
    fi
    
    if [[ "$INSTALL_TUNIX" == 1 ]]; then
        if [ -d "tunix" ] && [ "$(ls -A tunix)" ]; then
            print_success "tunix submodule successfully downloaded"
        else
            print_warning "tunix submodule not found or empty"
        fi
    fi
}

# Install verl in editable mode (includes torch dependencies)
install_verl() {
    if [[ "$INSTALL_VERL" != 1 ]]; then
        return
    fi
    
    print_step "Installing verl framework..."
    
    if [ ! -d "verl" ]; then
        print_error "verl directory not found. Run git submodule update first."
        exit 1
    fi
    
    cd verl
    pip install -e .
    cd ..
    
    print_success "verl installed in editable mode"
}

# Install tunix (JAX-based LLM post-training library)
install_tunix() {
    if [[ "$INSTALL_TUNIX" != 1 ]]; then
        return
    fi

    print_step "Installing tunix framework..."

    # Ensure tunix submodule is on the configured branch (default to main if unspecified)
    TUNIX_BRANCH=$(git config --file .gitmodules --get submodule.tunix.branch || echo "")
    if [[ -z "$TUNIX_BRANCH" ]]; then
        TUNIX_BRANCH="main"
    fi
    # Sync and init tunix submodule if needed
    git submodule sync -- tunix >/dev/null 2>&1 || true
    git submodule update --init tunix >/dev/null 2>&1 || true
    if [ -d "tunix/.git" ]; then
        print_step "Checking out tunix branch: $TUNIX_BRANCH"
        git -C tunix fetch origin "$TUNIX_BRANCH" >/dev/null 2>&1 || true
        git -C tunix checkout -B "$TUNIX_BRANCH" "origin/$TUNIX_BRANCH" >/dev/null 2>&1 || git -C tunix checkout "$TUNIX_BRANCH" >/dev/null 2>&1 || true
        git -C tunix reset --hard "origin/$TUNIX_BRANCH" >/dev/null 2>&1 || true
    fi

    # Remove any previously installed tunix to avoid path shadowing
    pip uninstall -y tunix >/dev/null 2>&1 || true

    # ---- Preferred path: local submodule (does not affect caller's CWD) ----
    if [[ -d "tunix" ]]; then
        if [[ -f "tunix/pyproject.toml" || -f "tunix/setup.py" ]]; then
            print_step "Installing tunix from local submodule (editable)"
            if ( cd tunix && pip install -e . ); then
                print_success "tunix installed from local submodule"
                return
            else
                print_warning "Local editable install failed; will try .gitmodules URL (if configured)"
            fi
        else
            print_warning "tunix/ missing build files (pyproject.toml/setup.py); will try .gitmodules URL"
        fi
    else
        print_warning "tunix submodule directory not found; will try .gitmodules URL"
    fi

    # ---- Fallback: install from .gitmodules URL (if present) ----
    TUNIX_URL=$(git config --file .gitmodules --get submodule.tunix.url || echo "")
    TUNIX_BRANCH=$(git config --file .gitmodules --get submodule.tunix.branch || echo "")

    if [[ -n "$TUNIX_URL" ]]; then
        if [[ -n "$TUNIX_BRANCH" ]]; then
            print_step "Installing tunix from submodule URL: $TUNIX_URL (branch: $TUNIX_BRANCH)"
            if pip install "git+$TUNIX_URL@$TUNIX_BRANCH"; then
                print_success "tunix installed from $TUNIX_URL@$TUNIX_BRANCH"
                return
            else
                print_warning "Failed to install tunix from $TUNIX_URL@$TUNIX_BRANCH"
            fi
        else
            print_step "Installing tunix from submodule URL: $TUNIX_URL"
            if pip install "git+$TUNIX_URL"; then
                print_success "tunix installed from $TUNIX_URL"
                return
            else
                print_warning "Failed to install tunix from $TUNIX_URL"
            fi
        fi
    fi

    # ---- Hard failure if neither local nor URL works ----
    print_error "Failed to install tunix. Ensure submodule is checked out and has build files (pyproject.toml or setup.py)."
    exit 1
}


# Install WebShop prerequisites
install_webshop_prereqs() {
    print_step "Installing WebShop prerequisites (faiss, JDK, Maven)"

    # FAISS (CPU) – use conda-forge only, skip Anaconda main channel
    if command_exists conda; then
        conda install -y --override-channels -c conda-forge faiss-cpu
        # Fresh SQLite (≥3.45) so Python's _sqlite3 extension finds all symbols
        conda install -y --override-channels -c conda-forge 'sqlite>=3.45'
        
        # JDK & Maven
        if ! command -v javac &>/dev/null; then
            print_step "JDK not found – installing OpenJDK 21 + Maven"
            conda install -y --override-channels -c conda-forge openjdk=21 maven
        else
            print_success "JDK already installed"
        fi
    else
        print_warning "conda not available - please install faiss-cpu, sqlite>=3.45, openjdk-21, and maven manually"
        # For systems without conda, suggest manual installation
        if ! command -v javac &>/dev/null; then
            print_warning "Java JDK not found. Please install OpenJDK 21 manually"
        fi
    fi
}

# Install webshop
install_webshop() {
    if [[ "$INSTALL_WEBSHOP" != 1 ]]; then
        return
    fi
    
    install_webshop_prereqs
    print_step "Installing WebShop-minimal (may take a few minutes)…"

    # Ensure the submodule exists
    if [[ ! -d external/webshop-minimal ]]; then
        print_warning "WebShop submodule not found; skipping"
        return
    fi

    # Editable install with all extras if defined
    if pip install -e 'external/webshop-minimal[full]' 2>/dev/null; then
        print_success "webshop-minimal installed (editable, full extras)"
    else
        # Fallback: plain editable + its own requirements.txt
        pip install -e external/webshop-minimal
        if [ -f "external/webshop-minimal/requirements.txt" ]; then
            pip install -r external/webshop-minimal/requirements.txt
        fi
        print_success "webshop-minimal installed (editable, basic deps)"
    fi

    # Install spacy models
    print_step "Installing spaCy language models..."
    python -m spacy download en_core_web_sm || print_warning "Failed to download en_core_web_sm"
    python -m spacy download en_core_web_lg || print_warning "Failed to download en_core_web_lg"
}

# Note: torch is installed as part of verl dependencies
# No separate torch installation needed
verify_torch_from_verl() {
    if [[ "$INSTALL_VERL" != 1 ]]; then
        return
    fi
    
    print_step "Verifying torch installation from verl..."
    
    if python -c "import torch; print(f'torch {torch.__version__} available')" 2>/dev/null; then
        TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
        print_success "torch available (version: $TORCH_VERSION)"
    else
        print_error "torch not available - verl installation may have failed"
        exit 1
    fi
}

# Verify critical dependencies for Stage 2
verify_stage1() {
    print_step "Verifying Stage 1 installation..."
    
    # Track if any verification failed
    verification_failed=0
    
    # Verify torch only if verl was installed
    if [[ "$INSTALL_VERL" == 1 ]]; then
        if python -c "import torch" 2>/dev/null; then
            print_success "torch ✓"
        else
            print_error "torch ✗ - Stage 2 will likely fail"
            verification_failed=1
        fi
        
        # Test verl import specifically (it might have different import structure)
        if python -c "import verl" 2>/dev/null; then
            print_success "verl ✓"
        elif python -c "from verl import *" 2>/dev/null; then
            print_success "verl ✓ (wildcard import)"
        elif python -c "import sys; sys.path.append('verl'); import verl" 2>/dev/null; then
            print_success "verl ✓ (path adjusted)"
        else
            print_warning "verl import test failed, but installation completed"
            print_warning "This may be normal if verl uses a different import structure"
        fi
    fi
    
    # Verify tunix if installed
    if [[ "$INSTALL_TUNIX" == 1 ]]; then
        if python -c "import tunix" 2>/dev/null; then
            print_success "tunix ✓"
        else
            print_warning "tunix import test failed, but installation completed"
            print_warning "This may be normal if tunix uses a different import structure"
        fi
    fi
    
    # Verify webshop dependencies if installed
    if [[ "$INSTALL_WEBSHOP" == 1 ]]; then
        if python -c "import faiss" 2>/dev/null; then
            print_success "faiss ✓"
        else
            print_warning "faiss not available - WebShop may not work properly"
        fi
    fi
    
    if [[ "$verification_failed" == 1 ]]; then
        exit 1
    fi
    
    print_success "Stage 1 verification completed - ready for 'pip install -e .'"
}

# Prepare Stage 2: ensure pinned Torch and FlashAttention versions (non-fatal)
prepare_stage2_installation() {
    print_step "Preparing Stage 2: pinning torch==2.7.1 and flash-attn==2.8.0.post2 (use official Linux wheels when available)"

    # Non-fatal preparation
    set +e

    # Detect platform and CUDA
    LINUX=$(python - <<'PY'
import platform; print('1' if platform.system().lower()=='linux' else '0')
PY
)
    CUDA_AVAIL=$(python - <<'PY'
try:
    import torch
    print('1' if getattr(torch, 'cuda', None) and torch.cuda.is_available() else '0')
except Exception:
    print('0')
PY
)

    if [ "$LINUX" = "1" ] && [ "$CUDA_AVAIL" = "1" ]; then
        print_step "Linux + CUDA detected: installing/upgrading torch==2.7.1 and flash-attn==2.8.0.post2"
        pip install --upgrade torch==2.7.1 flash-attn==2.8.0.post2
    else
        print_step "Non-Linux or no CUDA: installing/upgrading torch==2.7.1 (skip flash-attn)"
        pip install --upgrade torch==2.7.1
        if [ "$LINUX" = "1" ]; then
            print_warning "CUDA not available; skipping flash-attn"
        else
            print_warning "Non-Linux platform; skipping flash-attn"
        fi
    fi

    # Verify torch
    python - <<'PY'
import sys
try:
    import torch
    sys.exit(0 if getattr(torch, '__version__', '') == '2.7.1' else 1)
except Exception:
    sys.exit(1)
PY
    if [ $? -eq 0 ]; then
        print_success "torch==2.7.1 ✓"
    else
        print_warning "torch!=2.7.1 (wheel may be unavailable for this Python/CUDA); continuing"
    fi

    # Verify flash-attn only on Linux + CUDA
    if [ "$LINUX" = "1" ] && [ "$CUDA_AVAIL" = "1" ]; then
        python - <<'PY'
import sys
try:
    import importlib
    m = importlib.import_module('flash_attn')
    sys.exit(0 if getattr(m, '__version__', '') == '2.8.0.post2' else 1)
except Exception:
    sys.exit(1)
PY
        if [ $? -eq 0 ]; then
            print_success "flash-attn==2.8.0.post2 ✓"
        else
            print_warning "flash-attn!=2.8.0.post2 (wheel may be unavailable for this setup); continuing"
        fi
    fi

    # Restore 'exit on error'
    set -e
}

# Main function
main() {
    # Parse command line arguments first
    parse_arguments "$@"
    
    echo "=========================================="
    echo "grl Stage 1: Submodule Installation"
    echo "Started at: $(date)"
    echo "=========================================="
    
    # Show what will be installed
    echo "Installation plan:"
    [[ "$INSTALL_VERL" == 1 ]] && echo "  ✓ verl framework"
    [[ "$INSTALL_WEBSHOP" == 1 ]] && echo "  ✓ webshop-minimal"
    [[ "$INSTALL_TUNIX" == 1 ]] && echo "  ✓ tunix framework"
    echo ""
    
    check_prerequisites
    setup_submodules
    install_verl
    install_webshop
    install_tunix
    verify_torch_from_verl
    verify_stage1
    # prepare_stage2_installation
    
    echo "=========================================="
    echo -e "${GREEN}Stage 1 completed successfully!${NC}"
    echo "Completed at: $(date)"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Run: pip install -e ."
    echo "  2. Run: ./scripts/load_dataset.sh"
    echo ""
}

# Run main function
main "$@"

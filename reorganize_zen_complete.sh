#!/bin/bash

# Complete Zen Model Reorganization Script
# Reorganizes all Zen models to match Qwen3 structure

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_header() {
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check for HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    print_error "HuggingFace token not found!"
    echo "Please set your token:"
    echo "  export HF_TOKEN=your_token_here"
    exit 1
fi

print_success "HuggingFace token found"

# Check Python installation
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

print_success "Python 3 found: $(python3 --version)"

# Install required packages
print_header "Installing Required Packages"

pip install -q huggingface_hub transformers torch

print_success "Required packages installed"

# Stage 1: Initial verification
print_header "Stage 1: Initial Verification"
echo "Checking current state of repositories..."

python3 verify_zen_unified.py --report initial_verification.json

echo -e "\nInitial verification complete. Check initial_verification.md for details."
read -p "Continue with reorganization? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Reorganization cancelled"
    exit 0
fi

# Stage 2: Create new repositories with Qwen3 structure
print_header "Stage 2: Creating New Repository Structure"
echo "This will create/update repositories with Qwen3-style structure..."

python3 reorganize_zen_models_hf.py

print_success "Repository structure created"

# Stage 3: Migrate model weights
print_header "Stage 3: Migrating Model Weights"
echo "This will copy model weights from old repos to new ones..."
read -p "Start weight migration? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 migrate_zen_weights.py
    print_success "Weight migration complete"
else
    print_warning "Weight migration skipped"
fi

# Stage 4: Verification
print_header "Stage 4: Final Verification"
echo "Verifying all reorganized models..."

python3 verify_zen_unified.py --report final_verification.json

print_success "Verification complete"

# Stage 5: Optional cleanup
print_header "Stage 5: Cleanup Old Repositories (Optional)"
echo -e "${YELLOW}⚠️  WARNING: This will permanently delete old -instruct repositories!${NC}"
echo "Make sure all migrations have been verified first."
read -p "Proceed with cleanup? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 migrate_zen_weights.py --cleanup-only
    print_success "Cleanup complete"
else
    print_warning "Cleanup skipped - old repositories preserved"
fi

# Final summary
print_header "Reorganization Complete!"

echo "Summary of changes:"
echo "  • zen-nano-instruct → zen-nano (0.6B)"
echo "  • zen-eco-instruct → zen-eco (4B)"
echo "  • zen-omni-instruct → zen-omni (30B)"
echo "  • zen-coder-instruct → zen-coder (480B MoE)"
echo "  • zen-next-instruct → zen-next (80B)"
echo ""
echo "Each model now supports:"
echo "  ✓ Standard mode for fast responses"
echo "  ✓ Thinking mode with <think> blocks"
echo "  ✓ Qwen3-style model cards"
echo "  ✓ Unified configuration"
echo ""
echo "Verification reports:"
echo "  • initial_verification.md - State before reorganization"
echo "  • final_verification.md - State after reorganization"
echo ""

print_success "All done! 🎉"

# Show next steps
echo -e "\n${BLUE}Next Steps:${NC}"
echo "1. Review the verification reports"
echo "2. Test models with both standard and thinking modes"
echo "3. Update documentation and links"
echo "4. Announce the new unified model structure"
echo "5. Monitor community feedback"
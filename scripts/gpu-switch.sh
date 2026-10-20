#!/bin/bash
# ==========================================================
#  GPU Mode Switcher
#  Toggle between compute mode (Intel display) and
#  performance mode (NVIDIA display) with a single command
# ==========================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!!]${NC} $1"; }
err()  { echo -e "${RED}[ERR]${NC} $1"; }
info() { echo -e "${CYAN}[..]${NC} $1"; }

OFFLOAD_CONF="/usr/share/X11/xorg.conf.d/11-nvidia-offload.conf"
OFFLOAD_BACKUP="/usr/share/X11/xorg.conf.d/11-nvidia-offload.conf.bak"

usage() {
    CURRENT=$(prime-select query 2>/dev/null || echo "unknown")
    echo ""
    echo "  GPU Mode Switcher"
    echo "  ================="
    echo ""
    echo "  Current mode: $CURRENT"
    echo ""
    echo "  Usage: sudo bash $0 <mode>"
    echo ""
    echo "  Modes:"
    echo "    compute     Intel drives displays, NVIDIA free for Trove/LM Studio"
    echo "                (prime-select on-demand + offload config)"
    echo ""
    echo "    display     NVIDIA drives all displays (best multi-monitor perf)"
    echo "                (prime-select nvidia, GPU not available for compute)"
    echo ""
    echo "    status      Show current mode and GPU state"
    echo ""
    echo "  Examples:"
    echo "    sudo bash $0 compute    # Before Trove indexing"
    echo "    sudo bash $0 display    # After indexing, for best screen perf"
    echo "    bash $0 status          # Check current state (no sudo needed)"
    echo ""
    exit 0
}

show_status() {
    CURRENT=$(prime-select query 2>/dev/null || echo "unknown")
    echo ""
    echo "  Current GPU State"
    echo "  ================="
    echo ""
    echo "  prime-select:  $CURRENT"

    if [[ "$CURRENT" == "on-demand" ]]; then
        echo "  Mode:          COMPUTE (Intel display, NVIDIA free)"
    elif [[ "$CURRENT" == "nvidia" ]]; then
        echo "  Mode:          DISPLAY (NVIDIA drives screens)"
    else
        echo "  Mode:          $CURRENT"
    fi

    echo "  Driver:        $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "  GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "  VRAM used:     $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "  Monitors:      $(xrandr --query 2>/dev/null | grep " connected" | awk '{print $1}' | tr '\n' ' ')"

    TORCH_OK=$(python3 -c "import torch; t=torch.randn(10,10,device='cuda'); print('yes')" 2>/dev/null || echo "no")
    if [[ "$TORCH_OK" == "yes" ]]; then
        echo "  PyTorch CUDA:  WORKING"
    else
        echo "  PyTorch CUDA:  NOT AVAILABLE"
    fi

    echo ""
    exit 0
}

switch_to_compute() {
    if [[ $EUID -ne 0 ]]; then
        err "Run with sudo:  sudo bash $0 compute"
        exit 1
    fi

    CURRENT=$(prime-select query 2>/dev/null)
    if [[ "$CURRENT" == "on-demand" ]]; then
        log "Already in compute mode (on-demand)"
        echo "  Reboot if you just switched, or GPU is ready to use."
        exit 0
    fi

    info "Switching to COMPUTE mode..."
    echo "  Intel iGPU will drive all displays"
    echo "  The GPU freed for Trove / LM Studio"
    echo ""

    # Switch prime-select
    prime-select on-demand
    log "prime-select set to on-demand"

    # Restore offload config if it was backed up
    if [[ -f "$OFFLOAD_BACKUP" ]] && [[ ! -f "$OFFLOAD_CONF" ]]; then
        mv "$OFFLOAD_BACKUP" "$OFFLOAD_CONF"
        log "Restored PRIME offload Xorg config"
    fi

    # Rebuild initramfs
    update-initramfs -u 2>&1 | tail -1
    log "Initramfs updated"

    echo ""
    warn "REBOOT REQUIRED"
    read -p "Reboot now? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Rebooting..."
        sleep 2
        reboot
    else
        warn "Run 'sudo reboot' when ready"
    fi
}

switch_to_display() {
    if [[ $EUID -ne 0 ]]; then
        err "Run with sudo:  sudo bash $0 display"
        exit 1
    fi

    CURRENT=$(prime-select query 2>/dev/null)
    if [[ "$CURRENT" == "nvidia" ]]; then
        log "Already in display mode (nvidia)"
        echo "  Reboot if you just switched."
        exit 0
    fi

    info "Switching to DISPLAY mode..."
    echo "  NVIDIA GPU will drive all displays (best performance)"
    echo "  GPU will NOT be available for Trove/LM Studio compute"
    echo ""

    # Switch prime-select
    prime-select nvidia
    log "prime-select set to nvidia"

    # Move offload config out of the way (conflicts with nvidia mode)
    if [[ -f "$OFFLOAD_CONF" ]]; then
        mv "$OFFLOAD_CONF" "$OFFLOAD_BACKUP"
        log "Moved PRIME offload config aside (restored on switch to compute)"
    fi

    # Rebuild initramfs
    update-initramfs -u 2>&1 | tail -1
    log "Initramfs updated"

    echo ""
    warn "REBOOT REQUIRED"
    read -p "Reboot now? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Rebooting..."
        sleep 2
        reboot
    else
        warn "Run 'sudo reboot' when ready"
    fi
}

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
case "${1:-}" in
    compute)  switch_to_compute ;;
    display)  switch_to_display ;;
    status)   show_status ;;
    *)        usage ;;
esac

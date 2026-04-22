#!/usr/bin/env bash
# install-desktop.sh - Install mud-puppy-studio as a system-level desktop app.
#
# What this does (per-user, no sudo):
#   1. Symlink ui/build/mud-puppy-studio into ~/bin (must exist and be on PATH)
#   2. Copy the SVG icon to ~/.local/share/icons/hicolor/scalable/apps/
#   3. Install the .desktop file to ~/.local/share/applications/
#   4. Refresh desktop-file + icon caches so the launcher picks it up
#
# Run: ./install-desktop.sh
# Uninstall: ./install-desktop.sh --uninstall

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${HERE}/build/mud-puppy-studio"
APP_NAME="mud-puppy-studio"

BIN_LINK="${HOME}/bin/${APP_NAME}"
ICON_DST="${HOME}/.local/share/icons/hicolor/scalable/apps/${APP_NAME}.svg"
DESKTOP_DST="${HOME}/.local/share/applications/${APP_NAME}.desktop"

uninstall() {
    echo "Removing ${BIN_LINK}"
    rm -f "${BIN_LINK}"
    echo "Removing ${ICON_DST}"
    rm -f "${ICON_DST}"
    echo "Removing ${DESKTOP_DST}"
    rm -f "${DESKTOP_DST}"
    update-desktop-database "${HOME}/.local/share/applications" 2>/dev/null || true
    gtk-update-icon-cache -q -f -t "${HOME}/.local/share/icons/hicolor" 2>/dev/null || true
    echo "Done."
}

if [[ "${1-}" == "--uninstall" || "${1-}" == "-u" ]]; then
    uninstall
    exit 0
fi

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: ${BINARY} not found or not executable." >&2
    echo "Build first: cd ${HERE} && cmake -B build && cmake --build build" >&2
    exit 1
fi

if ! echo ":${PATH}:" | grep -q ":${HOME}/bin:"; then
    echo "WARN: ${HOME}/bin is not on PATH. The symlink will still work" >&2
    echo "      from the Activities/Rofi launcher but not from a shell." >&2
fi

mkdir -p "${HOME}/bin" \
         "${HOME}/.local/share/icons/hicolor/scalable/apps" \
         "${HOME}/.local/share/applications"

echo "Symlinking ${BIN_LINK} -> ${BINARY}"
ln -sf "${BINARY}" "${BIN_LINK}"

# Icon.  Generate if the SVG doesn't live in the repo yet; otherwise copy.
ICON_SRC="${HERE}/assets/icon.svg"
if [[ -f "${ICON_SRC}" ]]; then
    echo "Installing icon from ${ICON_SRC}"
    cp "${ICON_SRC}" "${ICON_DST}"
else
    # Inline fallback: emit the BrandMark-style hexagon directly.
    cat > "${ICON_DST}" <<'SVG'
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128">
  <defs>
    <radialGradient id="centerGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#ffcc33" stop-opacity="1"/>
      <stop offset="70%" stop-color="#ffcc33" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="#ffcc33" stop-opacity="0"/>
    </radialGradient>
    <filter id="cyanGlow" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="1.2" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <rect x="0" y="0" width="128" height="128" rx="16" fill="#04060b"/>
  <g transform="translate(64 64)" filter="url(#cyanGlow)">
    <polygon points="0,-54 46,-27 46,27 0,54 -46,27 -46,-27"
             fill="none" stroke="#00e5ff" stroke-width="2.4"/>
    <polygon points="0,-32 28,-16 28,16 0,32 -28,16 -28,-16"
             fill="none" stroke="#00e5ff" stroke-width="1.6" opacity="0.6"/>
    <line x1="0" y1="-54" x2="0" y2="-32" stroke="#00e5ff" stroke-width="1.2" opacity="0.5"/>
    <line x1="0" y1="32"  x2="0" y2="54"  stroke="#00e5ff" stroke-width="1.2" opacity="0.5"/>
  </g>
  <circle cx="64" cy="64" r="14" fill="url(#centerGlow)"/>
  <circle cx="64" cy="64" r="7"  fill="#ffcc33"/>
</svg>
SVG
    echo "Installed inline icon at ${ICON_DST}"
fi

echo "Installing .desktop at ${DESKTOP_DST}"
cat > "${DESKTOP_DST}" <<DESKTOP
[Desktop Entry]
Type=Application
Version=1.0
Name=Mud-Puppy Studio
GenericName=LLM Fine-Tuning Studio
Comment=ROCm-first LLM fine-tuning studio (mud-puppy v0.4)
Exec=${BIN_LINK}
Icon=${APP_NAME}
Terminal=false
Categories=Development;Science;ArtificialIntelligence;
Keywords=LLM;LoRA;QLoRA;fine-tuning;ROCm;training;
StartupNotify=true
StartupWMClass=${APP_NAME}
DESKTOP

update-desktop-database "${HOME}/.local/share/applications" 2>/dev/null || true
gtk-update-icon-cache -q -f -t "${HOME}/.local/share/icons/hicolor" 2>/dev/null || true

echo
echo "Installed. You can now:"
echo "  - Run 'mud-puppy-studio' from any shell (if ~/bin is on PATH)"
echo "  - Launch from your application menu / Rofi / KRunner"
echo
echo "Uninstall with: $0 --uninstall"

#!/usr/bin/env bash
# Rebuild the deck, strip ZIP folder entries (they break Google Slides / cloud upload), export PDF, render PNG previews.
set -euo pipefail
cd "$(dirname "$0")"
node build_deck.js
zip -dq VM_Reservations_OPT_final.pptx '*/' 2>/dev/null || true
SOFFICE=${SOFFICE:-soffice}
"$SOFFICE" --headless --convert-to pdf VM_Reservations_OPT_final.pptx >/dev/null 2>&1
mkdir -p previews && rm -f previews/slide-*.png
gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r110 -sOutputFile=previews/slide-%02d.png VM_Reservations_OPT_final.pdf
echo "built: VM_Reservations_OPT_final.pptx, VM_Reservations_OPT_final.pdf, previews/"

# Final presentation: Optimizing Cloud VM CPU Reservations

An 8-minute deck for MIT 15.095, in 13.33 × 7.5 widescreen. It has 9 main slides followed by 4 backup slides, marked
"Backup".

| File | What it is |
|---|---|
| `VM_Reservations_OPT_final.pptx` | Editable deck with speaker notes in the Notes pane |
| `VM_Reservations_OPT_final.pdf` | PDF export (LibreOffice) |
| `speaker_notes.md` | Full script, timing table, handoff, likely questions |
| `PRESENTATION_SOURCE_AUDIT.md` | Where every number comes from, plus issues noticed in the report |
| `build_deck.js` | Generator (pptxgenjs); all slide content and notes live here |
| `build.sh` | Rebuilds the .pptx, strips ZIP folder entries, exports the PDF, renders `previews/` |
| `assets/` | Backup-slide figures (report Figures 3 and 4 at 400 dpi) and LaTeX equation images |
| `equations/` | LaTeX sources for the slide 10 equations |
| `previews/` | PNG renders of every slide, for checking layout |

## Editing in PowerPoint or Google Slides

* **Charts:** every chart, the OPT tree, the timeline and the cost illustration are ordinary shapes and text boxes,
  not chart objects. They stay fully editable after a Google Slides import, which converts imported PowerPoint chart
  objects into pictures. To change a bar, resize the rectangle and edit the number beside it.
* **Font:** Arial throughout, available in PowerPoint, Keynote and Google Slides.
* **Google Slides:** upload the `.pptx` to Google Drive, then right-click → Open with → Google Slides (or in Slides:
  File → Import slides). Speaker notes come across. The slide master supplies the footer, the section label and the
  slide numbers.
* **Backup images:** the backup slides use two PNG figures from the report. Edit those in `report/` and re-render if
  needed.
* **Equations (backup slide 10):** typeset in LaTeX and inserted as transparent 600 dpi PNGs (`assets/eq_1.png` to
  `eq_4.png`), so they are pictures, not editable text, in PowerPoint and Google Slides. To change one, edit
  `equations/eqN.tex` and re-render (below). If its pixel size changes, update `EQ_PX` in `build_deck.js`.

## Regenerating

```bash
cd presentation
npm install          # once: installs pptxgenjs 3.12.0 from package.json
./build.sh           # needs LibreOffice (soffice) and Ghostscript (gs) for the PDF and previews
```

Edit text, numbers or notes in `build_deck.js` and rerun `./build.sh`. If the report figures change, re-render the
backup images:

```bash
for f in fig_beta_risk fig_customers; do
  gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r400 -sOutputFile=assets/$f.png ../report/figures/$f.pdf
done
```

To re-render the slide 10 equations:

```bash
cd equations
for i in 1 2 3 4; do
  pdflatex -interaction=nonstopmode eq$i.tex >/dev/null
  gs -q -dNOPAUSE -dBATCH -sDEVICE=pngalpha -r600 -sOutputFile=../assets/eq_$i.png eq$i.pdf
done
rm -f *.aux *.log; file ../assets/eq_*.png   # pixel sizes for EQ_PX
```

Rebuilding overwrites the `.pptx`. Make manual PowerPoint or Google Slides edits in a copy, or carry them back into
`build_deck.js`.

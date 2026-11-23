// Builds presentation/VM_Reservations_OPT_final.pptx (editable in PowerPoint and Google Slides).
// Every chart, the OPT tree and the timeline are drawn from native shapes and text boxes, so they stay editable after
// a Google Slides import (Slides converts imported PowerPoint chart objects to static pictures).
// Numbers come from ML_Final_Report.pdf; see PRESENTATION_SOURCE_AUDIT.md.   Run:  node build_deck.js
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in, like the class template
pres.author = "Chris Low and Emma Li";
pres.title = "Optimizing Cloud VM CPU Reservations";
pres.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };

// ---------------------------------------------------------------- palette (template blues + teal accent)
const BLUE = "215F9B", NUM = "2F5496", NAVY = "17375E", TEAL = "1A9E8F", ORANGE = "D9822B";
const CARD = "F2F2F2", TEXT = "1F1F1F", MUTED = "555555", LINE = "A6B3C2";
const MBLUE = "7FA7D3", GREY1 = "A9B8C8", GREY2 = "C3CDD8", GREY3 = "D9DEE4";
const TIER = { 25: "D6E6FA", 50: "9CC3F0", 75: "4A8AD8", 100: "173F73" };
const FONT = "Arial";
const FOOT = "MIT 15.095  ·  Optimizing Cloud VM CPU Reservations  ·  Chris Low and Emma Li";

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  objects: [
    { rect: { x: 0.5, y: 0.3, w: 0.07, h: 0.32, fill: { color: BLUE }, line: { color: BLUE, width: 0 } } },
    { text: { text: FOOT, options: { x: 0.5, y: 7.0, w: 9, h: 0.3, fontFace: FONT, fontSize: 10, color: MUTED, margin: 0 } } },
    { placeholder: { options: { name: "section", type: "body", x: 0.68, y: 0.26, w: 6, h: 0.4, fontFace: FONT, fontSize: 14,
        color: TEXT, margin: 0, valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.78, w: 12.33, h: 0.75, fontFace: FONT, fontSize: 28,
        color: BLUE, bold: false, margin: 0, valign: "middle", align: "left" }, text: "" } },
  ],
  slideNumber: { x: 12.33, y: 7.0, w: 0.5, h: 0.3, fontFace: FONT, fontSize: 10, color: MUTED, align: "right" },
});
pres.defineSlideMaster({
  title: "PLAIN",
  background: { color: "FFFFFF" },
  objects: [],
  slideNumber: { x: 12.33, y: 7.0, w: 0.5, h: 0.3, fontFace: FONT, fontSize: 10, color: MUTED, align: "right" },
});

// ---------------------------------------------------------------- helpers
function T(slide, text, o) {
  slide.addText(text, Object.assign({ fontFace: FONT, fontSize: 14, color: TEXT, margin: 0, valign: "top", isTextBox: true }, o));
}
function R(slide, o) {
  slide.addShape(pres.shapes.RECTANGLE, Object.assign({ line: { color: o.fill && o.fill.color ? o.fill.color : "FFFFFF", width: 0 } }, o));
}
function RR(slide, o) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, Object.assign({ rectRadius: 0.06 }, o));
}
function L(slide, x1, y1, x2, y2, o) {
  // line from (x1,y1) to (x2,y2)
  const opts = Object.assign({ color: LINE, width: 1 }, o || {});
  slide.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
    flipH: (x2 < x1) !== (y2 < y1) ? true : false, line: opts,
  });
}
function content(section, title, sectionTitle) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle });
  s.addText(section, { placeholder: "section" });
  s.addText(title, { placeholder: "title" });
  return s;
}
// template card: light grey panel with a blue left edge
function card(slide, x, y, w, h) {
  R(slide, { x, y, w, h, fill: { color: CARD } });
  R(slide, { x, y, w: 0.08, h, fill: { color: BLUE } });
}
// template label box: blue label cell + grey text cell
function labelBox(slide, x, y, w, h, label, text, lw, fs) {
  lw = lw || 1.9; fs = fs || 14;
  R(slide, { x, y, w: lw, h, fill: { color: BLUE } });
  T(slide, label, { x: x + 0.15, y, w: lw - 0.25, h, color: "FFFFFF", bold: true, fontSize: fs, valign: "middle" });
  R(slide, { x: x + lw, y, w: w - lw, h, fill: { color: CARD } });
  T(slide, text, { x: x + lw + 0.2, y, w: w - lw - 0.35, h, fontSize: fs, valign: "middle" });
}
function notes(slide, n) { slide.addNotes(n); }

// ================================================================ 1. Title
pres.addSection({ title: "Main presentation" });
{
  const s = pres.addSlide({ masterName: "PLAIN", sectionTitle: "Main presentation" });
  R(s, { x: 8.75, y: 0, w: 4.583, h: 7.5, fill: { color: BLUE } });
  T(s, "Optimizing Cloud VM\nCPU Reservations", { x: 0.7, y: 1.55, w: 7.7, h: 1.75, fontSize: 40, bold: true, color: BLUE, valign: "bottom" });
  T(s, "Predictive vs. Prescriptive Methods and Optimal Policy Trees", { x: 0.7, y: 3.5, w: 7.7, h: 0.9, fontSize: 22, color: TEXT });
  T(s, "Chris Low and Emma Li", { x: 0.7, y: 4.75, w: 7.7, h: 0.4, fontSize: 18, bold: true, color: TEXT });
  T(s, "MIT 15.095 Machine Learning under a Modern Optimization Lens  ·  December 2, 2026",
    { x: 0.7, y: 5.2, w: 7.9, h: 0.35, fontSize: 13, color: MUTED });
  T(s, "Customers request CPU they rarely use.", { x: 9.25, y: 2.2, w: 3.6, h: 1.0, fontSize: 20, color: "FFFFFF" });
  T(s, "How much should a provider reserve when each VM is created?",
    { x: 9.25, y: 3.35, w: 3.6, h: 1.8, fontSize: 24, bold: true, color: "FFFFFF" });
  notes(s, `[Chris | 0:20 | running 0:20]
Main message: Customers ask for more CPU than they use. When a VM is created, how much should the provider set aside?

Script:
Hi, I'm Chris, and this is Emma. When customers create a virtual machine, they ask for a number of CPUs, but most VMs use much less. So our question is simple: when a VM is created, how much CPU should the provider actually reserve?

Transition: Here is what that decision looks like for one VM.`);
}

// ================================================================ 2. The decision
{
  const s = content("Problem", "Each new VM needs a CPU reservation before its workload is known", "Main presentation");
  T(s, "Illustrative VM: requests 8 vCPUs, runs at P95 = 40%",
    { x: 0.6, y: 1.72, w: 6.6, h: 0.35, fontSize: 15, bold: true });
  // legend
  R(s, { x: 0.65, y: 2.2, w: 0.3, h: 0.18, fill: { color: TIER[50] } });
  T(s, "reserved", { x: 1.02, y: 2.13, w: 1.1, h: 0.3, fontSize: 12, color: MUTED });
  R(s, { x: 2.1, y: 2.2, w: 0.3, h: 0.18, fill: { color: ORANGE } });
  T(s, "shortfall (demand above reservation)", { x: 2.47, y: 2.13, w: 3.4, h: 0.3, fontSize: 12, color: MUTED });
  L(s, 5.75, 2.29, 6.15, 2.29, { color: ORANGE, width: 2, dashType: "dash" });
  T(s, "P95 demand", { x: 6.22, y: 2.13, w: 1.1, h: 0.3, fontSize: 12, color: MUTED });
  const top = 2.9, hgt = 2.5, bot = top + hgt, Y = 0.40;
  const tiers = [25, 50, 75, 100];
  const costs = ["1.00", "0.50", "0.75", "1.00"], parts = ["0.25 + 5 × 0.15", "0.50 + 0", "0.75 + 0", "1.00 + 0"];
  tiers.forEach((t, i) => {
    const x = 0.95 + i * 1.6, w = 1.05, best = t === 50;
    R(s, { x, y: top, w, h: hgt, fill: { color: "FFFFFF" }, line: { color: LINE, width: 1 } });
    R(s, { x, y: bot - (t / 100) * hgt, w, h: (t / 100) * hgt, fill: { color: best ? BLUE : TIER[50] } });
    if (t / 100 < Y) R(s, { x, y: bot - Y * hgt, w, h: (Y - t / 100) * hgt, fill: { color: ORANGE } });
    L(s, x - 0.08, bot - Y * hgt, x + w + 0.08, bot - Y * hgt, { color: ORANGE, width: 2, dashType: "dash" });
    T(s, `reserve ${t}%`, { x: x - 0.25, y: bot + 0.08, w: w + 0.5, h: 0.32, fontSize: 15, bold: true, align: "center" });
    T(s, `cost ${costs[i]}`, { x: x - 0.25, y: bot + 0.42, w: w + 0.5, h: 0.3, fontSize: 14, align: "center",
      color: best ? BLUE : TEXT, bold: best });
    T(s, parts[i], { x: x - 0.3, y: bot + 0.74, w: w + 0.6, h: 0.28, fontSize: 11, align: "center", color: MUTED });
  });
  T(s, "lowest cost", { x: 2.4, y: top - 0.3, w: 1.35, h: 0.28, fontSize: 12, bold: true, color: BLUE, align: "center" });
  // right: cost function
  R(s, { x: 7.45, y: 1.72, w: 5.38, h: 1.15, fill: { color: CARD } });
  T(s, "C(r, Y; β) = r + β · max(0, Y − r)", { x: 7.55, y: 1.72, w: 5.2, h: 1.15, fontSize: 26, bold: true,
    color: NAVY, align: "center", valign: "middle" });
  const defs = [["r", "reserved share of the VM's CPU request: 25, 50, 75 or 100%"],
    ["Y", "realized P95 CPU utilization, as a share of the request"],
    ["β", "penalty per unit of demand above r; β = 5 in our main results (assumed, not an Azure price)"]];
  defs.forEach(([k, v], i) => {
    T(s, k, { x: 7.55, y: 3.1 + i * 0.72, w: 0.45, h: 0.6, fontSize: 22, bold: true, color: BLUE });
    T(s, v, { x: 8.1, y: 3.12 + i * 0.72, w: 4.7, h: 0.62, fontSize: 14 });
  });
  labelBox(s, 7.45, 5.4, 5.38, 0.72, "Trade-off:", "Reserve too much: wasted capacity.\nReserve too little: shortfall penalty.", 1.55, 14);
  T(s, "Modeled decision: hypothetical reservations scored on observed traces. No Azure VM was resized.",
    { x: 7.45, y: 6.3, w: 5.38, h: 0.5, fontSize: 11, italic: true, color: MUTED });
  notes(s, `[Chris | 0:55 | running 1:15]
Main message: We pick one of four reservation levels, and a simple cost trades wasted capacity against shortfall.

Script:
Take a VM that requests 8 virtual CPUs. Its 95th-percentile usage turns out to be 40 percent of that request. We can reserve 25, 50, 75 or 100 percent. Our cost charges for what we reserve, plus beta times any demand above the reservation. With beta equal to 5, being short costs five times as much as reserving. Reserving 25 percent leaves the VM 15 points short, so it costs 1.0. Reserving 50 percent costs 0.5, the best choice here. Reserving 100 percent is safe, but it also costs 1.0. To be clear, this is a modeled cost. We score hypothetical reservations on real traces. We never resized any Azure VMs.

Transition: So where do those traces come from?`);
}

// ================================================================ 3. Data and design
{
  const s = content("Data", "2.47 million Azure VMs, split strictly by time", "Main presentation");
  const stats = [["2.47M", "VMs created during the trace\n(Azure Public Dataset V2)"], ["30 days", "of VM activity in one\nAzure region"],
    ["26", "creation-time features, including\nthe customer's past CPU use"]];
  stats.forEach(([n, l], i) => {
    const x = 0.6 + i * 4.13;
    card(s, x, 1.75, 3.85, 1.45);
    T(s, n, { x: x + 0.35, y: 1.88, w: 3.3, h: 0.62, fontSize: 32, bold: true, color: NUM });
    T(s, l, { x: x + 0.35, y: 2.52, w: 3.4, h: 0.62, fontSize: 14 });
  });
  T(s, "Strict chronological split", { x: 0.6, y: 3.48, w: 6, h: 0.35, fontSize: 15, bold: true });
  const x0 = 0.6, W = 12.13, px = W / 30, by = 3.92, bh = 0.78;
  const segs = [[0, 21, BLUE, "Train  ·  days 0–20  ·  1.70M VMs", 16], [21, 25, MBLUE, "Validation\n275K VMs", 12],
    [25, 30, TEAL, "Test  ·  days 25–29\n427K VMs", 13]];
  segs.forEach(([a, b, col, lab, fs]) => {
    R(s, { x: x0 + a * px, y: by, w: (b - a) * px - 0.03, h: bh, fill: { color: col } });
    T(s, lab, { x: x0 + a * px, y: by, w: (b - a) * px - 0.03, h: bh, fontSize: fs, bold: true, color: "FFFFFF",
      align: "center", valign: "middle" });
  });
  [[0, "day 0"], [21, "day 21"], [25, "day 25"], [30, "day 30"]].forEach(([d, t]) => {
    T(s, t, { x: x0 + d * px - 0.5, y: by + bh + 0.05, w: 1.0, h: 0.28, fontSize: 11, color: MUTED, align: "center" });
  });
  labelBox(s, 0.6, 5.25, 12.13, 0.62, "No leakage:",
    "Customer history uses only that customer's VMs that had already finished when the new VM was created.", 1.95, 14);
  labelBox(s, 0.6, 6.05, 12.13, 0.62, "Labels:",
    "Training and validation VMs count only if they ended before the cutoff. All tuning on validation; the test set is used once.", 1.95, 14);
  notes(s, `[Chris | 0:45 | running 2:00]
Main message: Real Azure data, features known at creation time, and a time-based split with no leakage.

Script:
We use the Azure Public Dataset: about 2.5 million VMs created over 30 days in one region. Each VM has 26 features known at creation time, including its size and the history of the same customer and deployment. We split by time. We train on the first 21 days, tune on the next four, and test on the last five. Two rules prevent leakage. Customer history only uses VMs that had already finished. And a training VM only counts if it ended before the cutoff.

Transition: With that data, we compared four ways to make the decision.`);
}

// ================================================================ 4. Methods and the OPT tree
{
  const s = content("Method", "Exact OPT learns a three-question reservation rule", "Main presentation");
  const chips = ["Predict P95, round up", "Greedy policy tree", "Exact OPT (ours)", "Cost-aware Random Forest"];
  chips.forEach((c, i) => {
    const x = 0.6 + i * 3.08, hi = i === 2;
    RR(s, { x, y: 1.68, w: 2.9, h: 0.48, fill: { color: hi ? BLUE : CARD }, line: { color: hi ? BLUE : CARD, width: 0 } });
    T(s, c, { x, y: 1.68, w: 2.9, h: 0.48, fontSize: 14, bold: hi, color: hi ? "FFFFFF" : TEXT, align: "center", valign: "middle" });
  });
  // tree geometry (saved beta = 5 tree, results/final_models/OPT-exact_full_b5.json)
  const leafX = [0, 1, 2, 3, 4, 5, 6, 7].map(i => 1.05 + i * 1.6);
  const l2X = [0, 1, 2, 3].map(i => (leafX[2 * i] + leafX[2 * i + 1]) / 2);
  const l1X = [(l2X[0] + l2X[1]) / 2, (l2X[2] + l2X[3]) / 2];
  const rootX = (l1X[0] + l1X[1]) / 2;
  const Y0 = 2.45, Y1 = 3.33, Y2 = 4.21, YL = 5.09, NH = 0.6;
  const node = (cx, y, w, txt) => {
    RR(s, { x: cx - w / 2, y, w, h: NH, fill: { color: CARD }, line: { color: LINE, width: 0.75 } });
    T(s, txt, { x: cx - w / 2, y, w, h: NH, fontSize: 12, align: "center", valign: "middle" });
  };
  const edge = (px, py, cx, cy, lab) => {
    L(s, px, py, cx, cy, { color: LINE, width: 1 });
    const mx = px + 0.45 * (cx - px), my = py + 0.45 * (cy - py);
    T(s, lab, { x: mx - 0.22, y: my - 0.12, w: 0.44, h: 0.22, fontSize: 10, color: MUTED, align: "center", valign: "middle",
      fill: { color: "FFFFFF" } });
  };
  // edges first (drawn under nodes)
  edge(rootX, Y0 + NH, l1X[0], Y1, "yes"); edge(rootX, Y0 + NH, l1X[1], Y1, "no");
  [0, 1].forEach(i => { edge(l1X[i], Y1 + NH, l2X[2 * i], Y2, "yes"); edge(l1X[i], Y1 + NH, l2X[2 * i + 1], Y2, "no"); });
  [0, 1, 2, 3].forEach(i => { edge(l2X[i], Y2 + NH, leafX[2 * i], YL, "yes"); edge(l2X[i], Y2 + NH, leafX[2 * i + 1], YL, "no"); });
  node(rootX, Y0, 3.5, "Share of customer's past VMs\nwith P95 ≤ 75% is ≤ 85.8%?");
  node(l1X[0], Y1, 3.0, "Customer's median past P95\n≤ 62.0%?");
  node(l1X[1], Y1, 3.0, "Share of past VMs with\nP95 ≤ 50% is ≤ 84.9%?");
  node(l2X[0], Y2, 2.2, "VM size: vCPUs ≤ 2?");
  node(l2X[1], Y2, 2.9, "Share of past VMs with\nP95 ≤ 75% is ≤ 61.9%?");
  node(l2X[2], Y2, 2.9, "Deployment's mean\npast P95 ≤ 38.6%?");
  node(l2X[3], Y2, 2.9, "Customer's mean\npast P95 ≤ 21.1%?");
  const leaves = [[75, "19.9%"], [100, "5.0%"], [100, "32.9%"], [75, "6.0%"], [50, "7.5%"], [75, "8.3%"], [25, "16.3%"], [50, "4.3%"]];
  leaves.forEach(([t, sh], i) => {
    const x = leafX[i] - 0.65;
    RR(s, { x, y: YL, w: 1.3, h: 0.5, fill: { color: TIER[t] }, line: { color: TIER[t], width: 0 } });
    T(s, `reserve ${t}%`, { x, y: YL, w: 1.3, h: 0.5, fontSize: 13, bold: true, color: t >= 75 ? "FFFFFF" : TEXT,
      align: "center", valign: "middle" });
    T(s, `${sh} of VMs`, { x: x - 0.1, y: YL + 0.54, w: 1.5, h: 0.26, fontSize: 11, color: MUTED, align: "center" });
  });
  labelBox(s, 0.6, 6.12, 12.13, 0.68, "Exact solver:",
    "Our dynamic program checks every split on 8 features × 31 thresholds: the certified lowest training cost in this tree class, in about 5 s per depth-3 tree.",
    1.95, 14);
  notes(s, `[Chris | 1:10 | running 3:10]
Main message: Exact OPT turns customer history into a short, readable rule, and our solver certifies it is the best tree of its kind on the training data.

Script:
The first approach predicts P95 usage and rounds up to the next tier. The other three learn from the cost itself. A greedy policy tree picks one split at a time. A cost-aware random forest estimates the shortfall at each tier and picks the cheapest. Our focus is the exact optimal policy tree. This is our actual tree for beta equals 5. It first asks how often the customer's past VMs ran above 75 percent. Near-idle customers, about 16 percent of VMs, get 25 percent. Customers with frequent hot VMs mostly get 100. We wrote our own dynamic program to find it. It checks every combination of splits on 8 features with 31 thresholds each. So within that class, it has the certified lowest training cost, and it solves in about 5 seconds.

Transition: So how much does each approach save?`);
}

// ================================================================ 5. Main results
{
  const s = content("Results", "Cost-aware policies save about 20%; exact OPT ties greedy", "Main presentation");
  const rows = [["Cost-aware Random Forest", 21.6, TEAL, true], ["Exact OPT (depth 3)", 20.1, BLUE, true],
    ["Greedy policy tree", 20.0, MBLUE, false], ["RF prediction + tuned margin", 19.8, GREY1, false],
    ["Customer-history rule", 19.1, GREY2, false], ["RF prediction, round up", 14.2, GREY3, false]];
  const lx = 0.6, lw = 3.0, bx = 3.75, scale = 3.3 / 25, rh = 0.66, y0 = 1.85;
  rows.forEach(([lab, v, col, hi], i) => {
    const y = y0 + i * rh;
    T(s, lab, { x: lx, y, w: lw, h: 0.48, fontSize: 15, bold: hi, align: "right", valign: "middle" });
    R(s, { x: bx, y: y + 0.04, w: v * scale, h: 0.4, fill: { color: col } });
    T(s, `${v.toFixed(1)}%`, { x: bx + v * scale + 0.1, y, w: 0.95, h: 0.48, fontSize: 16, bold: true,
      color: hi ? col : TEXT, valign: "middle" });
  });
  L(s, bx, y0 - 0.05, bx, y0 + rows.length * rh - 0.15, { color: MUTED, width: 1 });
  T(s, "Modeled-cost saving vs. the best fixed tier (reserve 100%), test set, β = 5",
    { x: 0.6, y: 5.95, w: 7.3, h: 0.3, fontSize: 12, color: MUTED });
  T(s, "Hindsight oracle (not deployable): 41.1%", { x: 0.6, y: 6.3, w: 7.3, h: 0.3, fontSize: 12, color: MUTED, italic: true });
  const cards = [["19–22%", "Every method trained or tuned on the cost lands in this band."],
    ["+0.1 pt", "Exact OPT over greedy on the test set. Not statistically significant at β = 5."],
    ["21.6%", "Cost-aware RF is best, but it is 100 trees, not a rule you can read."]];
  cards.forEach(([n, t], i) => {
    const y = 1.85 + i * 1.5;
    card(s, 8.3, y, 4.43, 1.3);
    T(s, n, { x: 8.6, y: y + 0.12, w: 3.9, h: 0.5, fontSize: 26, bold: true, color: i === 2 ? TEAL : NUM });
    T(s, t, { x: 8.6, y: y + 0.62, w: 3.95, h: 0.62, fontSize: 14 });
  });
  notes(s, `[Chris | 1:10 | running 4:20]
Main message: Every cost-aware policy saves about 20 percent; exact OPT barely beats greedy, and the cost-aware forest is best but not interpretable.

Script:
This chart shows savings in modeled cost against the best fixed rule. At beta equals 5, that rule reserves 100 percent for everyone. Predicting usage and rounding up saves 14 percent. Once a method accounts for the cost, savings jump to about 20 percent. The customer-history rule, the forest with a tuned margin, greedy and exact OPT all land within about a point of each other. Exact OPT saves 20.1 percent and greedy 20.0, and that difference is not statistically significant. OPT does reach a lower training cost, but that small edge doesn't carry over to new data. The cost-aware forest is best at 21.6 percent. But it's 100 trees, so nobody can read it.

Handoff: Emma will show where these savings come from.`);
}

// ================================================================ 6. Customer history
{
  const s = content("Results", "Customer history raises OPT savings from 6% to 20%", "Main presentation");
  const groups = [["VM size and\ncreation time", 6.0, 5.4, 7.2], ["+ customer\nhistory", 20.3, 20.2, 20.9],
    ["+ deployment\nhistory", 20.1, 20.0, 21.6]];
  const base = 5.35, scale = 3.0 / 22, centers = [1.85, 4.3, 6.75];
  // legend
  [["exact OPT", BLUE], ["greedy tree", MBLUE], ["cost-aware RF", TEAL]].forEach(([t, c], i) => {
    R(s, { x: 0.75 + i * 2.1, y: 1.83, w: 0.3, h: 0.2, fill: { color: c } });
    T(s, t, { x: 1.12 + i * 2.1, y: 1.76, w: 1.8, h: 0.32, fontSize: 13, color: MUTED });
  });
  R(s, { x: centers[1] - 1.15, y: 2.25, w: 2.3, h: base - 2.25 + 0.02, fill: { color: "EEF3FA" } });
  groups.forEach(([lab, opt, gr, rf], i) => {
    const c = centers[i];
    R(s, { x: c - 0.95, y: base - opt * scale, w: 1.1, h: opt * scale, fill: { color: BLUE } });
    R(s, { x: c + 0.2, y: base - gr * scale, w: 0.32, h: gr * scale, fill: { color: MBLUE } });
    R(s, { x: c + 0.57, y: base - rf * scale, w: 0.32, h: rf * scale, fill: { color: TEAL } });
    T(s, `${opt.toFixed(1)}%`, { x: c - 1.1, y: base - opt * scale - 0.48, w: 1.4, h: 0.42, fontSize: 20, bold: true,
      color: BLUE, align: "center" });
    T(s, lab, { x: c - 1.15, y: base + 0.1, w: 2.3, h: 0.6, fontSize: 14, align: "center" });
  });
  L(s, 0.65, base, 7.9, base, { color: MUTED, width: 1 });
  T(s, "+14.3 pts with\ncustomer history", { x: 0.85, y: 3.2, w: 2.15, h: 0.65, fontSize: 15, bold: true, color: NAVY, align: "center" });
  T(s, "Test set, β = 5; each method retrained on each feature set. At β = 10, OPT goes from 0.9% to 15.1%.",
    { x: 0.6, y: 6.25, w: 7.5, h: 0.5, fontSize: 12, color: MUTED });
  card(s, 8.45, 1.8, 4.28, 4.0);
  T(s, "Why history matters", { x: 8.75, y: 1.98, w: 3.8, h: 0.4, fontSize: 18, bold: true, color: NUM });
  T(s, [
    { text: "99.9% of test VMs come from customers with an earlier, finished VM.", options: { bullet: true, breakLine: true } },
    { text: "Those past VMs are a direct sample of how this customer's workloads run, especially how often they run hot.",
      options: { bullet: true, breakLine: true } },
    { text: "Once history is in, the choice of algorithm adds at most about 1.5 points.", options: { bullet: true } },
  ], { x: 8.75, y: 2.5, w: 3.8, h: 3.7, fontSize: 15, paraSpaceAfter: 12 });
  notes(s, `[Emma | 0:55 | running 5:15]
Main message: Most of the value comes from the customer's own history, not from a fancier algorithm.

Script:
Thanks. To see where the value comes from, we retrained each method on three feature sets. With only VM size and creation time, OPT saves 6 percent. Add the customer's history and it jumps to 20.3 percent. Adding deployment history barely changes anything. Greedy and the random forest show the same jump. Why does history matter so much? Almost every test VM comes from a returning customer. Their past VMs are basically a sample of how their next VM will behave, especially how often it runs hot. Once that is in, the choice of algorithm adds at most about one and a half points.

Transition: History tells us what to expect. The next question is how to turn that into a decision.`);
}

// ================================================================ 7. Prediction vs prescription
{
  const s = content("Results", "Same forest, better decision rule: savings rise from 14% to 21%", "Main presentation");
  T(s, "One P95 random forest, three decision rules (β = 5)",
    { x: 0.6, y: 1.72, w: 6.7, h: 0.35, fontSize: 15, bold: true });
  const cols = [["A", "Predict P95,\nround up", 14.2, "0.858", GREY2], ["B", "Predict P95 +\ntuned margin", 19.8, "0.802", MBLUE],
    ["C", "Forest-weighted\nexpected cost", 21.4, "0.786", TEAL]];
  const base = 4.95, scale = 2.5 / 25;
  cols.forEach(([k, lab, v, c, col], i) => {
    const x = 1.0 + i * 2.15;
    R(s, { x, y: base - v * scale, w: 1.35, h: v * scale, fill: { color: col } });
    T(s, `${v.toFixed(1)}%`, { x: x - 0.2, y: base - v * scale - 0.46, w: 1.75, h: 0.4, fontSize: 20, bold: true,
      color: k === "C" ? TEAL : TEXT, align: "center" });
    T(s, `${k}. ${lab}`, { x: x - 0.35, y: base + 0.1, w: 2.05, h: 0.6, fontSize: 14, align: "center", bold: k === "C" });
    T(s, `cost ${c}`, { x: x - 0.2, y: base + 0.72, w: 1.75, h: 0.28, fontSize: 12, color: MUTED, align: "center" });
  });
  L(s, 0.75, base, 7.25, base, { color: MUTED, width: 1 });
  card(s, 7.75, 1.72, 4.98, 3.95);
  T(s, "Why C beats B with the same forest", { x: 8.05, y: 1.88, w: 4.5, h: 0.4, fontSize: 17, bold: true, color: NUM });
  T(s, "VMs predicted at 40–50% P95 (24% of test VMs) actually span all four tiers.",
    { x: 8.05, y: 2.38, w: 4.5, h: 0.65, fontSize: 14 });
  [["41%", "exceed their reservation when every one is rounded up to 50%", GREY1],
    ["16%", "exceed when C picks each tier from the demand of similar past VMs", TEAL]].forEach(([n, t, col], i) => {
    const y = 3.2 + i * 1.18;
    T(s, n, { x: 8.05, y, w: 1.25, h: 0.62, fontSize: 30, bold: true, color: i ? TEAL : MUTED });
    T(s, t, { x: 9.35, y: y + 0.02, w: 3.2, h: 0.9, fontSize: 14 });
  });
  labelBox(s, 0.6, 6.05, 12.13, 0.68, "Takeaway:",
    "Better predictions do not automatically imply better decisions. The decision rule matters.", 1.95, 16);
  notes(s, `[Emma | 0:55 | running 6:10]
Main message: With the same predictive model, choosing the reservation by expected cost beats rounding up a point prediction.

Script:
Here we hold the prediction model fixed. All three rules use the same random forest. Rule A predicts P95 and rounds up, saving 14.2 percent. Rule B adds a safety margin tuned on validation and saves 19.8. Rule C uses the forest to estimate the whole distribution of demand, then picks the tier with the lowest expected cost. It saves 21.4 percent. Why? VMs predicted at 40 to 50 percent actually span all four tiers. Rounding up gives them all 50 percent, and 41 percent run over. Rule C adapts to each VM, and only 16 percent run over. So better predictions don't automatically give better decisions.

Transition: How complex does the policy itself need to be?`);
}

// ================================================================ 8. Interpretability, complexity, risk
{
  const s = content("Trade-offs", "A four-leaf tree keeps 98.6% of the savings", "Main presentation");
  const pols = [["Exact OPT, 4 leaves", "2 questions on customer history", 19.8, 7.7, true],
    ["Exact OPT, 8 leaves", "3 questions (depth 3)", 20.1, 9.3, false],
    ["Cost-aware Random Forest", "100 trees, not readable", 21.6, 9.1, false]];
  pols.forEach(([h, sub, sav, exc, hi], i) => {
    const x = 0.6 + i * 4.13, y = 1.72, w = 3.85;
    R(s, { x, y, w, h: 3.15, fill: { color: CARD }, line: hi ? { color: BLUE, width: 2 } : { color: CARD, width: 0 } });
    T(s, h, { x: x + 0.3, y: y + 0.15, w: w - 0.5, h: 0.4, fontSize: 18, bold: true, color: i === 2 ? TEAL : NUM });
    T(s, sub, { x: x + 0.3, y: y + 0.55, w: w - 0.5, h: 0.32, fontSize: 13, color: MUTED });
    T(s, "Modeled saving", { x: x + 0.3, y: y + 1.0, w: 2.2, h: 0.3, fontSize: 13 });
    T(s, `${sav.toFixed(1)}%`, { x: x + 2.2, y: y + 0.92, w: 1.35, h: 0.42, fontSize: 22, bold: true, align: "right" });
    R(s, { x: x + 0.3, y: y + 1.42, w: 3.25, h: 0.16, fill: { color: "FFFFFF" } });
    R(s, { x: x + 0.3, y: y + 1.42, w: 3.25 * sav / 25, h: 0.16, fill: { color: i === 2 ? TEAL : BLUE } });
    T(s, "VMs above reservation", { x: x + 0.3, y: y + 1.85, w: 2.3, h: 0.3, fontSize: 13 });
    T(s, `${exc.toFixed(1)}%`, { x: x + 2.2, y: y + 1.77, w: 1.35, h: 0.42, fontSize: 22, bold: true, align: "right", color: ORANGE });
    R(s, { x: x + 0.3, y: y + 2.27, w: 3.25, h: 0.16, fill: { color: "FFFFFF" } });
    R(s, { x: x + 0.3, y: y + 2.27, w: 3.25 * exc / 12, h: 0.16, fill: { color: ORANGE } });
    T(s, i === 0 ? "keeps 98.6% of the 8-leaf savings" : (i === 1 ? "main model" : "lowest modeled cost"),
      { x: x + 0.3, y: y + 2.6, w: w - 0.5, h: 0.32, fontSize: 13, italic: true, color: MUTED });
  });
  T(s, "Test set, β = 5. \"Above reservation\" means realized P95 exceeded the modeled reservation; it is not measured throttling or an SLA violation.",
    { x: 0.6, y: 4.98, w: 12.13, h: 0.3, fontSize: 12, color: MUTED });
  labelBox(s, 0.6, 5.45, 12.13, 0.6, "Simpler tree:",
    "4 vs. 8 leaves: cost difference not statistically significant. That is not proof of equivalence, but extra leaves buy little.", 1.95, 14);
  labelBox(s, 0.6, 6.18, 12.13, 0.6, "Customers:",
    "The median subscription saves 10.4% under OPT, versus 20.1% overall. Large customers drive the total.", 1.95, 14);
  notes(s, `[Emma | 1:10 | running 7:20]
Main message: A much simpler tree keeps nearly all the savings, and the cheapest policy is not the one that exceeds its reservation least.

Script:
A four-leaf tree, just two questions about customer history, saves 19.8 percent. That's 98.6 percent of the eight-leaf tree's savings, and the difference isn't statistically significant. That doesn't prove they're equivalent, but the extra leaves buy very little. The orange row is how often realized P95 lands above the reservation. It's not measured throttling. Notice the cheapest policy isn't the one that exceeds least: 7.7 percent for four leaves, 9.3 for eight, and 9.1 for the forest. Averages also hide differences between customers. The median customer saves 10.4 percent under OPT, about half the overall 20.1.

Transition: Let me pull this together.`);
}

// ================================================================ 9. Conclusions
{
  const s = pres.addSlide({ masterName: "PLAIN", sectionTitle: "Main presentation" });
  R(s, { x: 0, y: 0, w: 4.55, h: 7.5, fill: { color: BLUE } });
  T(s, "Takeaway", { x: 0.55, y: 1.7, w: 3.5, h: 0.4, fontSize: 16, color: "D6E6FA", bold: true });
  T(s, "About 20% modeled savings, driven by customer history and decision-aware learning.",
    { x: 0.55, y: 2.2, w: 3.55, h: 3.0, fontSize: 26, bold: true, color: "FFFFFF" });
  R(s, { x: 5.05, y: 0.3, w: 0.07, h: 0.32, fill: { color: BLUE } });
  T(s, "Conclusion", { x: 5.23, y: 0.26, w: 4, h: 0.4, fontSize: 14, valign: "middle" });
  T(s, "What we learned", { x: 5.05, y: 0.78, w: 7.7, h: 0.75, fontSize: 28, color: BLUE, valign: "middle" });
  const pts = [["Customer history carries the signal.", "OPT savings rise from 6% to 20% once past customer usage is added."],
    ["Decide on cost, not on point predictions.", "Same forest: 14.2% when rounding up, 21.4% with forest-weighted expected cost."],
    ["Exact OPT is readable and certified, but gains are small.",
      "Certified training optimum in its tree class; +0.1 pt over greedy on test (not significant); cost-aware RF is 1.5 pts better."]];
  pts.forEach(([h, d], i) => {
    const y = 1.78 + i * 1.18;
    s.addShape(pres.shapes.OVAL, { x: 5.05, y: y + 0.03, w: 0.5, h: 0.5, fill: { color: BLUE }, line: { color: BLUE, width: 0 } });
    T(s, `${i + 1}`, { x: 5.05, y: y + 0.03, w: 0.5, h: 0.5, fontSize: 18, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    T(s, h, { x: 5.8, y, w: 6.95, h: 0.38, fontSize: 17, bold: true });
    T(s, d, { x: 5.8, y: y + 0.4, w: 6.95, h: 0.62, fontSize: 14, color: TEXT });
  });
  R(s, { x: 5.05, y: 5.4, w: 7.7, h: 1.25, fill: { color: CARD } });
  T(s, [{ text: "Limitation  ", options: { bold: true, color: NUM } },
    { text: "All savings come from an assumed cost (β = 5) applied to observed CPU traces. We do not show Azure billing savings, capacity consolidation, or application performance under enforced reservations." }],
  { x: 5.3, y: 5.5, w: 7.25, h: 1.05, fontSize: 14, valign: "middle" });
  notes(s, `[Emma | 0:40 | running 8:00]
Main message: Customer history and the right objective deliver about 20 percent modeled savings; exact OPT adds readability and a certificate, not much extra accuracy.

Script:
So, three takeaways. Customer history carries most of the signal. Deciding on cost beats rounding up predictions, even with the same model. And exact OPT gives a readable rule with a certified training optimum, but on test it only ties greedy, and the forest is about one and a half points better. One limitation: these are modeled savings on observed traces, not real billing or performance under enforced reservations. Overall, about 20 percent modeled savings, driven by customer history and the right objective. Thanks, we're happy to take questions.`);
}

// ================================================================ Backup slides
pres.addSection({ title: "Backup (not presented)" });
{
  const s = content("Backup", "Backup: the exact solver and what it certifies", "Backup (not presented)");
  // Equations are typeset in LaTeX (equations/eqN.tex) and rasterized at 600 dpi to assets/eq_N.png.
  // EQ_SCALE 1.55 makes 10pt LaTeX read at about 15.5pt on the slide.
  const EQ_SCALE = 1.55;
  const EQ_PX = { 1: [3146, 119], 2: [3088, 381], 3: [1883, 232], 4: [1598, 277] };
  const rows = [
    ["Tree class", 1, "8 creation-time features, 31 candidate thresholds each; every leaf takes one of the 4 tiers.", 0.95],
    ["Recursion", 2, null, 1.2],
    ["Why it is fast", 3, "Leaf cost depends on 4 sufficient statistics for every \u03b2. The 1.70M training VMs collapse to 46,673 patterns; 3-D histograms with 2-D prefix sums price all lower splits at once. Depth 3: ~5 s; depth 4: 28\u201336 min.", 1.35],
    ["What is certified", 4, "Global minimum of training cost in this class only: not other features, deeper trees, or test cost.", 1.15],
  ];
  let y = 1.6;
  rows.forEach(([k, n, cap, h]) => {
    card(s, 0.6, y, 12.13, h);
    T(s, k, { x: 0.95, y, w: 2.5, h, fontSize: 16, bold: true, color: NUM, valign: "middle" });
    const [pw, ph] = EQ_PX[n];
    const ew = pw / 600 * EQ_SCALE, eh = ph / 600 * EQ_SCALE;
    const capH = cap ? 0.5 : 0;
    const top = y + (h - eh - capH) / 2;
    s.addImage({ path: "assets/eq_" + n + ".png", x: 3.6, y: top, w: ew, h: eh, altText: k + " equation (LaTeX)" });
    if (cap) T(s, cap, { x: 3.6, y: top + eh + 0.04, w: 8.95, h: capH, fontSize: 13, color: MUTED, valign: "top" });
    y += h + 0.1;
  });
  notes(s, `[Backup, for questions]
If asked why the tree is "exact": the tree class is finite, so the dynamic program can evaluate every allowed split. That gives a certificate by enumeration, not an MIO bound. The certificate is about training cost in this class only. It says nothing about test performance, other features, or continuous thresholds.`);
}
{
  const s = content("Backup", "Backup: solver cross-check against Gurobi and local search", "Backup (not presented)");
  const hdr = ["Instance (β = 5)", "Exact DP (ours)", "Gurobi MIO, 600 s limit", "IAI local search", "Greedy tree"];
  const data = [
    ["Depth 2, 1,000 VMs", "0.003 s, optimal", "86 s, optimal (0% gap)", "3.8 s, = optimum", "+0.07%"],
    ["Depth 2, 5,000 VMs", "0.004 s, optimal", "212 s, optimal (0% gap)", "3.8 s, = optimum", "+0.07%"],
    ["Depth 3, 1,000 VMs", "0.07 s, optimal", "time limit (13.1% gap)", "3.9 s, +0.09%", "+0.49%"],
    ["Depth 3, 5,000 VMs", "0.07 s, optimal", "time limit (7.9% gap)", "4.0 s, = optimum", "+0.50%"],
  ];
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: BLUE }, fontFace: FONT, fontSize: 14 } });
  const Cc = (t, i) => ({ text: t, options: { fontFace: FONT, fontSize: 14, color: TEXT, fill: { color: i % 2 ? "FFFFFF" : CARD } } });
  s.addTable([hdr.map(H), ...data.map((r, i) => r.map(t => Cc(t, i)))],
    { x: 0.6, y: 1.8, w: 12.13, colW: [2.7, 2.2, 2.85, 2.3, 2.08], rowH: 0.55, border: { type: "solid", pt: 0.5, color: "D0D7DF" },
      valign: "middle", margin: [0, 0.12, 0, 0.12] });
  labelBox(s, 0.6, 5.0, 12.13, 0.68, "Reading:",
    "Percentages are training cost above the certified optimum. Our DP matches Gurobi's proven optimum at depth 2; Gurobi cannot close its gap at depth 3.", 1.95, 14);
  notes(s, `[Backup, for questions]
Subsamples use 8 bins per feature. The DP is several orders of magnitude faster than the MIO and still certifies optimality where Gurobi times out. Local search usually finds the optimum but has no certificate, and greedy is about half a percent worse on training cost at depth 3.`);
}
{
  const s = content("Backup", "Backup: savings across β and the reservation-shortfall trade-off", "Backup (not presented)");
  s.addImage({ path: "assets/fig_beta_risk.png", x: 0.85, y: 1.6, w: 11.6, h: 11.6 * 2.65 / 6.5,
    altText: "Savings across 12 beta values and mean reservation versus expected shortfall, from the final report Figure 3" });
  T(s, "Every policy re-tuned on validation at each β. OPT: 5.2% at β = 1.5, 20.1% at β = 5, 13.3% at β = 20; the peak partly reflects the fixed baseline switching from 75% to 100% near β = 5. For a true β between 4 and 8, the β = 5 tree costs under 1% more than a re-tuned tree.",
    { x: 0.6, y: 6.35, w: 12.13, h: 0.6, fontSize: 12, color: MUTED });
  notes(s, `[Backup, for questions]
Beta is an assumed economic parameter, so we never pick beta by comparing costs. Across beta from 3 to 10, OPT saves about 15 to 20 percent, and the cost-aware forest stays the best method. The red cross in panel b is OPT's tier mix assigned at random: it would cost 13.4 percent more than always reserving 100 percent, so the savings come from targeting, not from reserving less.`);
}
{
  const s = content("Backup", "Backup: customer heterogeneity and interpretable clustering", "Backup (not presented)");
  s.addImage({ path: "assets/fig_customers.png", x: 1.15, y: 1.55, w: 11.0, h: 11.0 * 2.3 / 6.5,
    altText: "Per-subscription savings distribution, savings by subscription size, and share of subscriptions worse off, from the final report Figure 4" });
  labelBox(s, 0.6, 5.62, 12.13, 0.55, "Customers:",
    "94 subscriptions with ≥1,000 test VMs deliver 85% of OPT's saving; 11.9% of subscriptions pay more than always-100% under OPT.", 1.95, 13);
  labelBox(s, 0.6, 6.27, 12.13, 0.55, "Clustering:",
    "k-means (K = 8) + OCT saves 19.7% at β = 5 vs. 20.1% for OPT (not significant); behind at β = 10 (13.0% vs. 15.3%).", 1.95, 13);
  notes(s, `[Backup, for questions]
The objective weights every VM equally, so very large customers dominate the aggregate. Weighting subscriptions equally reverses the tree ranking: greedy, the four-leaf tree and the cost-aware forest do better per subscription than the eight-leaf OPT. The clustering refinement from Homework 6 groups VMs by feature similarity, not cost, which is why it falls behind when beta is high.`);
}

pres.writeFile({ fileName: "VM_Reservations_OPT_final.pptx" }).then(f => console.log("wrote", f));

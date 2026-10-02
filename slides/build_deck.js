// Lightning talk deck for the ECE 382V initial project idea.
// Usage: node build_deck.js   (reads deck_data.json, writes Kapur_ECE382V_Initial_Idea.pptx)
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const path = require("path");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");
const D = require("./deck_data.json");

const THEME = {
  name: "Flaky Signal",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2430", lt1: "FFFFFF", dk2: "2E3A4B", lt2: "EEF1F4",
    accent1: "E8A33D", accent2: "2A9D8F", accent3: "C8553D", accent4: "6C7A89",
    accent5: "8FB8DE", accent6: "F4DDB0", hlink: "2A9D8F", folHlink: "6C7A89",
  },
};
const HEX = THEME.colors;
const OUT = "Kapur_ECE382V_Initial_Idea.pptx";

async function icon(Comp, hex, px = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size: px }));
  const buf = await sharp(Buffer.from(svg)).resize(px, px).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
const fmt = (n) => n.toLocaleString("en-US");

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
  pres.author = "Shaunak Kapur";
  pres.title = "Are LLM Written Unit Tests Flaky";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  // ---------- layouts ----------
  pres.defineSlideMaster({
    title: "DARK_TITLE",
    background: { color: C.text1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.35, w: 8.8, h: 1.3, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.8, w: 8.8, h: 0.9, fontSize: 18, color: C.accent6, valign: "top", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "LIGHT_CONTENT",
    background: { color: C.background1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontSize: 30, bold: true, color: C.text1, valign: "middle", margin: 0 }, text: "" } },
      { text: { text: "ECE 382V   Initial project idea", options: { x: 0.5, y: 5.2, w: 5, h: 0.3, fontSize: 10, color: C.accent4, margin: 0 } } },
    ],
    slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
  });
  pres.defineSlideMaster({
    title: "DARK_CONTENT",
    background: { color: C.text1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontSize: 30, bold: true, color: C.background1, valign: "middle", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
  });

  // run strip motif: a row of reruns, teal = pass, amber = the one flaky outcome
  function runStrip(slide, x, y, n, size, gap, flakyIdx, name) {
    for (let i = 0; i < n; i++) {
      slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: x + i * (size + gap), y, w: size, h: size, rectRadius: 0.04,
        fill: { color: i === flakyIdx ? C.accent1 : C.accent2 }, line: { type: "none" },
        objectName: `${name}_${i}`,
      });
    }
  }

  // ---------- slide 1 ----------
  pres.addSection({ title: "Idea" });
  let s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Idea" });
  s.addText("Are LLM Written Unit Tests Flaky?", { placeholder: "title" });
  s.addText([
    { text: "Initial project idea for ECE 382V", options: { breakLine: true } },
    { text: "Shaunak Kapur, working individually", options: { fontSize: 14, color: C.accent4 } },
  ], { placeholder: "body" });
  runStrip(s, 0.6, 4.35, 30, 0.2, 0.08, 22, "runstrip");
  s.addText("30 reruns of one test, one of them failed", { x: 0.6, y: 4.65, w: 8.5, h: 0.3, fontSize: 11, italic: true, color: C.accent4, margin: 0, isTextBox: true, objectName: "runstrip_caption" });
  s.addNotes("Hi, I am Shaunak and I am working on this project on my own. The question is simple. When an LLM writes your unit tests, are those tests flaky? This strip is what a flaky test looks like. Thirty runs, twenty nine pass, one fails, with no code change at all.");

  // ---------- slide 2 ----------
  s = pres.addSlide({ masterName: "LIGHT_CONTENT", sectionTitle: "Idea" });
  s.addText("Generation pipelines trust one passing run", { placeholder: "title" });
  const steps = ["LLM writes a test", "Run it once", "It passed", "Keep it forever"];
  const bx = 0.5, bw = 1.85, bg = 0.53, by = 1.45, bh = 0.95;
  for (let i = 0; i < steps.length; i++) {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: bx + i * (bw + bg), y: by, w: bw, h: bh, rectRadius: 0.12,
      fill: { color: i === 3 ? C.accent1 : C.background2 }, line: { type: "none" }, objectName: `step_${i}`,
    });
    s.addText(steps[i], { x: bx + i * (bw + bg), y: by, w: bw, h: bh, fontSize: 16, bold: true, align: "center", valign: "middle", color: C.text1, margin: 4, isTextBox: true, objectName: `step_text_${i}` });
    if (i < steps.length - 1) {
      s.addShape(pres.shapes.LINE, { x: bx + i * (bw + bg) + bw + 0.08, y: by + bh / 2, w: bg - 0.16, h: 0, line: { color: C.accent4, width: 2, endArrowType: "triangle" }, objectName: `arrow_${i}` });
    }
  }
  s.addText([
    { text: "TestPilot and most Copilot style pipelines keep a generated test if it passes one run.", options: { breakLine: true, bullet: true } },
    { text: "Every big flaky test study so far looks at tests written by people.", options: { breakLine: true, bullet: true } },
    { text: "So I want to know if LLM tests are more flaky, why, and if one run is enough to catch it.", options: { bullet: true } },
  ], { x: 0.5, y: 2.75, w: 5.6, h: 2.1, fontSize: 15, color: C.text1, paraSpaceAfter: 8, valign: "top", margin: 0, isTextBox: true, objectName: "why_bullets" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.45, y: 2.8, w: 3.05, h: 1.95, rectRadius: 0.12, fill: { color: C.text1 }, line: { type: "none" }, objectName: "callout_box" });
  runStrip(s, 6.7, 3.05, 8, 0.24, 0.08, 5, "mini_strip");
  s.addText("A flaky test can pass the one run that decides if it gets kept.", { x: 6.7, y: 3.45, w: 2.6, h: 1.15, fontSize: 14, italic: true, color: C.background1, valign: "top", margin: 0, isTextBox: true, objectName: "callout_text" });
  s.addNotes("Tools like TestPilot, and most Copilot style setups, generate a test, run it once, and keep it if it passes. The problem is that a flaky test can pass that one run. All of the big flaky test studies, like Luo et al. and the Python study by Gruber et al., look at tests written by people. Nobody has really measured this for LLM written tests yet.");

  // ---------- slide 3 ----------
  s = pres.addSlide({ masterName: "LIGHT_CONTENT", sectionTitle: "Idea" });
  s.addText("What I am running", { placeholder: "title" });
  const libs = ["cachetools", "schedule", "tinydb", "humanize", "toolz", "tabulate", "tenacity", "filelock", "boltons", "more-itertools", "arrow", "pybreaker"];
  s.addText("12 Python libraries", { x: 0.5, y: 1.2, w: 4.1, h: 0.35, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "libs_header" });
  libs.forEach((l, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.5 + col * 1.37, y = 1.65 + row * 0.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 1.27, h: 0.38, rectRadius: 0.08, fill: { color: i >= 6 ? C.accent6 : C.background2 }, line: { type: "none" }, objectName: `lib_${i}` });
    s.addText(l, { x, y, w: 1.27, h: 0.38, fontSize: 11, align: "center", valign: "middle", color: C.text1, margin: 0, isTextBox: true, objectName: `lib_text_${i}` });
  });
  s.addText("Tan ones were added because they use time, randomness, threads, or files.", { x: 0.5, y: 3.7, w: 4.0, h: 0.5, fontSize: 11, italic: true, color: C.accent4, margin: 0, isTextBox: true, objectName: "libs_note" });

  const conds = [
    [fa.FaUserEdit, "Human", "the library's own test suite"],
    [fa.FaSyncAlt, "Agentic Sonnet", "runs its tests and fixes failures"],
    [fa.FaPenNib, "One shot Sonnet", "writes tests, never runs them"],
    [fa.FaFeatherAlt, "One shot Haiku", "same, with a smaller model"],
  ];
  s.addText("4 test suites per library", { x: 5.0, y: 1.2, w: 4.5, h: 0.35, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "conds_header" });
  for (let i = 0; i < conds.length; i++) {
    const y = 1.65 + i * 0.52;
    s.addShape(pres.shapes.OVAL, { x: 5.0, y, w: 0.4, h: 0.4, fill: { color: C.text1 }, line: { type: "none" }, objectName: `cond_circle_${i}` });
    s.addImage({ data: await icon(conds[i][0], HEX.lt1), x: 5.09, y: y + 0.09, w: 0.22, h: 0.22, objectName: `cond_icon_${i}` });
    s.addText([{ text: conds[i][1], options: { bold: true, breakLine: true } }, { text: conds[i][2], options: { fontSize: 11, color: C.accent4 } }],
      { x: 5.55, y: y - 0.05, w: 3.95, h: 0.5, fontSize: 13, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: `cond_text_${i}` });
  }
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.0, y: 3.85, w: 4.5, h: 0.95, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: "perturb_box" });
  const pert = [[fa.FaRandom, "random order and hash seed"], [fa.FaGlobeAmericas, "10 time zones"], [fa.FaMicrochip, "CPU stress on every core"]];
  s.addText("Every suite reruns 70 times with", { x: 5.15, y: 3.92, w: 4.2, h: 0.3, fontSize: 12, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "perturb_header" });
  for (let i = 0; i < pert.length; i++) {
    const x = 5.15 + i * 1.45;
    s.addImage({ data: await icon(pert[i][0], HEX.accent1), x, y: 4.3, w: 0.28, h: 0.28, objectName: `pert_icon_${i}` });
    s.addText(pert[i][1], { x: x + 0.34, y: 4.22, w: 1.08, h: 0.68, fontSize: 10, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: `pert_text_${i}` });
  }
  s.addNotes("I picked twelve small Python libraries. The first six are a pilot and the other six were added because they deal with time, randomness, threads, or files, which is where flaky tests usually come from. For each library there are four suites. The library's own human tests, an agentic Sonnet setup that runs its tests and fixes failures, and two one shot setups that write the tests and never run them, one with Sonnet and one with Haiku. Every suite gets rerun seventy times while I shuffle test order and hash seed, switch between ten time zones, and load every CPU core.");

  // ---------- slide 4 ----------
  s = pres.addSlide({ masterName: "LIGHT_CONTENT", sectionTitle: "Idea" });
  s.addText("Early results", { placeholder: "title" });
  const stats = [
    [fmt(D.executions), `test executions across ${D.suites} suites`, C.text1],
    [String(D.llm_flaky), "LLM tests flipped, all about real time, and a one run filter kept all of them", C.accent1],
    [String(D.human_flaky), "human tests flipped, mostly because of test order", C.accent2],
  ];
  stats.forEach((st, i) => {
    const y = 1.15 + i * 1.3;
    s.addText(st[0], { x: 0.5, y, w: 3.4, h: 0.65, fontSize: 38, bold: true, fontFace: THEME.headFontFace, color: st[2], margin: 0, valign: "bottom", isTextBox: true, objectName: `stat_num_${i}` });
    s.addText(st[1], { x: 0.5, y: y + 0.68, w: 3.4, h: 0.55, fontSize: 12, color: C.accent4, margin: 0, valign: "top", isTextBox: true, objectName: `stat_label_${i}` });
  });
  const labels = Object.keys(D.smell_any_pct);
  const vals = labels.map((k) => D.smell_any_pct[k]);
  s.addChart(pres.charts.BAR, [{ name: "Tests with a risk pattern", labels, values: vals }], {
    x: 4.1, y: 1.15, w: 5.4, h: 3.85, barDir: "col",
    chartColors: [HEX.accent4, HEX.accent1, HEX.accent1, HEX.accent1],
    showTitle: true, title: "Tests with a flakiness risk pattern (%)", titleFontSize: 13, titleColor: HEX.dk1, titleFontFace: "+mj-lt",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0"%"', dataLabelFontSize: 12, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: HEX.accent4, catAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt",
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
    objectName: "smell_chart",
  });
  s.addNotes(`So far I have about ${fmt(D.executions)} test executions across ${D.suites} suites. Real flakiness is rare. ${D.human_flaky} human tests flipped and four of those are order dependent, where one test leaves something in a global registry. ${D.llm_flaky} LLM tests flipped, all from the one shot Haiku suites, and all three are about real time. One asserts on a random retry delay, one has a tight deadline that breaks under CPU load, and one builds a time two hours from now that crosses midnight in some time zones. All three passed the single run filter, so a real pipeline would have kept them. The chart shows the bigger difference. LLM suites have about twice as many tests with risky patterns like real sleeps, unfrozen clock reads, and unseeded randomness.`);

  // ---------- slide 5 ----------
  pres.addSection({ title: "Plan" });
  s = pres.addSlide({ masterName: "DARK_CONTENT", sectionTitle: "Plan" });
  s.addText("Next steps and open questions", { placeholder: "title" });
  const cols = [
    [fa.FaChartBar, "How I will evaluate", ["Flaky rate per suite with confidence intervals", "Root cause of every flaky test", "Risk patterns and coverage per suite", "What a one run filter keeps and throws away"]],
    [fa.FaCalendarAlt, "Timeline", ["Oct 9 proposal", "Oct 24 more models and reruns", "Nov 7 slower CI machine runs", "Nov 21 mitigation prompt", "Dec 12 final report"]],
    [fa.FaQuestion, "What I am unsure about", ["Flips are rare, so do I need 1000 reruns", "Do risk patterns predict real flips", "Is a laptop too fast to show timing bugs", "Ideas for other perturbations welcome"]],
  ];
  for (let i = 0; i < cols.length; i++) {
    const x = 0.5 + i * 3.08;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.25, w: 2.86, h: 3.4, rectRadius: 0.12, fill: { color: C.text2 }, line: { type: "none" }, objectName: `plan_card_${i}` });
    s.addShape(pres.shapes.OVAL, { x: x + 0.22, y: 1.45, w: 0.45, h: 0.45, fill: { color: C.accent1 }, line: { type: "none" }, objectName: `plan_circle_${i}` });
    s.addImage({ data: await icon(cols[i][0], HEX.dk1), x: x + 0.33, y: 1.56, w: 0.23, h: 0.23, objectName: `plan_icon_${i}` });
    s.addText(cols[i][1], { x: x + 0.22, y: 2.0, w: 2.45, h: 0.4, fontSize: 15, bold: true, color: C.background1, margin: 0, isTextBox: true, objectName: `plan_head_${i}` });
    s.addText(cols[i][2].map((t, j, a) => ({ text: t, options: { bullet: true, breakLine: j < a.length - 1 } })),
      { x: x + 0.22, y: 2.45, w: 2.5, h: 2.45, fontSize: 12, color: C.lt2 || C.background2, paraSpaceAfter: 6, valign: "top", margin: 0, isTextBox: true, objectName: `plan_body_${i}` });
  }
  s.addNotes("For evaluation I will report the flaky rate of each suite with confidence intervals, label the root cause of every flaky test using the Luo et al. categories, compare risk patterns and coverage, and measure what a one run filter keeps and throws away. The thing I am least sure about is the base rate. Flips are rare, so I might need far more reruns or a slower machine to see real differences. I would love ideas on other perturbations to try. Thanks.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", path.resolve(OUT));
})();

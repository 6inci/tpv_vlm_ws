// Offline check of the page logic (no browser): node experiments/human_evaluation/test_page_logic.js
// Extracts the inline script of index.html, runs it with the real data.js, and checks the rules of the page.
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm"), assert = require("assert");
const here = __dirname;
const sandbox = { window: {}, module: { exports: {} }, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(here, "data.js"), "utf8"), sandbox);
const html = fs.readFileSync(path.join(here, "index.html"), "utf8");
const script = html.split("<script>")[1].split("</script>")[0];
vm.runInContext(script, sandbox);
const P = sandbox.module.exports, D = sandbox.window.HE_DATA;

// data: 5 tasks, 10 conditions, 6 items, no AI preliminary cube labels in the data
assert.strictEqual(D.tasks.length, 5); assert.strictEqual(D.conditions.length, 10); assert.strictEqual(D.items.length, 6);
const raw = fs.readFileSync(path.join(here, "data.js"), "utf8");
// the AI-preliminary cube label boxes of the three photos (cube_candidate_funnel config) must not be in the page data
[[434, 417, 451, 432], [534, 279, 550, 294], [1607, 253, 1623, 271]].forEach(b => {
  assert.ok(!raw.includes(JSON.stringify(b).replace(/,/g, ", ")) && !raw.includes(JSON.stringify(b)), "AI preliminary label found in the page data: " + b);
});

// empty state: INCOMPLETE, nothing rated, nothing auto-filled
let S = P.emptyState(); P.setState(S);
let st = P.overallStatus(S);
assert.deepStrictEqual([st.gt_saved, st.items_rated, st.items_total, st.status], [0, 0, 60, "INCOMPLETE"]);
const ev = P.exportEval(S);
Object.values(ev.conditions).forEach(c => Object.values(c.ratings).forEach(r => assert.strictEqual(r.rating, null)));
assert.strictEqual(ev.status, "INCOMPLETE");
assert.ok(Object.values(ev.conditions).every(c => c.ai_reference_not_human && !("rating" in c.ai_reference_not_human)));
const csv = P.exportCSV(S).trim().split("\n");
assert.strictEqual(csv.length, 61);
assert.ok(csv.slice(1).every(l => l.split(",")[6] === ""), "no rating may be pre-filled in the CSV");

// ground-truth validation: empty reference is refused; boxes or explicit holds + relations + ambiguity + certainty are required
const g = P.emptyGT();
assert.ok(P.gtValidate(g).length >= 4);
g.source_boxes.push({ x0: 1, y0: 1, x1: 9, y1: 9, label: "", certainty: "CONFIRMED" }); g.destination_hold = true;
g.source_relation = "from"; g.destination_relation = "purpose_only"; g.ambiguity = "multiple_acceptable";
g.certainty = { source: "CONFIRMED", destination: "UNCERTAIN", relation: "CONFIRMED", rules: "UNCERTAIN" };
assert.strictEqual(P.gtValidate(g).length, 0, "complete reference must validate");

// one saved reference + one rating -> counted, still INCOMPLETE; ground-truth export keeps the human certainty
g.saved_at = "2026-11-19T00:00:00Z"; S.ground_truth.E1_T1 = g;
P.evalOf(S, "E1_T1_R0").items.source_accuracy.rating = "UNCERTAIN";
st = P.overallStatus(S);
assert.deepStrictEqual([st.gt_saved, st.items_rated, st.status], [1, 1, "INCOMPLETE"]);
const gt = P.exportGT(S);
assert.strictEqual(gt.tasks.E1_T1.certainty.destination, "UNCERTAIN"); assert.strictEqual(gt.tasks.E1_T2.saved_at, null); assert.strictEqual(gt.status, "INCOMPLETE");
assert.ok(P.exportCSV(S).includes("E1_T1_R0,E1_T1,R0,source_accuracy,출발 대상 정확성,UNCERTAIN"));
console.log("page logic checks passed");

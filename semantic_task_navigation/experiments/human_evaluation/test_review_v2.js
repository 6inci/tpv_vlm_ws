// Offline check of review_v2.html logic (no browser):  node experiments/human_evaluation/test_review_v2.js
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm"), assert = require("assert");
const here = __dirname;
const html = fs.readFileSync(path.join(here, "review_v2.html"), "utf8");
const script = html.split("<script>")[1].split("</script>")[0];
const sb = { window: {}, module: { exports: {} }, console };
vm.createContext(sb);
vm.runInContext(fs.readFileSync(path.join(here, "data.js"), "utf8"), sb);
vm.runInContext(script, sb);
const R = sb.module.exports, D = sb.window.HE_DATA;

// v1 storage is only ever read: the v1 key never appears in a write / remove call
assert.ok(/getItem\(V1_KEY\)/.test(script));
assert.ok(!/setItem\(V1_KEY|removeItem\(V1_KEY|localStorage\.clear\(/.test(script), "v2 must not write, remove or clear v1 storage");
assert.ok(/var KEY = "he_review_v2"/.test(script) && !/var KEY = "he_task_context_2d_navigation_v1"/.test(script));
// index.html (v1) is untouched by v2: it still has its own key
assert.ok(fs.readFileSync(path.join(here, "index.html"), "utf8").includes('var KEY = "he_task_context_2d_navigation_v1"'));

// empty state: nothing answered, nothing pre-filled
let S = R.emptyState(); R.setState(S);
const nPath = D.conditions.filter(c => R.hasPath(c)).length;
let p = R.progress(S);
assert.strictEqual(p.answers_total, 2 * D.conditions.length + nPath); assert.strictEqual(p.status, "INCOMPLETE"); assert.strictEqual(p.q1 + p.answers, 0);
const rows0 = R.summaryRows(S);
assert.ok(rows0.every(r => r.q2 === "" && r.q3 === "" && r.q4 === "" && r.source_agreement === "n/a"), "no answer may be pre-filled");
assert.strictEqual(R.exportCSV(S).trim().split("\n").length, D.conditions.length + 1);

// importing v1: the given v1 object is not modified; Q1 is pre-filled with its origin; other questions stay empty
const v1 = { schema: "human_evaluation_state_v1", evaluator: "kim", ground_truth: {
  E1_T1: { saved_at: "2026-11-19T10:00:00Z", ambiguity: "multiple_acceptable", comment: "트럭 또는 팔레트", source_boxes: [{ x0: 1, y0: 2, x1: 3, y1: 4 }], destination_boxes: [] },
  E1_T2: { saved_at: "2026-11-19T10:05:00Z", ambiguity: "instruction_ambiguous", comment: "", source_boxes: [], destination_boxes: [] },
  E1_T3: { saved_at: null, ambiguity: "unique" } } };
const before = JSON.stringify(v1);
assert.strictEqual(R.importV1(S, v1), 2);
assert.strictEqual(JSON.stringify(v1), before, "the v1 object must not be modified");
assert.strictEqual(S.tasks.E1_T1.q1.value, "valid"); assert.ok(S.tasks.E1_T1.q1.source.startsWith("imported_v1"));
assert.strictEqual(S.tasks.E1_T2.q1.value, "ambiguous"); assert.ok(!S.tasks.E1_T3 || !S.tasks.E1_T3.q1, "unsaved v1 references are not imported");
assert.strictEqual(S.tasks.E1_T1.q1_note, "트럭 또는 팔레트"); assert.strictEqual(S.evaluator, "kim");
assert.ok(Object.keys(S.conditions).length === 0, "Q2-Q4 are never pre-filled from v1");
assert.strictEqual(R.importV1(S, { schema: "something_else" }), 0);

// agreement rules (AI chose / held vs human)
assert.strictEqual(R.agreement("chose", "correct", ["correct"]), "agree");
assert.strictEqual(R.agreement("chose", "wrong", ["correct"]), "disagree");
assert.strictEqual(R.agreement("chose", "proper_hold", ["correct"]), "disagree");
assert.strictEqual(R.agreement("held", "proper_hold", ["correct"]), "agree");
assert.strictEqual(R.agreement("held", "wrong", ["correct"]), "disagree");
assert.strictEqual(R.agreement("held", "cannot_judge", ["correct"]), "n/a");

// flag: code says AUTONOMOUS but the human says the destination is inappropriate (E1_T3_R0 in the stored run)
S.conditions.E1_T3_R0 = { q3: { value: "inappropriate", source: "user_v2", at: "x" } };
const r = R.summaryRows(S).filter(x => x.cond_id === "E1_T3_R0")[0];
assert.ok(r.code_autonomous_but_user_inappropriate && r.destination_agreement === "disagree");
const ej = R.exportJSON(S).conditions.filter(x => x.cond_id === "E1_T3_R0")[0];
assert.strictEqual(ej.human.q3.value, "inappropriate"); assert.ok(ej.ai_reference_not_human.destination.startsWith("SELECTED"));

// plain-language AI explanations exist for every condition (ties and held destinations are explained, not hidden)
D.conditions.forEach(c => { assert.ok(R.sourceText(c).head && R.destText(c).head && R.routeText(c)); });
const tie = D.conditions.filter(c => c.source.status === "AMBIGUOUS")[0];
assert.ok(R.sourceText(tie).head.includes("동률"));
assert.ok(R.routeText(tie).includes("진단용"));
console.log("review v2 logic checks passed (" + D.conditions.length + " conditions, " + nPath + " with a path)");

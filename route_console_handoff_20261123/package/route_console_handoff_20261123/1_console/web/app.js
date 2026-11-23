"use strict";
// 경로 계획 콘솔 화면. 서버(server.py)가 계산한 값을 그리기만 한다. 좌표 계산은 모두 서버(Python)에서 한다.

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const SVGNS = "http://www.w3.org/2000/svg";
const COL = { A: "#e5484d", B: "#2f6fe0", 1: "#12b886", 2: "#8e4ec6", start: "#ffffff" };

const st = {
  cfg: null, mode: null, photo: null, livePhoto: null, liveOk: false, job: null, poll: null,
  result: null, view: "photo", layers: { zones: true, cands: false, route: true }, mapMeta: null, busy: false,
};

function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else if (v !== null && v !== undefined) e.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}
function s(tag, attrs = {}, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "-" : Number(v).toFixed(d));
const pts = (a) => a.map((p) => p.join(",")).join(" ");
async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `${r.status}`);
  return j;
}

// ---------------------------------------------------------------- 시작
async function init() {
  st.cfg = await api("/api/config");
  $("#choice-photo-meta").textContent = `사진 ${st.cfg.photos.length}장, ${st.cfg.photos_dir}`;
  $$(".choice").forEach((b) => b.addEventListener("click", () => enter(b.dataset.mode)));
  $$(".seg button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  $$(".layers input").forEach((c) => c.addEventListener("change", () => { st.layers[c.dataset.layer] = c.checked; draw(); }));
  buildThumbs();
  buildExamples();
  buildRail();
  for (const d of st.cfg.live_devices) $("#dev").append(h("option", { value: d }, `/dev/video${d}`));
  $("#dev").addEventListener("change", startStream);
  $("#capture").addEventListener("click", capture);
  $("#ask").addEventListener("submit", (e) => { e.preventDefault(); run(); });
  $("#instruction").addEventListener("input", updateRun);
  pollStatus();
  setInterval(pollStatus, 3000);
  // 주소로 바로 들어가기: ?mode=photo|live, ?photo=<파일>, ?job=<id> (지난 결과 다시 보기)
  const q = new URLSearchParams(location.search);
  if (q.get("mode")) {
    enter(q.get("mode"));
    if (q.get("photo")) selectPhoto(q.get("photo"));
    if (q.get("view")) setView(q.get("view"));
    if (q.get("job")) {
      const j = await api(`/api/job?id=${q.get("job")}`).catch(() => null);
      if (j) {
        if (j.source === "photo") selectPhoto(j.photo);
        $("#instruction").value = j.instruction;
        updateRun();
        renderRail(j);
        if (j.state === "done" || j.state === "error") { st.result = j.result; renderResult(j); draw(); }
      }
    }
  }
}

function enter(mode) {
  $("#start").hidden = true;
  $("#app").hidden = false;
  setMode(mode);
  if (mode === "photo" && !st.photo) selectPhoto(st.cfg.photos[0]);
  $("#instruction").focus();
}

function setMode(mode) {
  st.mode = mode;
  $$(".seg button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
  $("#src-photo").hidden = mode !== "photo";
  $("#src-live").hidden = mode !== "live";
  if (mode === "live") startStream(); else stopStream();
  resetRun();
  showPhoto(mode === "photo" ? st.photo : st.livePhoto);
  updateRun();
}

// ---------------------------------------------------------------- 사진
function buildThumbs() {
  $("#photo-count").textContent = `${st.cfg.photos.length}장`;
  const box = $("#thumbs");
  for (const n of st.cfg.photos) {
    const t = n.replace(/^\d{8}_(\d{2})(\d{2})(\d{2})_view\.png$/, "$1:$2:$3");
    const b = h("button", { class: "thumb", role: "option", "aria-selected": "false", title: n, onclick: () => selectPhoto(n) },
      h("img", { src: `/api/photo?name=${encodeURIComponent(n)}&w=320`, alt: "", loading: "lazy" }), h("span", {}, t));
    b.dataset.name = n;
    box.append(b);
  }
}
function selectPhoto(n) {
  st.photo = n;
  $$(".thumb").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.name === n)));
  resetRun();
  showPhoto(n);
  updateRun();
}
function showPhoto(n) {
  const img = $("#photo");
  if (n) img.src = `/api/photo?name=${encodeURIComponent(n)}&w=1920`; else img.removeAttribute("src");
  img.hidden = !n;
  $("#stage-empty").hidden = !!n;
  draw();
}

// ---------------------------------------------------------------- 실시간
function startStream() {
  $("#live-img").src = `/api/live/stream?dev=${$("#dev").value}&t=${Date.now()}`;
}
function stopStream() {
  if ($("#live-img").getAttribute("src")) { $("#live-img").removeAttribute("src"); api("/api/live/stop", {}).catch(() => {}); }
}
async function capture() {
  const msg = $("#capture-msg");
  msg.textContent = "찍는 중";
  try {
    const r = await api("/api/live/capture", { dev: Number($("#dev").value) });
    st.liveOk = !!r.ok;
    if (r.name) {
      st.livePhoto = r.name;
      $("#captured").hidden = false;
      $("#captured img").src = `/api/photo?name=${encodeURIComponent(r.name)}&w=480`;
      $("#captured span").textContent = r.name;
      showPhoto(r.name);
    }
    msg.textContent = r.ok ? "카메라 자세가 실험 사진과 같습니다. 이 장면으로 계획할 수 있습니다." : r.why;
  } catch (e) { msg.textContent = `찍지 못함: ${e.message}`; st.liveOk = false; }
  updateRun();
}

// ---------------------------------------------------------------- 지시와 실행
function buildExamples() {
  const box = $("#examples");
  for (const ex of st.cfg.examples) box.append(h("button", { type: "button", onclick: () => { $("#instruction").value = ex; updateRun(); $("#instruction").focus(); } }, ex));
}
function updateRun() {
  const ins = $("#instruction").value.trim();
  const ready = st.mode === "photo" ? !!st.photo : (!!st.livePhoto && st.liveOk);
  $("#run").disabled = st.busy || !ins || !ready;
  $("#run").title = !ready ? (st.mode === "photo" ? "사진을 고르세요" : "먼저 카메라 자세가 맞는 장면을 찍으세요") : (!ins ? "작업지시를 입력하세요" : "");
}
async function run() {
  if ($("#run").disabled) return;
  resetRun();
  st.busy = true; updateRun();
  $("#run").textContent = "만드는 중";
  try {
    const r = await api("/api/run", { source: st.mode, photo: st.mode === "photo" ? st.photo : st.livePhoto, instruction: $("#instruction").value.trim() });
    st.job = r.job;
    st.poll = setInterval(pollJob, 450);
  } catch (e) {
    st.busy = false; $("#run").textContent = "경로 만들기"; updateRun();
    $("#result").replaceChildren(h("div", { class: "holdbox" }, h("b", {}, "시작하지 못함"), e.message));
  }
}
async function pollJob() {
  let j;
  try { j = await api(`/api/job?id=${st.job}`); } catch { return; }
  renderRail(j);
  if (j.state === "done" || j.state === "error") {
    clearInterval(st.poll); st.poll = null; st.busy = false;
    $("#run").textContent = "경로 만들기"; updateRun();
    st.result = j.result; st.jobInfo = j;
    renderResult(j);
    draw();
  }
}
function resetRun() {
  if (st.poll) { clearInterval(st.poll); st.poll = null; }
  st.result = null; st.jobInfo = null;
  buildRail();
  $("#hold").hidden = true;
  $("#runmeta").textContent = "";
  $("#result").replaceChildren(h("p", { class: "empty" }, "지시를 입력하고 ", h("b", {}, "경로 만들기"), "를 누르면 여기에 경로와 경유점이 나옵니다."));
  draw();
}

// ---------------------------------------------------------------- 추론 과정
function buildRail() {
  const rail = $("#rail");
  rail.replaceChildren();
  st.cfg.stages.forEach((sg, i) => {
    rail.append(h("li", { class: `st ${sg.kind} waiting`, "data-key": sg.key },
      h("span", { class: "num", "aria-hidden": "true" }, String(i + 1)),
      h("div", { class: "st-title" }, sg.title, h("span", { class: "who" }, sg.who)),
      h("div", { class: "st-q" }, sg.question),
      h("div", { class: "st-sum" }, "대기"),
    ));
  });
}
const STATUS_KO = { waiting: "대기", running: "진행 중", done: "완료", hold: "보류", skipped: "건너뜀", failed: "오류" };
function renderRail(j) {
  const keys = st.cfg.stages.map((g) => g.key);
  const holdIdx = j.hold ? keys.indexOf(j.hold.stage) : -1;
  for (const li of $$("#rail .st")) {
    const key = li.dataset.key;
    const x = j.stages[key];
    const after = holdIdx >= 0 && keys.indexOf(key) >= holdIdx;      // 보류한 단계부터는 테이프를 칠하지 않음
    li.className = `st ${li.className.includes(" model") ? "model" : "rule"} ${x.status}${after ? " after-hold" : ""}`;
    const sum = li.querySelector(".st-sum");
    sum.replaceChildren();
    if (x.status === "hold") sum.append(h("span", { class: "badge hold" }, "보류"), " ");
    if (x.status === "skipped") sum.append(h("span", { class: "badge skip" }, "건너뜀"), " ");
    sum.append(x.summary || STATUS_KO[x.status]);
    li.querySelector(".progress")?.remove();
    li.querySelector("details")?.remove();
    if (x.status === "running") li.append(h("div", { class: "progress", role: "progressbar", "aria-label": "진행 중" }));
    const det = detailFor(key, x.detail);
    if (det) li.append(det);
  }
  const hb = $("#hold");
  if (j.hold) {
    hb.hidden = false;
    hb.replaceChildren(h("b", {}, "경로를 만들지 않았습니다"), j.hold.why, h("div", { class: "note" }, holdHint(j.hold.stage)));
  } else hb.hidden = true;
  if (j.state === "error") { hb.hidden = false; hb.replaceChildren(h("b", {}, "오류"), j.error || ""); }
  if (j.result) $("#runmeta").textContent = `${fmt(j.result.wall_s, 1)}초 걸림. 저장: ${j.result.export.dir}`;
}
function holdHint(stage) {
  return {
    interpret: "지시를 더 분명하게 바꿔 보세요. 예: 옮길 물체와 도착지를 함께 말하기.",
    decide: "사진에서 대상을 하나로 정하지 못했습니다. 다른 사진을 고르거나 지시를 바꿔 보세요.",
    plan: "정차 자리나 경로가 안전 여유를 만족하지 않습니다. 물체 위치를 바꾸거나 다른 목적지를 고르세요.",
  }[stage] || "";
}
function kv(obj) {
  const dl = h("dl", { class: "kv" });
  for (const [k, v] of Object.entries(obj)) if (v !== null && v !== undefined && v !== "") dl.append(h("dt", {}, k), h("dd", {}, String(v)));
  return dl;
}
function detailFor(key, d) {
  if (!d || !Object.keys(d).length) return null;
  const box = h("details", { class: "more" }, h("summary", {}, "자세히"));
  if (key === "interpret") {
    box.append(kv({ ...(d.summary || {}), "걸린 시간": `${d.seconds}초` }), h("pre", { class: "raw" }, d.raw || ""));
  } else if (key === "perceive") {
    box.append(kv({ "인식 자료": d.source, "DINO 후보": `${d.n_dino}개`, "예시와 비교": `${d.n_gated}개 (크기 10–90 px)` }));
  } else if (key === "decide") {
    for (const k of ["A", "B"]) {
      const x = d[k];
      if (!x) continue;
      const t = x.top || {};
      box.append(kv({
        [k === "A" ? "옮길 것 (A)" : "놓을 곳 (B)"]: x.status_ko, "근거": x.why,
        "정책 전 1위": t.id ? `${t.id} 점수 ${fmt(t.score, 3)}${t.failed_at ? `, ${t.failed_at}에서 탈락` : ", 통과"}` : (x.place ? "지도의 장소" : ""),
        "탈락 이유": (x.fails || []).join(" / ") || null,
      }));
    }
  } else if (key === "project") {
    const o = d.objects || {};
    const rows = {};
    for (const [k, v] of Object.entries(o)) rows[`${k} 지도 좌표`] = `(${fmt(v.map_xy_m[0], 3)}, ${fmt(v.map_xy_m[1], 3)}) m, 높이 ${v.z_top_m * 100} cm, 노란 선 밖 ${fmt(v.beyond_line_cm, 1)} cm`;
    if (d.start) rows["출발"] = d.start.status === "OK" ? `(${fmt(d.start.xy_m[0], 3)}, ${fmt(d.start.xy_m[1], 3)}) m, ${d.start.yaw_deg}°` : d.start.status;
    if (d.start && d.start.how) rows["출발 근거"] = d.start.how;
    box.append(kv(rows));
  } else if (key === "plan") {
    for (const [k, sp] of Object.entries(d.stops || {})) {
      if (!sp) continue;
      box.append(kv({
        [k === "A" ? "파지 정차" : "도착 정차"]: `(${fmt(sp.xy_m[0], 3)}, ${fmt(sp.xy_m[1], 3)}) m, ${sp.yaw_deg}°`,
        "몸체 여유": `${fmt(sp.body_clearance_cm, 1)} cm`,
        "그리퍼 끝 → 물체": sp.reach_gap_cm !== undefined ? `${fmt(sp.reach_gap_cm, 1)} cm 모자람, 옆 ${fmt(sp.lateral_offset_cm, 1)} cm` : (sp.place_name ? `${sp.place_name} 정차 자리${sp.snapped_cm ? `, ${sp.snapped_cm} cm 옮김` : ""}` : null),
        "그리퍼 방향": sp.facing_error_deg > 10 ? `구역을 ${Math.round(sp.facing_error_deg)}° 비껴 봄 (이 자리에서 되는 방향: ${sp.ok_headings.join("°, ")}°)` : null,
      }));
    }
  }
  return box;
}

// ---------------------------------------------------------------- 결과
function wpRows(rt) {
  const P = [rt.start_xy_m, ...rt.waypoints_xy_m, rt.goal_xy_m];
  const rot = rt.rotate_at && rt.rotate_at.startsWith("W") ? Number(rt.rotate_at.slice(1)) : null;
  return P.map((p, i) => {
    const last = i === P.length - 1;
    const lab = i === 0 ? "출발" : last ? "도착" : `W${i}`;
    const yaw = last || (rot !== null && i >= rot) ? rt.goal_yaw_deg : rt.start_yaw_deg;
    const turn = (rot !== null && i === rot) || (last && rt.rotate_at === "도착점");
    return { lab, x: p[0], y: p[1], yaw, turn };
  });
}
function renderResult(j) {
  const box = $("#result");
  box.replaceChildren();
  if (j.state === "error") { box.append(h("div", { class: "holdbox" }, h("b", {}, "계산 중 오류"), j.error)); return; }
  const r = j.result;
  const ok = r.status === "PATH_OK";
  const legs = r.legs.filter((L) => L.route);
  const tot = legs.reduce((a, L) => a + (L.check ? L.check.length_m : 0), 0);
  const file = (name, label) => h("a", { class: "btn", href: `/api/file?job=${j.id}&name=${encodeURIComponent(name)}` }, label);
  box.append(h("div", { class: "res-head" },
    h("h3", {}, ok ? "경로를 만들었습니다" : "경로를 만들지 않았습니다"),
    h("span", { class: "sub" }, ok ? `경로 ${legs.length}개, 총 ${Math.round(tot * 100)} cm` : "보류"),
    h("span", { class: "acts" }, ok ? file("route_package.zip", "경로 파일 받기") : null, ok ? file("waypoints.csv", "경유점 CSV") : null, file("overlay.jpg", "그림 받기")),
  ));
  if (!ok) box.append(h("div", { class: "holdbox" }, h("b", {}, "보류한 이유"), r.why || ""));
  const wrap = h("div", { class: "legs" });
  legs.forEach((L, i) => {
    const c = COL[i === 0 ? 1 : 2];
    const rows = wpRows(L.route);
    const tb = h("table", { class: "wps" }, h("thead", {}, h("tr", {}, h("th", {}, "점"), h("th", {}, "x (m)"), h("th", {}, "y (m)"), h("th", {}, "차체 방향"), h("th", {}, ""))),
      h("tbody", {}, rows.map((w) => h("tr", {}, h("td", {}, w.lab), h("td", {}, fmt(w.x, 3)), h("td", {}, fmt(w.y, 3)), h("td", {}, `${w.yaw}°`), h("td", { class: w.turn ? "rot" : "" }, w.turn ? "여기서 회전" : "")))));
    const ck = L.check || {};
    wrap.append(h("div", { class: "leg-card" }, h("div", { class: "bar", style: `background:${c}` }),
      h("div", {}, h("div", { class: "leg-title" }, `${L.from} → ${L.to}`),
        h("div", { class: "leg-stats" },
          h("span", {}, `${Math.round((ck.length_m || 0) * 100)} cm`), h("span", {}, `경유점 ${L.route.waypoints_xy_m.length}개`),
          h("span", {}, L.route.rotate_at ? `회전 ${L.route.rotate_at}` : "회전 없음"), h("span", {}, `몸체 여유 최소 ${fmt(ck.min_body_clearance_cm, 1)} cm`),
          L.strafe_longest_cm > 60 ? h("span", {}, `옆걸음 최장 ${Math.round(L.strafe_longest_cm)} cm`) : null),
        tb)));
  });
  box.append(wrap);
  for (const line of (r.summary || []).filter((x) => x.startsWith("주의"))) box.append(h("div", { class: "holdbox warn" }, line));
  if (ok) box.append(h("p", { class: "files" }, `팀원 경로 형식(start_xy_m, waypoints_xy_m, goal_xy_m, yaw, rotate_at)으로 저장했고 팀원 검사 코드로 다시 확인했습니다. 그리퍼 도달은 실측 전입니다. ${r.export.dir}`));
}

// ---------------------------------------------------------------- 그림 (사진 위 / 지도)
function setView(v) {
  st.view = v;
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.view === v)));
  $("#frame-photo").hidden = v !== "photo";
  $("#frame-map").hidden = v !== "map";
  draw();
}
function draw() {
  drawPhoto();
  if (st.view === "map") drawMap();
}
function tipOn(el, text) {
  el.addEventListener("mouseenter", (e) => { const t = $("#tip"); t.textContent = text; t.hidden = false; moveTip(e); });
  el.addEventListener("mousemove", moveTip);
  el.addEventListener("mouseleave", () => { $("#tip").hidden = true; });
}
function moveTip(e) {
  const r = $("#stage").getBoundingClientRect();
  const t = $("#tip");
  t.style.left = `${e.clientX - r.left}px`; t.style.top = `${e.clientY - r.top}px`;
}

function drawPhoto() {
  const sv = $("#ov");
  sv.replaceChildren();
  const r = st.result;
  if (!r || !r.overlay) return;
  const o = r.overlay;
  if (st.layers.zones) {
    const g = s("g");
    for (const z of o.zones) {
      g.append(s("polygon", { class: `z-poly${z.excluded ? " ex" : ""}`, points: pts(z.polygon_px) }));
      const c = centroid(z.polygon_px);
      g.append(s("text", { class: "z-name", x: c[0], y: c[1], "text-anchor": "middle" }, z.name));
    }
    g.append(s("polygon", { class: "drive", points: pts(o.drive_px) }));
    sv.append(g);
  }
  if (st.layers.cands) {
    const g = s("g");
    for (const [k, list] of Object.entries(o.candidates || {})) {
      list.forEach((c, i) => {
        const [x0, y0, x1, y1] = c.box;
        g.append(s("rect", { class: "cand", x: x0, y: y0, width: x1 - x0, height: y1 - y0 }));
        g.append(s("text", { class: "cand-t", x: x0, y: y1 + 18 + (k === "B" ? 18 : 0) }, `${k}${i + 1} ${c.score}${c.failed_at ? ` ×${c.failed_at}` : ""}`));
      });
    }
    sv.append(g);
  }
  // A, B
  for (const k of ["A", "B"]) {
    const t = o.targets[k];
    if (!t || !t.box) continue;
    const [x0, y0, x1, y1] = t.box;
    const p = 9;
    sv.append(s("rect", { class: "tgt", x: x0 - p, y: y0 - p, width: x1 - x0 + 2 * p, height: y1 - y0 + 2 * p, rx: 6, stroke: COL[k] }));
    sv.append(s("text", { class: "tgt-t", x: x0 - p, y: y0 - p - 10, fill: COL[k] }, k === "A" ? "A 옮길 것" : "B 놓을 곳"));
  }
  if (st.layers.route) {
    o.legs.forEach((L, i) => {
      const c = COL[i === 0 ? 1 : 2];
      const P = L.points_px;
      sv.append(s("polyline", { class: "halo", points: pts(P), "stroke-width": 16 }));
      sv.append(s("polyline", { class: "leg", points: pts(P), stroke: c, "stroke-width": 9 }));
      P.slice(1, -1).forEach((q, j) => {
        const m = L.points_m[j + 1];
        const rot = L.rotate_at === `W${j + 1}`;
        sv.append(s("circle", { class: "wp", cx: q[0], cy: q[1], r: rot ? 15 : 12, fill: rot ? "#f2c230" : c }));
        sv.append(s("text", { class: "wp-t", x: q[0] + 20, y: q[1] - 14 }, `${i + 1}-W${j + 1}${rot ? " 회전" : ""}`));
        const hit = s("circle", { class: "hit", cx: q[0], cy: q[1], r: 22 });
        tipOn(hit, `경로 ${i + 1} W${j + 1}: (${fmt(m[0], 3)}, ${fmt(m[1], 3)}) m${rot ? ", 여기서 회전" : ""}`);
        sv.append(hit);
      });
    });
    for (const [k, lab] of [["start", "출발"], ["A", "파지 정차"], ["B", "도착 정차"]]) {
      const p = o.poses[k];
      if (!p) continue;
      const c = k === "start" ? COL.start : COL[k === "A" ? 1 : 2];
      sv.append(s("polygon", { class: "body", points: pts(p.body_px), stroke: c }));
      sv.append(s("line", { class: "grip", x1: p.tip_px[0][0], y1: p.tip_px[0][1], x2: p.tip_px[1][0], y2: p.tip_px[1][1] }));
      const top = p.body_px.reduce((a, q) => (q[1] < a[1] ? q : a));
      sv.append(s("text", { class: "pose-t", x: centroid(p.body_px)[0], y: top[1] - 12, "text-anchor": "middle" }, `${lab} ${p.yaw_deg}°`));
    }
  }
}
function centroid(P) { const n = P.length; return [P.reduce((a, p) => a + p[0], 0) / n, P.reduce((a, p) => a + p[1], 0) / n]; }

async function drawMap() {
  const sv = $("#mapsvg");
  if (!st.mapMeta) st.mapMeta = await api("/api/map_meta");
  const M = st.mapMeta;
  const X = (x) => (x - M.ox) / M.res, Y = (y) => M.ny - (y - M.oy) / M.res;
  const mp = (a) => a.map((p) => [X(p[0]), Y(p[1])]);
  sv.setAttribute("viewBox", `-14 -6 ${M.nx + 20} ${M.ny + 18}`);
  sv.replaceChildren();
  sv.append(s("image", { href: "/api/map.png", x: 0, y: 0, width: M.nx, height: M.ny, preserveAspectRatio: "none" }));
  const g0 = s("g");
  for (let x = 0; x <= M.nx * M.res + 1e-6; x += 0.5) {
    g0.append(s("line", { class: "grid", x1: X(x), y1: 0, x2: X(x), y2: M.ny }));
    g0.append(s("text", { class: "grid-t", x: X(x), y: M.ny + 8, "text-anchor": "middle" }, `${x} m`));
  }
  for (let y = 0; y <= M.ny * M.res + 1e-6; y += 0.5) {
    g0.append(s("line", { class: "grid", x1: 0, y1: Y(y), x2: M.nx, y2: Y(y) }));
    g0.append(s("text", { class: "grid-t", x: -3, y: Y(y) + 1.5, "text-anchor": "end" }, `${y}`));
  }
  sv.append(g0);
  const r = st.result;
  const o = r && r.overlay;
  if (o && st.layers.zones) {
    for (const z of o.zones) {
      const P = mp(z.polygon_m);
      sv.append(s("polygon", { class: `z-poly${z.excluded ? " ex" : ""}`, points: pts(P) }));
      const c = centroid(P);
      sv.append(s("text", { class: "z-name", x: c[0], y: c[1], "text-anchor": "middle" }, z.name));
    }
  }
  if (o) sv.append(s("polygon", { class: "drive", points: pts(mp(o.drive_m)) }));
  if (!o) return;
  if (st.layers.route) {
    o.legs.forEach((L, i) => {
      const c = COL[i === 0 ? 1 : 2];
      const P = mp(L.points_m);
      sv.append(s("polyline", { class: "halo", points: pts(P), "stroke-width": 4 }));
      sv.append(s("polyline", { class: "leg", points: pts(P), stroke: c, "stroke-width": 2.2 }));
      P.slice(1, -1).forEach((q, j) => {
        const rot = L.rotate_at === `W${j + 1}`;
        sv.append(s("circle", { class: "wp", cx: q[0], cy: q[1], r: rot ? 3.2 : 2.6, fill: rot ? "#f2c230" : c }));
        sv.append(s("text", { class: "wp-t", x: q[0] + 4, y: q[1] - 3 }, `${i + 1}-W${j + 1}`));
        const hit = s("circle", { class: "hit", cx: q[0], cy: q[1], r: 6 });
        const m = L.points_m[j + 1];
        tipOn(hit, `경로 ${i + 1} W${j + 1}: (${fmt(m[0], 3)}, ${fmt(m[1], 3)}) m${rot ? ", 여기서 회전" : ""}`);
        sv.append(hit);
      });
    });
    for (const [k, lab] of [["start", "출발"], ["A", "파지"], ["B", "도착"]]) {
      const p = o.poses[k];
      if (!p) continue;
      const c = k === "start" ? "#16202a" : COL[k === "A" ? 1 : 2];
      sv.append(s("polygon", { class: "body", points: pts(mp(p.body_m)), stroke: c }));
      const T = mp(p.tip_m);
      sv.append(s("line", { class: "grip", x1: T[0][0], y1: T[0][1], x2: T[1][0], y2: T[1][1] }));
      const B = mp(p.body_m);
      sv.append(s("text", { class: "pose-t", x: centroid(B)[0], y: Math.min(...B.map((q) => q[1])) - 3, "text-anchor": "middle" }, lab));
    }
  }
  for (const k of ["A", "B"]) {
    const t = o.targets[k];
    if (!t || !t.map_xy_m) continue;
    const [x, y] = [X(t.map_xy_m[0]), Y(t.map_xy_m[1])];
    sv.append(s("circle", { cx: x, cy: y, r: 3.4, fill: COL[k], stroke: "#fff", "stroke-width": 1 }));
    sv.append(s("text", { class: "dot-t", x: x + 5, y: y + 2.5, fill: COL[k] }, k));
  }
}

// ---------------------------------------------------------------- 상태 표시
async function pollStatus() {
  let r;
  try { r = await api("/api/status"); } catch { return; }
  const p = $("#pill-model");
  const m = r.boot.status !== "준비됨" ? r.boot : r.model;
  p.className = "pill " + ({ 준비됨: "ok", 오류: "err" }[m.status] || "busy");
  p.querySelector("span").textContent = m.status === "준비됨" ? "모델 준비됨" : (m.status === "오류" ? "모델 오류" : `모델 ${m.status}`);
  p.title = m.detail || "";
  $("#pill-gpu").textContent = r.gpu ? `GPU ${(r.gpu.used_mib / 1024).toFixed(1)} / ${(r.gpu.total_mib / 1024).toFixed(0)} GB` : "GPU 확인 불가";
}

init();

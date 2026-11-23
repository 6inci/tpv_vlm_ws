#!/usr/bin/env python3
"""정답 찍는 도구 (브라우저). 사람이 새 사진마다 큐브·빈 팔레트 윗면 가운데를 클릭하고 시나리오를 고른 뒤 '확정'한다.

  $PY -B gt_tool.py        -> http://127.0.0.1:8766 를 브라우저로 연다 (이 컴퓨터에서만 열림)

- 저장: data/gt_holdout.json (사진마다 sha256, 점, 시나리오, 검토 완료)
- '정답 확정'을 누르면 이름·시각이 기록되고 그 뒤에는 고칠 수 없다. 시험 실행(run_holdout.py)은 확정된 정답만 받는다.
- 표준 라이브러리 http.server만 쓴다 (새 패키지 없음).
"""
import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import hold as H

PORT = 8766


def state():
    gt = H.load_gt() or {"photos": {}}
    files = H.photos()
    return {"photos": [{"name": p.name, "stem": p.stem, "sha256": H.sha256_file(p)} for p in files],
            "gt": gt, "locked": bool(gt.get("confirmed_by")) or bool(H.run_dirs()),
            "scenarios": {k: v["what"] for k, v in H.SCENARIOS.items()},
            "primary": list(H.CFG["scenarios"]["primary"]), "rule": H.CFG["ground_truth"]}


def save(body, confirm_name=None):
    cur = H.load_gt() or {"photos": {}}
    if cur.get("confirmed_by") or H.run_dirs():
        return 409, {"error": "정답이 이미 확정됐거나 실행이 있어 고칠 수 없습니다"}
    shas = {p.stem: H.sha256_file(p) for p in H.photos()}
    photos = {}
    for stem, e in body.get("photos", {}).items():
        if stem not in shas:
            continue
        photos[stem] = {"sha256": shas[stem], "scenario": e.get("scenario"), "reviewed": bool(e.get("reviewed")),
                        "cube": [[round(float(u), 1), round(float(v), 1)] for u, v in e.get("cube", [])],
                        "empty_pallet": [[round(float(u), 1), round(float(v), 1)] for u, v in e.get("empty_pallet", [])]}
    gt = {"experiment_id": H.CFG["experiment_id"], "point_rule": H.CFG["ground_truth"]["point"], "updated_at": datetime.now().isoformat(timespec="seconds"),
          "photos": photos}
    if confirm_name:
        gt["confirmed_by"], gt["confirmed_at"] = confirm_name, gt["updated_at"]
        probs = H.gt_problems(gt, H.photos())
        if probs:
            return 400, {"error": "확정할 수 없습니다", "problems": probs}
    H.GT_FILE.parent.mkdir(parents=True, exist_ok=True)
    H.GT_FILE.write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
    return 200, {"ok": True, "confirmed": bool(confirm_name), "problems": H.gt_problems(gt, H.photos())}


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>정답 찍기</title>
<style>
body{margin:0;font:15px/1.5 "Noto Sans CJK KR",sans-serif;background:#16181b;color:#e8e8e6;display:grid;grid-template-columns:270px 1fr;height:100vh}
aside{border-right:1px solid #33373c;overflow:auto;padding:10px}
aside button{display:block;width:100%;text-align:left;margin:2px 0;padding:6px 8px;background:#22262a;color:inherit;border:1px solid #33373c;border-radius:4px;cursor:pointer}
aside button.on{border-color:#e0b73a} .ok{color:#6cc08e} .todo{color:#e2a94b}
main{display:grid;grid-template-rows:auto 1fr;min-width:0}
header{padding:8px 12px;border-bottom:1px solid #33373c;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
header button,header select,header input{background:#22262a;color:inherit;border:1px solid #444a50;border-radius:4px;padding:5px 9px;font:inherit}
header button.mode.on{background:#e0b73a;color:#111}
#wrap{overflow:auto;position:relative} #stage{position:relative;display:inline-block}
#img{display:block} .fit #img{max-width:calc(100vw - 300px)}
.pt{position:absolute;width:14px;height:14px;margin:-9px 0 0 -9px;border:2px solid;border-radius:50%;pointer-events:none}
.pt.cube{border-color:#ff4a4a} .pt.empty_pallet{border-color:#4aa3ff}
#msg{padding:4px 12px;color:#e2a94b;font-size:14px}
</style></head><body>
<aside><div id="rule"></div><div id="list"></div></aside>
<main><header>
<button class="mode on" data-k="cube">큐브 찍기 (빨강)</button><button class="mode" data-k="empty_pallet">빈 팔레트 찍기 (파랑)</button>
<button id="undo">이 종류 마지막 점 지우기</button>
<select id="scen"></select><label><input type="checkbox" id="rev"> 이 사진 검토 완료</label>
<button id="fit">원본 크기</button><button id="save">저장</button>
<input id="who" placeholder="확정하는 사람 이름" size="12"><button id="confirm">정답 확정 (이후 수정 불가)</button>
</header><div id="msg"></div><div id="wrap" class="fit"><div id="stage"><img id="img"></div></div></main>
<script>
let S,cur=0,mode="cube",G={};
const $=s=>document.querySelector(s);
async function load(){S=await (await fetch("/api/state")).json();G=S.gt.photos||{};
 $("#rule").innerHTML="<b>찍는 법</b><br>"+S.rule.point+"<br>"+S.rule.empty_pallet+"<br>"+S.rule.none+"<hr>";
 $("#scen").innerHTML='<option value="">시나리오 고르기</option>'+Object.entries(S.scenarios).map(([k,v])=>`<option value="${k}">${k}${S.primary.includes(k)?"":" (보고만)"}: ${v}</option>`).join("");
 if(S.locked){document.querySelectorAll("header button,header input,header select").forEach(b=>{if(b.id!=="fit")b.disabled=true});msg("정답이 확정되어 고칠 수 없습니다 ("+(S.gt.confirmed_by||"실행 있음")+")")}
 if(!S.photos.length)msg("data/photos 에 새 사진(*.png)을 넣고 새로고침하세요");else show(0)}
function e(){const st=S.photos[cur].stem;return G[st]||(G[st]={cube:[],empty_pallet:[],scenario:"",reviewed:false})}
function list(){$("#list").innerHTML=S.photos.map((p,i)=>{const g=G[p.stem];const d=g&&g.reviewed&&g.scenario;return `<button class="${i===cur?"on":""}" onclick="show(${i})"><span class="${d?"ok":"todo"}">${d?"✓":"…"}</span> ${p.name}<br><small>큐브 ${g?g.cube.length:0} · 빈 팔레트 ${g?g.empty_pallet.length:0} · ${g&&g.scenario||"-"}</small></button>`}).join("")}
function show(i){cur=i;const x=e();$("#img").src="/photo/"+encodeURIComponent(S.photos[i].name);$("#scen").value=x.scenario||"";$("#rev").checked=!!x.reviewed;draw();list()}
function draw(){document.querySelectorAll(".pt").forEach(n=>n.remove());const im=$("#img");if(!im.naturalWidth)return;const k=im.clientWidth/im.naturalWidth;const x=e();
 for(const c of ["cube","empty_pallet"])for(const [u,v] of x[c]){const d=document.createElement("div");d.className="pt "+c;d.style.left=u*k+"px";d.style.top=v*k+"px";$("#stage").appendChild(d)}}
$("#img").onload=draw;window.onresize=draw;
$("#img").onclick=ev=>{if(S.locked)return;const im=$("#img"),r=im.getBoundingClientRect(),k=im.naturalWidth/im.clientWidth;e()[mode].push([(ev.clientX-r.left)*k,(ev.clientY-r.top)*k]);e().reviewed=false;$("#rev").checked=false;draw();list()};
document.querySelectorAll(".mode").forEach(b=>b.onclick=()=>{mode=b.dataset.k;document.querySelectorAll(".mode").forEach(x=>x.classList.toggle("on",x===b))});
$("#undo").onclick=()=>{e()[mode].pop();draw();list()};
$("#scen").onchange=()=>{e().scenario=$("#scen").value;list()};
$("#rev").onchange=()=>{e().reviewed=$("#rev").checked;list()};
$("#fit").onclick=()=>{$("#wrap").classList.toggle("fit");$("#fit").textContent=$("#wrap").classList.contains("fit")?"원본 크기":"화면에 맞춤";setTimeout(draw,0)};
async function post(url,b){const r=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)});return [r.status,await r.json()]}
$("#save").onclick=async()=>{const [s,j]=await post("/api/save",{photos:G});msg(s===200?"저장함. 남은 문제: "+(j.problems.length?j.problems.join(" / "):"없음"):j.error)};
$("#confirm").onclick=async()=>{const w=$("#who").value.trim();if(!w)return msg("확정하는 사람 이름을 적어 주세요");if(!confirm("확정하면 정답을 고칠 수 없습니다. 확정할까요?"))return;
 const [s,j]=await post("/api/confirm",{photos:G,name:w});if(s===200){msg("확정했습니다. 이제 run_holdout.py --check-only 로 확인하세요");load()}else msg(j.error+(j.problems?": "+j.problems.join(" / "):""))};
function msg(t){$("#msg").textContent=t}
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if self.path == "/api/state":
            return self._send(200, state())
        if self.path.startswith("/photo/"):
            p = H.PHOTOS / unquote(self.path[len("/photo/"):])
            if p.parent == H.PHOTOS and p.suffix == ".png" and p.exists():
                return self._send(200, p.read_bytes(), "image/png")
        self._send(404, {"error": "없음"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/api/save":
            return self._send(*save(body))
        if self.path == "/api/confirm":
            return self._send(*save(body, confirm_name=str(body.get("name", "")).strip() or None))
        self._send(404, {"error": "없음"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"http://127.0.0.1:{PORT}  (Ctrl+C로 끝)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

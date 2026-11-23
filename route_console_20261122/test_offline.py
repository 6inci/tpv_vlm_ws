#!/usr/bin/env python3
"""모델 없는 검사 (GPU 안 씀). 지시 해석만 가짜로 넣고 나머지 단계(저장된 인식 자료, 규칙, 좌표, 경로, 저장)를 실제로 돌린다.
  /home/binci/miniconda3/envs/stnav_gui/bin/python -B test_offline.py
결과 파일은 sessions/test_<시각>/ 아래에 생긴다 (runs/ 와 섞이지 않게).
"""
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine as EN  # noqa: E402

PHOTOS = Path(EN.CFG["sources"]["photos_dir"])


def slots(**kw):
    base = {"task": "transport", "object": "red cube", "object_near": "none", "start": "none", "destination_kind": "empty_pallet",
            "destination_place": "none", "words": ""}
    base.update(kw)
    return base


def test_parse():
    ok, err = EN.parse_slots('설명 {"task": "go_to", "object": "none", "object_near": "none", "start": "none", '
                             '"destination_kind": "place", "destination_place": "charging_station", "words": "충전소"} 끝')
    assert ok and not err and ok["destination_place"] == "charging_station"
    bad, err = EN.parse_slots('{"task": "fly", "object": "none"}')
    assert bad is None and "task" in err
    assert EN.parse_slots("없음")[0] is None
    # 해석이 서로 맞지 않으면 보류
    assert EN.task_plan(slots(object="none"))[0] is None
    assert EN.task_plan(slots(task="go_to"))[0] is None
    assert EN.task_plan(slots(destination_kind="place", destination_place="shelf"))[0] is None       # 선반에는 정차 자리 없음
    assert EN.task_plan(slots(object="other"))[0] is None
    p, why = EN.task_plan(slots(object_near="conveyor"))
    assert p and p["near_zone"] == "Z07"
    print("parse / task_plan OK")


def run_cases():
    eng = EN.Engine(progress=lambda m: None)
    out_root = HERE / "sessions" / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cases = [
        ("20261120_215815_view.png", "빨간 큐브를 비어 있는 팔레트로 옮겨줘.", slots(), "PATH_OK"),
        ("20261120_215815_view.png", "충전소로 이동해.", slots(task="go_to", object="none", destination_kind="place", destination_place="charging_station"), "PATH_OK"),
        ("20261120_215815_view.png", "빨간 큐브를 하역장으로 가져가.", slots(destination_kind="place", destination_place="truck_loading_bay"), None),
        ("20261120_215815_view.png", "하역장에서 출발해서 컨베이어로 가.", slots(task="go_to", object="none", start="truck_loading_bay",
                                                                         destination_kind="place", destination_place="conveyor"), None),
        ("20261120_215815_view.png", "충전소 옆 빨간 큐브를 빈 팔레트로.", slots(object_near="charging_station"), "HOLD"),
        ("20261120_215805_view.png", "빨간 큐브를 비어 있는 팔레트로 옮겨줘.", slots(), None),
        # 2026-11-22 사용자 실행의 실제 Qwen 해석: '하역장에서'가 object_near로 들어감 -> Python 보정으로 출발 = 하역장
        ("20261120_220632_view.png", "하역장에서 충전소 옆으로 이동시켜줘", slots(task="go_to", object="none", object_near="truck_loading_bay",
                                                                          destination_kind="place", destination_place="charging_station"), "PATH_OK"),
    ]
    res = []
    for photo, ins, sl, expect in cases:
        eng.interpret = lambda instruction, log, s=sl: {"prompt_id": "fake", "prompt": "", "raw": json.dumps(s, ensure_ascii=False), "slots": s, "error": None, "seconds": 0}
        events = []
        rec = eng.run(PHOTOS / photo, ins, False, lambda k, st, su, d: events.append((k, st, su)), out_root)
        st = rec["result"]["status"]
        res.append((photo[9:15], ins, st, rec["result"].get("why"), rec["export"]["teammate_check"]))
        print(f"{photo[9:15]} | {ins} -> {st} {rec['result'].get('why') or ''}")
        for e in events:
            if e[1] != "running":
                print("    ", e)
        for c in rec["export"]["teammate_check"]:
            assert c["bad_cm"] == 0, c
        if expect:
            assert st == expect, (ins, st, rec["result"])
        if "이동시켜줘" in ins:
            legs = [(L["from"], L["to"]) for L in rec["geometry"]["legs"]]
            assert legs == [("하역장", "충전소")], legs              # 출발이 지금 위치가 아니라 하역장이어야 함
    print("cases OK ->", out_root)
    return res


if __name__ == "__main__":
    test_parse()
    run_cases()

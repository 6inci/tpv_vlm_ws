#!/usr/bin/env python3
"""GPT에게 모델 구조 정리를 맡길 때 올릴 파일 묶음 (2026-11-22). 원본은 읽기만 하고 복사본만 만든다.
  /home/binci/miniconda3/envs/stnav_gui/bin/python -B make_bundle.py
결과: files/ (원래 폴더 구조 그대로 복사), 00_GPT_지시문.txt, 00_파일목록.txt, gpt_structure_bundle_20261122.zip
"""
import hashlib
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = Path("/home/binci/vlm_ws")
TB = "nl_to_route_20261122/vendor/teammate_extra_20261120"
RUN = "route_console_20261122/runs/20261122_200350_20261120_220632_view"

# (단계, vlm_ws 아래 경로, 설명)
FILES = [
    ("전체 흐름", "route_console_20261122/README.md", "다섯 단계 설명"),
    ("전체 흐름", "route_console_20261122/engine.py", "다섯 단계를 잇는 코드 (Engine.run에서 시작)"),
    ("전체 흐름", "route_console_20261122/config/console.yaml", "Qwen 프롬프트, 장소 표, 규칙 값"),
    ("② 장면 인식", "hybrid_exemplar_20261121/hybrid_pipeline.py", "DINO 후보 크기 조건, SigLIP2 예시 비교"),
    ("② 장면 인식", "hybrid_exemplar_20261121/config.yaml", "모델 이름, 기준값"),
    ("② 장면 인식", "context_policy_20261122/extract.py", "사진 한 장에서 후보와 점수 뽑기"),
    ("② 장면 인식 (DINO 세부)", "semantic_task_navigation/experiments/multimodel_target_viewer/mm_pipeline.py", "DINO 호출 (MMViewer.photo, 312-366행)"),
    ("② 장면 인식 (DINO 세부)", "semantic_task_navigation/experiments/multimodel_target_viewer/config.yaml", "추가 검출어, 모델 목록"),
    ("② 장면 인식 (DINO 세부)", "semantic_task_navigation/experiments/task_context_2d_navigation/config.yaml", "DINO 검출어와 기준값"),
    ("③ 대상 결정", "context_policy_20261122/policy.py", "작업 공간, 5S 구역, 받침, 점유, 1·2위 차이"),
    ("③ 대상 결정", "context_policy_20261122/common.py", "카메라, 구역, 작업 공간 띠"),
    ("③ 대상 결정", "context_policy_20261122/config/policy.yaml", "규칙 설정"),
    ("③ 대상 결정", "context_policy_20261122/data/dev/calibration.json", "판단 기준값"),
    ("③ 대상 결정", "context_policy_20261122/zones/zones_env2.json", "5S 구역 (사람이 그린 지도 정보)"),
    ("④ 좌표 변환", f"{TB}/1_required/pixel_to_map.py", "팀원 코드: 픽셀 -> 지도"),
    ("⑤ 경로 생성", "nl_to_route_20261122/nl_to_path.py", "정차 자세, 4방향 A*"),
    ("⑤ 경로 생성", "nl_to_route_20261122/config/nl_to_path.yaml", "경로 설정"),
    ("⑤ 경로 생성", "nl_to_route_20261122/POLICIES.md", "경로 규칙 설명"),
    ("⑤ 경로 생성", f"{TB}/2_optional/zones.json", "장소별 정차 자리 (팀원)"),
    ("⑤ 경로 생성", f"{TB}/2_optional/robot_params.json", "차체 크기 (팀원)"),
    ("⑤ 경로 생성", f"{TB}/2_optional/clearance_check.py", "경로 여유 검사 (팀원 코드)"),
    ("실행 예", f"{RUN}/record.json", "실제 실행 1건의 모든 중간값"),
    ("실행 예", f"{RUN}/overlay.jpg", "그 실행의 결과 그림"),
    ("실행 예", f"{RUN}/1_현재위치_큐브파지.json", "경로 파일 1 (팀원 형식)"),
    ("실행 예", f"{RUN}/2_큐브파지_팔레트배치.json", "경로 파일 2 (팀원 형식)"),
]
EXTRA = [("논문 그림", Path("/home/binci/Pictures/vlm_five_stages.pdf"), "figure/vlm_five_stages.pdf", "다섯 단계 이름")]

PROMPT = """첨부 파일은 자연어 지시 + 사진 한 장으로 로봇 경로(경유점)를 만드는 시스템의 코드와 설정이다.
파일 목록과 각 파일의 역할은 00_파일목록.txt에 있다.
engine.py의 Engine.run에서 시작해, 지시 입력부터 경로 생성까지의 모델 구조를 개조식 텍스트로 정리해라.

단계마다 다음을 적는다:
- 입력과 출력
- 담당: 학습된 모델인지, Python 규칙·계산인지
- 근거: 파일 이름과 함수 이름

반드시 지킬 것:
- 입력은 "자연어 지시"와 "사진" 두 가지로만 쓴다. "저장된 사진", "미리 찍은 사진", "미리 계산한 인식 자료(data/test, cached)", "실시간 카메라 모드"는 쓰지 않는다. 장면 인식은 "사진 → Grounding DINO → SigLIP2"로 쓴다.
- Qwen3-VL-4B는 이미지를 보지 않고 지시 글자만 정해진 항목(JSON)으로 바꾼다. 위치, 좌표, 박스는 만들지 않는다.
- 대상 결정, 좌표 변환, 경로 생성은 모델이 아니라 Python 규칙과 기하 계산이다.
- 5S 구역과 장소별 정차 자리는 사람이 정한 지도 정보다. 학습된 것이 아니다.
- task_plan의 해석 보정 규칙(옮길 물체 없는 이동에서 물체 옆 장소를 출발로 보는 것)도 적는다.
- 코드에서 확인하지 못한 내용은 추측하지 말고 "확인 안 됨"으로 표시한다.
"""


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = HERE / "files"
    if out.exists():
        raise SystemExit(f"{out} 이미 있음. 덮어쓰지 않음")
    rows = []
    for stage, rel, what in FILES:
        src = WS / rel
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        assert sha(src) == sha(dst)
        rows.append((stage, f"files/{rel}", what, str(src), sha(src)))
    for stage, src, rel, what in EXTRA:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        assert sha(src) == sha(dst)
        rows.append((stage, f"files/{rel}", what, str(src), sha(src)))

    (HERE / "00_GPT_지시문.txt").write_text(PROMPT, encoding="utf-8")
    lines = [f"모델 구조 정리용 파일 묶음 ({datetime.now().isoformat(timespec='seconds')} 생성). 모두 원본의 복사본.", ""]
    cur = None
    for stage, path, what, src, h in rows:
        if stage != cur:
            lines += ["", f"[{stage}]"]
            cur = stage
        lines.append(f"- {path}: {what}")
    lines += ["", "원본 경로와 sha256:"] + [f"- {src}  {h}" for _, _, _, src, h in rows]
    (HERE / "00_파일목록.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    z = HERE / "gpt_structure_bundle_20261122.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("00_GPT_지시문.txt", "00_파일목록.txt"):
            zf.write(HERE / name, name)
        for _, path, _, _, _ in rows:
            zf.write(HERE / path, path)
    with zipfile.ZipFile(z) as zf:
        bad = zf.testzip()
        n = len(zf.namelist())
    print(f"복사 {len(rows)}개, zip 항목 {n}개, 손상 {bad}, 크기 {z.stat().st_size / 1e6:.2f} MB -> {z}")


if __name__ == "__main__":
    main()

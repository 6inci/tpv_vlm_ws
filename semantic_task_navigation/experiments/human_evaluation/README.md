# Human Evaluation — Task-Conditioned Semantic Navigation

연구자가 `task_context_2d_navigation`의 저장된 AI 결과(실행 `results/20261119_174000/`)를 직접 평가하는 로컬 화면입니다. 새 모델 추론, 경로계획, 서버는 없습니다.

## 여는 방법

브라우저(Chrome 또는 Firefox) 주소창에 아래 경로를 넣거나, 파일 관리자에서 `index.html`을 더블클릭합니다.

```
file:///home/binci/vlm_ws/semantic_task_navigation/experiments/human_evaluation/index.html
```

또는 터미널에서:

```bash
xdg-open /home/binci/vlm_ws/semantic_task_navigation/experiments/human_evaluation/index.html
```

사진과 그림은 상대 경로로 읽습니다. 이 폴더와 `../task_context_2d_navigation/results/20261119_174000/`의 위치를 바꾸지 마세요.

## 평가 순서

1. **위쪽에 평가자 이름을 입력합니다.**
2. **화면 A · 정답 기준 입력** (작업지시 5개, R0/R1 공통)
   - 원본 사진과 작업지시만 보입니다. AI 결과와 AI 예비 라벨은 보이지 않습니다.
   - "출발 물체 그리기" 또는 "허용 목적지/배치 대상 그리기"를 고른 뒤 사진에서 드래그해 박스를 그립니다. 여러 개를 그릴 수 있습니다. 작은 물체는 확대(150–300 %) 후 그리세요.
   - 박스마다 설명과 확실성(확정/불확실)을 넣습니다. 정할 수 없으면 "판단 보류"를 체크합니다.
   - 공간 관계, 모호성, 적용할 운영 규칙, 항목별 확실성을 고르고 **"기준 저장"**을 누릅니다. 필수 항목이 비면 저장되지 않습니다.
3. **화면 B · AI 결과 평가** (조건 10개)
   - 해당 작업의 기준을 저장하기 전에는 "잠김"으로 표시되고 AI 결과가 나오지 않습니다.
   - 공개되는 내용: 원본 사진 위 AI 탐색 영역·기준 물체·선택 결과와 내 기준 박스, AI 해석과 구역 판단, 후보별 역할 표, AI에 실제로 입력된 번호 이미지, R0/R1 비교, 2D 지도(F2–F4), 원본 사진 위 경로(F5).
   - 6개 항목을 `SUCCESS / FAILURE / UNCERTAIN / NOT_EVALUATED` 중 하나로 고르고 의견을 적습니다. 기본값은 비어 있음(미평가)이며 자동으로 채워지지 않습니다.
   - "자동 검사 결과(코드 계산)"는 참고용입니다. 사용자 평가와 별도로 저장됩니다.
4. **화면 C · 종합 · 내보내기**
   - 진행 상태를 확인하고 결과 파일 3개를 내려받습니다.

## 저장과 내보내기

- 입력은 이 브라우저의 로컬 저장소에 **자동 저장**됩니다(같은 브라우저, 같은 경로로 다시 열면 이어서 평가).
- 브라우저 데이터를 지우면 사라집니다. 화면 C의 **"작업 상태 백업"**으로 수시로 백업하세요. **"백업 불러오기"**로 복원할 수 있습니다.
- 화면 C의 버튼이 아래 파일을 내려받기 폴더에 저장합니다. 연구 기록으로 남기려면 이 폴더의 `results/<날짜>/`로 옮기세요.

| 파일 | 내용 |
|---|---|
| `human_ground_truth.json` | 작업별 사람 기준: 박스(원본 px), 판단 보류, 관계, 모호성, 규칙, 항목별 확실성, AI 공개 후 수정 여부와 이전 값 |
| `human_evaluation.json` | 조건별 6개 항목 평가와 의견. AI·코드 결과는 `ai_reference_not_human` 블록에 따로 둠 |
| `human_evaluation.csv` | 조건 × 항목 60행 (평가자, 평가, 의견, 시각, AI 공개 후 수정 여부, 참고용 AI 분류) |
| `human_evaluation_state.json` | 작업 상태 백업 (복원용) |

모든 기준이 저장되고 60개 항목이 모두 선택될 때까지 `status`는 `INCOMPLETE`입니다.

## 평가 원칙 (화면에 구현됨)

- AI 결과는 해당 작업의 기준을 저장한 뒤에만 공개됩니다. 공개 후 기준을 고치면 `revised_after_reveal: true`와 이전 값이 남습니다.
- AI 예비 라벨(큐브 박스)은 데이터에 넣지 않았습니다. 테스트로 확인합니다.
- E1의 사용자 확인 사실("트럭 앞 콘 사이의 작은 회색 물체는 팔레트 모형")은 참고 정보로만 표시합니다. 모든 출하 명령의 정답 목적지로 쓰지 않습니다.
- 그리퍼 도달 가능성과 물리적 충돌 안전성은 사진만으로 확정하지 않습니다. 자동 검사와 사용자 판단은 분리해 저장합니다.

## 파일

| 파일 | 설명 |
|---|---|
| `index.html` | 평가 화면 (외부 라이브러리 없음, 오프라인 동작) |
| `data.js` | 저장된 AI 결과에서 만든 데이터 (`build_data.py`가 생성) |
| `assets/*.png` | E1·D1·P4 원본 사진 사본 (원본: `/home/binci/Pictures/Calibration/`) |
| `human_evaluation_template.json` | 빈 평가 양식 (모든 항목 미평가) |
| `build_data.py` | 저장된 실행 결과만 읽어 `data.js`·양식·사진 사본 생성 (모델·계획 없음) |
| `test_page_logic.js` | 페이지 로직 점검 (node) |
| `results/` | 내보낸 사람 평가 파일을 두는 곳 (현재 비어 있음 = 평가 미완료) |

다시 만들기와 점검:

```bash
cd /home/binci/vlm_ws/semantic_task_navigation
/home/binci/miniconda3/envs/stnav_gui/bin/python experiments/human_evaluation/build_data.py
node experiments/human_evaluation/test_page_logic.js
```

## 한계

- `data.js`에는 AI 결과가 들어 있습니다. 화면은 기준 저장 전에는 보여 주지 않지만, 파일을 직접 열면 볼 수 있습니다. 평가 순서를 지키는 것은 평가자의 몫입니다.
- 진행 상태는 브라우저마다 따로 저장됩니다. 다른 브라우저나 컴퓨터에서는 백업 파일로 옮겨야 합니다.
- 기준 박스는 사람이 드래그한 사각형이며, 물체 윤곽은 아닙니다.

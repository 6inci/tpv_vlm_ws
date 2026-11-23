# Next Step

## APPROVAL REQUIRED — Controlled Observed-Blocking Data Acquisition

STEP 30N은 기존 offline navigation chain을 두 개의 독립 evidence track으로 검증했다. Observed TRACK A는 navigation에 정상 진입했지만 `VALID / NO_PATH_EFFECT / CONTINUE_CURRENT_PATH`였고, development TRACK B만 block/persistence/replanning positive chain을 제공한다.

- 주요 남은 evidence gap은 active path와 실제로 교차하는 real observed obstacle과 그 `VALID → BLOCKED` 전환이다.
- 다음 단계가 승인되면 새 planner 구현이 아니라 별도의 controlled data-acquisition / observed-blocking experiment로 설계한다.
- `track_039`, D1 path 또는 world point를 이동해 positive evidence를 제조하지 않는다. D2 fixture도 observed evidence로 재분류하지 않는다.
- Live temporal persistence, ROS/Nav2, hardware motion과 physical safety는 계속 별도 미검증 범위다.

STEP23M physical measurement는 독립적인 optional branch로 계속 `WAITING_FOR_USER_MEASUREMENT / NOT_READY`다. Data acquisition은 자동 시작하지 않는다.

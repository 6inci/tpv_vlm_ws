# v0.1 동결 스냅샷 (2026-11-15)

- 이 폴더는 v0.2 수정 전 v0.1 코드·설정·테스트를 그대로 복사한 것이다. 수정하지 않는다.
- 코드 해시(src/*.py + run.py, config.code_fingerprint 방식): `b7ba777e8917c4eb…`
  - 대표 실행 `runs/20261115_225944_49f5/manifest.json`의 `code.sha256`과 같다.
- 파일 해시: `SHA256SUMS` (assets/ 복사 이전에 생성. assets는 원본 `assets/reference`의 사본)
- 독립 실행: `cd archive/v0.1_2026-11-15 && <stnav_gui python> -m pytest -q`
  - 결과는 이 폴더의 `runs/`에 저장된다.

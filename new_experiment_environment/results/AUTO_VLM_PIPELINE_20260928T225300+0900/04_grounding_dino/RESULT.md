# Grounding DINO Result

Status: PASS

실행되어 detection을 생성했다.

Attempts: [
  {
    "attempt": 1,
    "hypothesis": "local cache",
    "status": "BLOCKED",
    "error": "OSError(\"We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.\\nCheck your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.\")"
  },
  {
    "attempt": 2,
    "hypothesis": "download with installed Transformers; no package change",
    "status": "PASS"
  },
  {
    "attempt": 3,
    "hypothesis": "inference on cuda",
    "status": "PASS",
    "detections": 63
  }
]

# Read-only scene MCP prototype

Status: PARTIAL

`server/server.py` exposes five read-only JSON-RPC tools over newline-delimited stdio. It separates orchestration from Qwen/detector/SAM2/geometry computation and reads only this run's generated scene/map JSON. Local self-check and one-shot truck query were executed, but no external MCP client/conformance test or authentication is included; this is not a research-performance claim.

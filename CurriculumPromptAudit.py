"""
CurriculumPromptAudit.py
========================
Run from repo root (no GPU needed):

    python CurriculumPromptAudit.py

Does three things:
  1. Validates the full data → JoinConf → pack → format pipeline
     for every config key in NN_gen_curriculum.json
  2. Prints one rendered sample prompt per curriculum level
     so you can inspect quality before committing to training
  3. Reports per-level statistics: rows fetched, chunks formed,
     jaccard range, anchor selected

This is the fast iteration loop for Phase 3 prompt design.
No model loading, no GPU, no ChatBot call.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd

import ab.nn.api as lemur
from ab.nn.api import JoinConf


from ab.gpt.util.prompt.NNGenPromptCurriculum import NNGenPrompt as NNGenPromptCurriculum

# ── config ────────────────────────────────────────────────────────────────────
CURRICULUM_JSON = Path("ab/gpt/conf/prompt/train/NN_gen_Curriculum.json")
MAX_ROWS        = 200      # cap for audit — keeps it fast
PROMPT_PREVIEW  = 1200     # max chars to print per rendered prompt
# ─────────────────────────────────────────────────────────────────────────────


#sql_conf = NNGenPromptCurriculum._build_sql_conf(cfg)
#packed   = NNGenPromptCurriculum._pack_k_models(chunk, k)

def _build_chunks(data: pd.DataFrame, k: int) -> list[list[pd.Series]]:
    """
    Group by anchor_nn, take top-k per group sorted by accuracy desc.
    Falls back to sliding window if anchor_nn column absent.
    """
    if "anchor_nn" not in data.columns:
        # fallback: sliding window (wide mode / no anchor)
        rows = [pd.Series(r._asdict()) for r in data.itertuples(index=False)]
        return [rows[i:i + k] for i in range(0, len(rows) - k + 1, k)]

    sort_cols  = ["anchor_nn"]
    ascending  = [True]
    for col, asc in [("accuracy", False), ("anchor_jaccard", False), ("nn", True)]:
        if col in data.columns:
            sort_cols.append(col)
            ascending.append(asc)

    df = data.sort_values(sort_cols, ascending=ascending)
    chunks = []
    for _, g in df.groupby("anchor_nn"):
        if len(g) >= k:
            chunks.append([pd.Series(r) for r in g.head(k).to_dict(orient="records")])
    return chunks


def _render_prompt(packed: dict, cfg: dict) -> str:
    prompt_template = "\n".join(cfg["prompt"])
    para = {it["para"]: packed.get(it["value"], packed.get(it["para"], "")) for it in cfg["input_list"]}
    try:
        return prompt_template.format(**para)
    except KeyError as e:
        return f"[RENDER ERROR] missing key {e}\nAvailable: {sorted(para.keys())}"


def _separator(title: str) -> None:
    print(f"\n{'═' * 65}")
    print(f"  {title}")
    print(f"{'═' * 65}")


def _section(title: str) -> None:
    print(f"\n  ── {title} {'─' * (55 - len(title))}")


# ── per-key audit ─────────────────────────────────────────────────────────────

def audit_key(key: str, cfg: dict) -> dict:
    """
    Run the full pipeline for one config key.
    Returns a summary dict.
    """
    _separator(f"KEY: {key}")

    band     = cfg.get("similarity_band", "n/a")
    k        = int(cfg.get("num_joint_nns") or 1)
    sel_mode = cfg.get("selection_mode", "wide")
    is_gen   = cfg.get("is_generation", False)

    print(f"  band={band}  k={k}  mode={sel_mode}  is_generation={is_gen}")

    # 1. fetch data
    _section("Data fetch")
    sql_conf = NNGenPromptCurriculum._build_sql_conf(cfg)
    data = lemur.data(
        only_best_accuracy=True,
        task=cfg.get("task"),
        dataset=cfg.get("dataset"),
        metric=cfg.get("metric"),
        sql=sql_conf,
        max_rows=MAX_ROWS,
    )
    print(f"  rows fetched : {len(data)}")
    if data.empty:
        print("  WARN: no data returned — skipping")
        return {"key": key, "rows": 0, "chunks": 0, "status": "NO_DATA"}

    print(f"  columns      : {sorted(data.columns.tolist())}")

    # anchor resolution
    if "anchor_nn" in data.columns:
        anchors = data["anchor_nn"].dropna().unique()
        print(f"  anchors      : {len(anchors)}  ({list(anchors[:3])}{'...' if len(anchors) > 3 else ''})")
    else:
        print("  anchor_nn    : not in columns (wide mode or no anchor)")

    # jaccard range
    if "anchor_jaccard" in data.columns:
        jmin = data["anchor_jaccard"].min()
        jmax = data["anchor_jaccard"].max()
        jmean= data["anchor_jaccard"].mean()
        print(f"  jaccard      : min={jmin:.4f}  max={jmax:.4f}  mean={jmean:.4f}")
    else:
        print("  jaccard      : not in columns")

    # nn_code coverage
    if "nn_code" in data.columns:
        nn_code_ok = data["nn_code"].apply(lambda x: isinstance(x, str) and bool(x.strip()))
        print(f"  nn_code      : {nn_code_ok.sum()}/{len(data)} rows have valid code")
    else:
        print("  nn_code      : NOT IN COLUMNS — prompts will fail")

    # 2. build chunks
    _section("Chunk formation")
    if sel_mode == "tall" and k > 1:
        chunks = _build_chunks(data, k)
    else:
        rows   = [pd.Series(r._asdict()) for r in data.itertuples(index=False)]
        chunks = [[row] for row in rows]

    print(f"  chunks formed: {len(chunks)}  (need >= 1)")
    if not chunks:
        print("  WARN: 0 chunks — check band coverage and nn_minhash population")
        return {"key": key, "rows": len(data), "chunks": 0, "status": "NO_CHUNKS"}

    # 3. pack first chunk
    _section("Packing (first chunk)")
    first_chunk = chunks[0]
    try:
        packed = NNGenPromptCurriculum._pack_k_models(first_chunk, k,cfg)
        pack_keys = [k for k in sorted(packed.keys()) if not k.startswith("nn_")]
        print(f"  packed keys  : {pack_keys}")
        print(f"  anchor       : {packed.get('anchor_nn', 'n/a')}")
        print(f"  jaccard range: {packed.get('anchor_jaccard_min', 'n/a'):.4f} – {packed.get('anchor_jaccard_max', 'n/a'):.4f}")
        for i in range(1, k + 1):
            acc = packed.get(f"acc_{i}", "?")
            hp  = packed.get(f"hp_{i}", "")
            print(f"  model_{i}      : acc={acc}  hp={str(hp)[:60]}...")
    except ValueError as e:
        print(f"  PACK ERROR: {e}")
        return {"key": key, "rows": len(data), "chunks": len(chunks), "status": f"PACK_ERROR: {e}"}

    # 4. render prompt
    _section("Rendered prompt (first chunk)")
    rendered = _render_prompt(packed, cfg)
    preview  = rendered[:PROMPT_PREVIEW]
    if len(rendered) > PROMPT_PREVIEW:
        preview += f"\n  ... [{len(rendered) - PROMPT_PREVIEW} chars truncated] ..."
    for line in preview.splitlines():
        print(f"  {line}")

    # 5. token estimate (rough: 1 token ≈ 4 chars)
    _section("Token estimate")
    est_tokens = len(rendered) // 4
    print(f"  ~{est_tokens} tokens  ({len(rendered)} chars)")
    if est_tokens > 8000:
        print("  WARN: prompt may exceed context window for smaller models")

    print(f"\n  ✓ KEY '{key}' PASSED")
    return {
        "key":     key,
        "band":    band,
        "k":       k,
        "rows":    len(data),
        "chunks":  len(chunks),
        "tokens":  est_tokens,
        "status":  "OK",
    }


# ── wide mode audit ───────────────────────────────────────────────────────────

def audit_wide_key(key: str, cfg: dict) -> dict:
    _separator(f"KEY: {key}  [WIDE MODE]")
    k = int(cfg.get("num_joint_nns") or 1)
    sql_conf = NNGenPromptCurriculum._build_sql_conf(cfg)
    data = lemur.data(
        only_best_accuracy=True,
        task=cfg.get("task"),
        dataset=cfg.get("dataset"),
        metric=cfg.get("metric"),
        sql=sql_conf,
        max_rows=MAX_ROWS,
    )
    print(f"  rows fetched : {len(data)}")
    if data.empty:
        return {"key": key, "rows": 0, "chunks": 0, "status": "NO_DATA"}

    # check prm_id guard
    input_spec = cfg.get("input_list", [])
    prm_id_mapped = [it for it in input_spec if it.get("value") == "prm_id"]
    if prm_id_mapped:
        print(f"  ERROR: prm_id still mapped in input_list: {prm_id_mapped}")
        return {"key": key, "rows": len(data), "chunks": 0, "status": "PRMid_MAPPED"}
    print("  prm_id guard : OK (no prm_id in input_list)")

    # render one row
    row = pd.Series(data.iloc[0])
    para = {}
    missing = []
    for it in input_spec:
        val = row.get(it["value"])
        if val is None:
            missing.append(it["value"])
        para[it["para"]] = val

    if missing:
        print(f"  WARN missing columns: {missing}")

    prompt_template = "\n".join(cfg["prompt"])
    try:
        rendered = prompt_template.format(**para)
        preview  = rendered[:PROMPT_PREVIEW]
        if len(rendered) > PROMPT_PREVIEW:
            preview += f"\n  ... [{len(rendered) - PROMPT_PREVIEW} chars truncated] ..."
        print(f"\n  ── Rendered prompt (first row) {'─' * 30}")
        for line in preview.splitlines():
            print(f"  {line}")
        est_tokens = len(rendered) // 4
        print(f"\n  ~{est_tokens} tokens")
        print(f"\n  ✓ KEY '{key}' PASSED")
        return {"key": key, "rows": len(data), "chunks": len(data), "tokens": est_tokens, "status": "OK"}
    except KeyError as e:
        print(f"  RENDER ERROR: missing key {e}")
        return {"key": key, "rows": len(data), "status": f"RENDER_ERROR: {e}"}


# ── summary ───────────────────────────────────────────────────────────────────

def _print_summary(results: list[dict]) -> None:
    _separator("AUDIT SUMMARY")
    col_w = [30, 8, 4, 8, 8, 8, 10]
    headers = ["key", "band", "k", "rows", "chunks", "tokens", "status"]
    header_line = "  " + "".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(header_line)
    print("  " + "─" * sum(col_w))
    for r in results:
        row_line = "  " + "".join(
            str(r.get(h, "")).ljust(w) for h, w in zip(headers, col_w)
        )
        print(row_line)
    passed = sum(1 for r in results if r.get("status") == "OK")
    print(f"\n  Passed: {passed} / {len(results)}")
    if passed < len(results):
        print("\n  FAILED keys:")
        for r in results:
            if r.get("status") != "OK":
                print(f"    {r['key']} — {r['status']}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("CurriculumPromptAudit — Phase 3 pipeline validation")
    print(f"Config: {CURRICULUM_JSON}")
    print(f"Max rows per key: {MAX_ROWS}")

    with open(CURRICULUM_JSON) as f:
        prompt_cfg = json.load(f)

    results = []
    for key, cfg in prompt_cfg.items():
        sel_mode = cfg.get("selection_mode", "wide")
        try:
            if sel_mode == "tall":
                result = audit_key(key, cfg)
            else:
                result = audit_wide_key(key, cfg)
        except Exception as e:
            print(f"\n  UNEXPECTED ERROR in key='{key}': {e}")
            result = {"key": key, "status": f"EXCEPTION: {e}"}
        results.append(result)

    _print_summary(results)


if __name__ == "__main__":
    main()
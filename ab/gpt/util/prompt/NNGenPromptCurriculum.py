import json
import time
from typing import List, Dict

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from overrides import override
from transformers import PreTrainedTokenizerBase

import ab.nn.api as lemur
from ab.nn.api import JoinConf
from ab.gpt.util.prompt.Prompt import Prompt

from typing import Optional


class NNGenPrompt(Prompt):
    """
    Prompt generator for NN curriculum learning.

    Supports:
    - selection_mode = "wide"  (legacy pairwise / wide join)
    - selection_mode = "tall"  (SQL variable-N + Python packing)
    """

    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, prompts_path: str):
        super().__init__(max_len, tokenizer)
        self.prompts_path = prompts_path

    # ------------------------------------------------------------------
    # Packing logic 
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_k_models(
            rows: List[pd.Series],
            k: int,
            cfg: Optional[dict],
    ) -> Dict[str, object]:
        packed = {}

        for i, row in enumerate(rows, start=1):
            # --- strict guards: fail loud, never silently pass None ---
            nn_code = row.get("nn_code")
            if not isinstance(nn_code, str) or not nn_code.strip():
                raise ValueError(
                    f"nn_code missing or empty for nn='{row.get('nn')}' "
                    f"at position {i}. "
                    f"Ensure lemur.data() work table includes nn_code."
                )

            prm = row.get("prm")
            if not isinstance(prm, dict):
                raise ValueError(
                    f"prm is not a dict for nn='{row.get('nn')}' at position {i}. "
                    f"Got type={type(prm)}. "
                    f"fill_hyper_prm() must have run before packing."
                )

            transform_code = row.get("transform_code")
            if not isinstance(transform_code, str) or not transform_code.strip():
                raise ValueError(
                    f"transform_code missing for nn='{row.get('nn')}' at position {i}."
                )
            truncate = cfg.get("nn_code_truncate") if cfg else None
            nn_code_packed = (
                nn_code[:truncate] + "\n# ... [truncated]"
                if truncate and len(nn_code) > truncate
                else nn_code
            )

            # --- contract: hp_i is always the dict, never the UUID ---
            packed[f"acc_{i}"] = row["accuracy"]
            packed[f"hp_{i}"] = json.dumps(prm, sort_keys=True, separators=(",", ":"), ensure_ascii=False)  # dict , eg: {"lr": 0.001, "batch_size": 128, "optimizer": "Adam", "weight_decay": 0.005}
            packed[f"tr_{i}"] = transform_code
            packed[f"nn_{i}"] = nn_code_packed

        # shared metadata — identical across all rows in a chunk by construction
        for key in ("dataset", "task", "metric", "epoch"):
            val = rows[0].get(key)
            if val is not None:
                packed[key] = val
        packed["anchor_nn"] = rows[0].get("anchor_nn")
        packed["anchor_jaccard_min"] = min(r.get("anchor_jaccard", 0.0) for r in rows)
        packed["anchor_jaccard_max"] = max(r.get("anchor_jaccard", 0.0) for r in rows)

        return packed

    # ------------------------------------------------------------------
    # SQL config builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_sql_conf( cfg: dict) -> JoinConf | None:
        n = int(cfg.get("num_joint_nns") or 1)
        if n < 2:
            return None

        anchor_strategy = cfg.get("anchor_strategy", "auto")
        anchor_nn = cfg.get("anchor_nn") if anchor_strategy == "fixed" else None

        return JoinConf(
            num_joint_nns=n,
            same_columns=tuple(cfg.get("keep_same") or ()),
            diff_columns=tuple(cfg.get("no_repeat") or ()),
            enhance_nn=cfg.get("improve"),
            task=cfg.get("task"),  # NEW — required for _resolve_anchor
            dataset=cfg.get("dataset"),  # NEW — required for _resolve_anchor
            metric=cfg.get("metric"),  # NEW — required for _resolve_anchor
            similarity_mode=cfg.get("similarity_mode", "none"),
            similarity_band=cfg.get("similarity_band"),
            anchor_nn=anchor_nn,  # None triggers auto-resolve
        )

    # ------------------------------------------------------------------
    # get_raw_dataset — is_generation branch
    # For generation configs (is_generation=true), there is no ground
    # truth output yet. Only the instruction side is written.
    # The output column is left empty — it will be filled by the LLM
    # at inference time.
    # ------------------------------------------------------------------
    @override
    def get_raw_dataset(self, only_best_accuracy, n_training_prompts=None) -> DataFrame:
        prompt_frames = []

        with open(self.prompts_path) as f:
            prompt_cfg = json.load(f)

        for key, cfg in prompt_cfg.items():
            print(f"[NNGenPrompt] Preparing key='{key}'", flush=True)

            df_out = DataFrame(columns=["instruction", "context", "response", "category", "text"])
            prompt_frames.append(df_out)

            is_generation = cfg.get("is_generation", False)
            selection_mode = cfg.get("selection_mode", "wide")
            k = int(cfg.get("num_joint_nns") or 1)
            sql_conf = NNGenPrompt._build_sql_conf(cfg)

            t0 = time.time()
            data = lemur.data(
                only_best_accuracy=only_best_accuracy,
                task=cfg.get("task"),
                dataset=cfg.get("dataset"),
                metric=cfg.get("metric"),
                nn_prefixes=tuple(cfg.get("nn_prefixes") or ()),
                max_rows=n_training_prompts,
                sql=sql_conf,
            )
            print(f"[NNGenPrompt] fetched rows={len(data)} in {time.time() - t0:.1f}s")

            prompt_template = "\n".join(cfg["prompt"])
            output_template = "\n".join(cfg["output"]) if not is_generation else None
            input_spec = cfg["input_list"]

            # ----------------------------------------------------------
            # WIDE MODE
            # ----------------------------------------------------------
            if selection_mode == "wide":
                for _, row in tqdm(data.iterrows(), total=len(data)):
                    # --- enforce column contract for wide mode ---
                    # prm_id is internal only; prm (dict) is what goes in prompt
                    if "prm_id" in [it["value"] for it in input_spec]:
                        raise ValueError(
                            f"Config '{key}': input_list maps 'prm_id' (a UUID) into "
                            f"a prompt parameter. Use 'prm' (the dict) instead."
                        )

                    para = {it["para"]: row[it["value"]] for it in input_spec}
                    inst = prompt_template.format(**para)

                    if is_generation:
                        # inference config — no ground truth output
                        df_out.loc[len(df_out)] = [inst, "", "", "generation", inst]
                    else:
                        resp = output_template.format(**para)
                        text = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": inst},
                             {"role": "assistant", "content": resp}],
                            tokenize=False,
                        )
                        df_out.loc[len(df_out)] = [inst, "", resp, "", text]

            # ----------------------------------------------------------
            # TALL MODE
            # ----------------------------------------------------------
            else:
                df = data.copy()

                # We require a grouping key, otherwise curriculum is random soup.
                if "anchor_nn" not in df.columns:
                    raise ValueError(
                        f"[{key}] selection_mode='tall' requires 'anchor_nn' in dataframe. "
                        f"Got columns: {list(df.columns)}"
                    )

                # Sort so we pick the best k per anchor deterministically
                sort_cols = ["anchor_nn"]
                ascending = [True]

                if "accuracy" in df.columns:
                    sort_cols.append("accuracy")
                    ascending.append(False)

                if "anchor_jaccard" in df.columns:
                    sort_cols.append("anchor_jaccard")
                    ascending.append(False)

                # tie-breakers for stable output
                if "nn" in df.columns:
                    sort_cols.append("nn")
                    ascending.append(True)
                if "epoch" in df.columns:
                    sort_cols.append("epoch")
                    ascending.append(True)

                df = df.sort_values(sort_cols, ascending=ascending)

                # One prompt per anchor group (top-k neighbors)
                for anchor, g in tqdm(df.groupby("anchor_nn"), total=df["anchor_nn"].nunique()):
                    if len(g) < k:
                        continue

                    gk = g.head(k)

                    # sanity: should always be true
                    if gk["anchor_nn"].nunique() != 1:
                        raise RuntimeError(f"[{key}] mixed anchors in one group: {gk['anchor_nn'].unique()}")

                    # Convert to Series list for _pack_k_models
                    chunk = [pd.Series(r) for r in gk.to_dict(orient="records")]

                    packed = NNGenPrompt._pack_k_models(chunk,k,cfg) # strict guards run here
                    print("[TALL] anchor:", packed.get("anchor_nn", packed.get("anchor_nn_1", None)), "k:", k)

                    para = {it["para"]: packed[it["value"]] for it in input_spec}
                    inst = prompt_template.format(**para)

                    if is_generation:
                        df_out.loc[len(df_out)] = [inst, "", "", "generation", inst]
                    else:
                        resp = output_template.format(**para)
                        text = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": inst},
                             {"role": "assistant", "content": resp}],
                            tokenize=False,
                        )
                        df_out.loc[len(df_out)] = [inst, "", resp, "", text]
        out = pd.concat(prompt_frames, ignore_index=True)
        print(f"[NNGenPrompt] Prompt generation complete. rows={len(out)}", flush=True)
        return out
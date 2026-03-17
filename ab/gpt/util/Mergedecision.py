import json
import subprocess
from pathlib import Path

from ab.nn.util.Const import out_dir, ab_root_path

NNGPT_DIR = out_dir /  'nngpt'
LINEAGE_FILE = NNGPT_DIR / "accepted_adapters.json"
IMPROVEMENT_EPS = 1e-4
RUN_META = NNGPT_DIR / 'run_config.json'
CONF_LLM_DIR = ab_root_path / "ab/gpt/conf/llm"


def infer_base_model():
    """
    Read base model from run_config.json (created by TuneNNGen).
    Fallback to adapter_config.json for backward compatibility.

    Returns:
        str: Base model name (e.g., "deepseek-ai/deepseek-coder-1.3b-instruct")

    Raises:
        RuntimeError: If no config source found
    """
    # Priority 1: run_config.json (deterministic source)
    run_config_file = NNGPT_DIR / "run_config.json"

    if run_config_file.exists():
        try:
            with open(run_config_file, encoding="utf-8") as f:
                config = json.load(f)

            # First try base_model_name (new format)
            base_model = config.get("base_model_name")
            if base_model:
                return base_model

            # Fallback to llm_conf lookup (old format)
            llm_conf = config.get("llm_conf")
            if llm_conf:
                from ab.gpt.util.Const import conf_llm_dir
                llm_conf_path = conf_llm_dir / llm_conf

                if llm_conf_path.exists():
                    with open(llm_conf_path, encoding="utf-8") as f2:
                        llm_config = json.load(f2)
                    base_model = llm_config.get("base_model_name")
                    if base_model:
                        return base_model

        except Exception as e:
            print(f"Failed to read run_config.json: {e}")

    # Priority 2: adapter_config.json (fallback for old runs)
    epoch_dir = NNGPT_DIR / "llm" / "epoch"

    if epoch_dir.exists():
        for cfg_path in epoch_dir.rglob("adapter_config.json"):
            if "synth_nn" not in str(cfg_path):
                try:
                    with open(cfg_path, encoding="utf-8") as f:
                        config = json.load(f)

                    base_model = config.get("base_model_name_or_path")
                    if base_model:
                        return base_model
                except Exception:
                    continue

    raise RuntimeError(
        "Cannot infer base model. No run_config.json or adapter found. "
        "Run TuneNNGen at least once to create run_config.json."
    )


def load_lineage():
    """Load previous adapters or make a new one."""
    if LINEAGE_FILE.exists():
        with open(LINEAGE_FILE) as f:
            data = json.load(f)
        data.setdefault("adapters", [])
        data.setdefault("base_model", infer_base_model())
        return data

    return {
        "adapters": [],
        "base_model": infer_base_model()
    }


def save_lineage(lineage):
    """Save the adapters."""
    LINEAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LINEAGE_FILE, "w") as f:
        json.dump(lineage, f, indent=2)


def get_cycle_files():
    """Get cycle_results.json files for comparison."""
    current = NNGPT_DIR / "cycle_results.json"
    if not current.exists():
        raise RuntimeError("cycle_results.json not found")

    backups = list(NNGPT_DIR.glob("cycle_results_*.json"))
    if not backups:
        raise RuntimeError("No backup cycle_results_*.json found")

    # Sort by time to see two latest cycle_results.json
    previous = max(backups, key=lambda p: p.stat().st_mtime)
    return previous, current


def load_metrics(path):
    """Extract metrics from cycle results file."""
    with open(path) as f:
        data = json.load(f)

    evaluation = data.get("evaluation", {})

    return {
        "accuracy": evaluation.get("best_accuracy"),
        "cycle": data.get("cycle")
    }


def rebuild_model():
    """If accuracy increases, adapter is merged."""
    print("REBUILDING MODEL FROM UPDATED LINEAGE")

    try:
        subprocess.run(
            ["python", "-m", "ab.gpt.util.MergeLLM"],
            check=True
        )
        print("Rebuild complete\n")
        return True
    except subprocess.CalledProcessError as e:
        print("Rebuild failed")
        print(e)
        return False


def main():
    try:
        prev_file, curr_file = get_cycle_files()
        prev_m = load_metrics(prev_file)
        curr_m = load_metrics(curr_file)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    # --- Safe accuracy extraction ---
    prev_acc = prev_m.get("accuracy") or 0.0
    curr_acc = curr_m.get("accuracy") or 0.0

    current_cycle = curr_m.get("cycle")
    if current_cycle is None:
        raise RuntimeError("Cycle number missing in cycle_results.json")

    current_epoch = current_cycle - 1

    print("CYCLE COMPARISON")
    print(f"Previous: {prev_file.name} → Acc: {prev_acc:.6f}")
    print(f"Current : {curr_file.name} → Acc: {curr_acc:.6f}")

    # --- Decision rule: accuracy only ---
    delta = curr_acc - prev_acc

    if delta > IMPROVEMENT_EPS:
        decision = "KEEP"
        reason = f"Improved by {delta:.6f}"
    else:
        decision = "REVERT"
        reason = f"Delta {delta:.6f} below threshold"

    print(f"\nDecision: {decision} ({reason})")

    # --- Load lineage ---
    lineage = load_lineage()
    accepted_epochs = [a["epoch"] for a in lineage["adapters"]]

    # --- Update lineage ---
    if decision == "KEEP":
        if current_epoch not in accepted_epochs:
            lineage["adapters"].append({
                "epoch": current_epoch,
                "path": str(NNGPT_DIR /'llm' / f"epoch/A{current_epoch}")
            })
            accepted_epochs.append(current_epoch)
            print(f"Adapter A{current_epoch} ACCEPTED")
        else:
            print(f"Adapter A{current_epoch} already accepted")
    else:
        print(f"Adapter A{current_epoch} REJECTED")

    # ALWAYS save lineage (even on REVERT)
    save_lineage(lineage)

    print(f"Accepted lineage: {accepted_epochs}\n")

    # --- Save decision summary ---
    with open(NNGPT_DIR / "merge_decision.json", "w") as f:
        json.dump({
            "current_epoch": current_epoch,
            "previous_accuracy": prev_acc,
            "current_accuracy": curr_acc,
            "delta": delta,
            "decision": decision,
            "reason": reason,
            "accepted_lineage": accepted_epochs
        }, f, indent=2)

    print("Decision saved\n")

    # --- Rebuild if KEEP ---
    if decision == "KEEP":
        print("Triggering model rebuild...")
        rebuild_model()
        print("✓ Ready for next cycle\n")
    else:
        print("No rebuild needed\n")


if __name__ == "__main__":
    main()
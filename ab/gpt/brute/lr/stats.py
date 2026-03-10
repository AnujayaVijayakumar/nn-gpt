"""
Post-Evaluation Statistics for BO Model Generation

Scans eval_info.json files from NNEval output and produces
comprehensive statistics: best models, architecture rankings,
scheduler rankings, weight-decay analysis.

Usage:
    python -m ab.gpt.brute.lr.stats
"""
import json
import os
import re
from pathlib import Path
from collections import defaultdict

from ab.gpt.util.Const import epoch_dir, synth_dir

OUTPUT_DIR = synth_dir(epoch_dir(0))


def parse_model_config(model_id, new_nn_path):
    """Extract architecture, scheduler, and weight_decay from generated new_nn.py."""
    config = {"arch": "unknown", "scheduler": "unknown", "weight_decay": 0.0}
    try:
        with open(new_nn_path) as f:
            code = f.read()
        # Extract architecture
        arch_patterns = [
            ("ResNet18", r"resnet18"),
            ("ResNet34", r"resnet34"),
            ("ResNet50", r"resnet50"),
            ("DenseNet121", r"densenet121"),
            ("MobileNetV2", r"mobilenet_v2"),
            ("EfficientNetB0", r"efficientnet_b0"),
            ("VGG11bn", r"vgg11_bn"),
            ("ShuffleNetV2", r"shufflenet_v2"),
            ("SqueezeNet", r"squeezenet1_1"),
            ("MNASNet", r"mnasnet1_0"),
            ("GoogLeNet", r"googlenet"),
            ("RegNetY400MF", r"regnet_y_400mf"),
        ]
        for name, pat in arch_patterns:
            if re.search(pat, code):
                config["arch"] = name
                break
        # Extract weight_decay
        wd_match = re.search(r"weight_decay=([\d.e\-]+)", code)
        if wd_match:
            config["weight_decay"] = float(wd_match.group(1))
        # Extract scheduler type
        sched_patterns = [
            ("StepLR", r"lr_scheduler\.StepLR\("),
            ("MultiStepLR", r"lr_scheduler\.MultiStepLR\("),
            ("ExponentialLR", r"lr_scheduler\.ExponentialLR\("),
            ("PolynomialLR", r"lr_scheduler\.PolynomialLR\("),
            ("CosineAnnealingWR", r"CosineAnnealingWarmRestarts\("),
            ("CosineAnnealingLR", r"lr_scheduler\.CosineAnnealingLR\("),
            ("CyclicLR", r"lr_scheduler\.CyclicLR\("),
            ("OneCycleLR", r"lr_scheduler\.OneCycleLR\("),
            ("LinearLR", r"lr_scheduler\.LinearLR\(optimizer"),
            ("ConstantLR", r"lr_scheduler\.ConstantLR\("),
            ("LambdaLR", r"lr_scheduler\.LambdaLR\("),
            ("MultiplicativeLR", r"lr_scheduler\.MultiplicativeLR\("),
            ("ChainedScheduler", r"lr_scheduler\.ChainedScheduler\("),
            ("SequentialLR", r"lr_scheduler\.SequentialLR\("),
        ]
        for name, pat in sched_patterns:
            if re.search(pat, code):
                config["scheduler"] = name
                break
    except Exception:
        pass
    return config


def collect_results():
    """Collect all eval_info.json results."""
    results = []
    models_dir = Path(str(OUTPUT_DIR))
    if not models_dir.exists():
        print(f"Output directory {models_dir} does not exist.")
        return results

    for model_id in sorted(os.listdir(models_dir)):
        model_dir = models_dir / model_id
        if not model_dir.is_dir():
            continue
        eval_path = model_dir / "eval_info.json"
        nn_path = model_dir / "new_nn.py"
        if not eval_path.exists():
            continue
        try:
            with open(eval_path) as f:
                data = json.load(f)
            eval_results = data.get("eval_results", {})
            config = parse_model_config(model_id, nn_path) if nn_path.exists() else {}
            acc = None
            quality = None
            if isinstance(eval_results, dict):
                acc = eval_results.get("acc") or eval_results.get("accuracy")
                quality = eval_results.get("quality") or eval_results.get("code_quality")
            results.append({
                "model_id": model_id,
                "arch": config.get("arch", "unknown"),
                "scheduler": config.get("scheduler", "unknown"),
                "weight_decay": config.get("weight_decay", 0.0),
                "accuracy": acc,
                "quality": quality,
                "eval_results": eval_results,
            })
        except Exception as e:
            print(f"  Error reading {eval_path}: {e}")
    return results


def print_stats(results):
    """Print comprehensive statistics."""
    if not results:
        print("No evaluation results found.")
        return

    # Filter successful (those with accuracy)
    successful = [r for r in results if r["accuracy"] is not None]
    failed = [r for r in results if r["accuracy"] is None]

    print(f"\n{'='*80}")
    print(f"  MODEL GENERATION STATISTICS  (BO prefix, nn_dataset framework)")
    print(f"{'='*80}")
    print(f"\n  Total models evaluated:  {len(results)}")
    print(f"  Successful:              {len(successful)}")
    print(f"  Failed:                  {len(failed)}")

    if not successful:
        print("\nNo successful models to analyze.")
        return

    # Sort by accuracy descending
    successful.sort(key=lambda r: r["accuracy"], reverse=True)

    # --- Top 20 Best Models ---
    print(f"\n{'─'*80}")
    print(f"  TOP 20 MODELS BY ACCURACY")
    print(f"{'─'*80}")
    print(f"  {'Rank':<6}{'Model ID':<14}{'Arch':<16}{'Scheduler':<22}{'WD':<10}{'Acc %':<10}{'Quality':<8}")
    print(f"  {'─'*6}{'─'*14}{'─'*16}{'─'*22}{'─'*10}{'─'*10}{'─'*8}")
    for i, r in enumerate(successful[:20], 1):
        q = f"{r['quality']:.2f}" if r['quality'] else "N/A"
        print(f"  {i:<6}{r['model_id']:<14}{r['arch']:<16}{r['scheduler']:<22}{r['weight_decay']:<10.0e}{r['accuracy']*100:<10.2f}{q:<8}")

    # --- Per-Architecture Stats ---
    arch_stats = defaultdict(list)
    for r in successful:
        arch_stats[r["arch"]].append(r["accuracy"])

    print(f"\n{'─'*80}")
    print(f"  ARCHITECTURE RANKING (by mean accuracy)")
    print(f"{'─'*80}")
    print(f"  {'Architecture':<20}{'Count':<8}{'Mean %':<10}{'Best %':<10}{'Worst %':<10}{'Std %':<8}")
    print(f"  {'─'*20}{'─'*8}{'─'*10}{'─'*10}{'─'*10}{'─'*8}")
    arch_ranking = sorted(arch_stats.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
    for arch, accs in arch_ranking:
        mean_a = sum(accs) / len(accs)
        std_a = (sum((a - mean_a)**2 for a in accs) / len(accs)) ** 0.5
        print(f"  {arch:<20}{len(accs):<8}{mean_a*100:<10.2f}{max(accs)*100:<10.2f}{min(accs)*100:<10.2f}{std_a*100:<8.2f}")

    # --- Per-Scheduler Stats ---
    sched_stats = defaultdict(list)
    for r in successful:
        sched_stats[r["scheduler"]].append(r["accuracy"])

    print(f"\n{'─'*80}")
    print(f"  SCHEDULER RANKING (by mean accuracy)")
    print(f"{'─'*80}")
    print(f"  {'Scheduler':<24}{'Count':<8}{'Mean %':<10}{'Best %':<10}{'Worst %':<10}")
    print(f"  {'─'*24}{'─'*8}{'─'*10}{'─'*10}{'─'*10}")
    sched_ranking = sorted(sched_stats.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
    for sched, accs in sched_ranking:
        mean_a = sum(accs) / len(accs)
        print(f"  {sched:<24}{len(accs):<8}{mean_a*100:<10.2f}{max(accs)*100:<10.2f}{min(accs)*100:<10.2f}")

    # --- Weight Decay Analysis ---
    wd_stats = defaultdict(list)
    for r in successful:
        wd_stats[r["weight_decay"]].append(r["accuracy"])

    print(f"\n{'─'*80}")
    print(f"  WEIGHT DECAY ANALYSIS")
    print(f"{'─'*80}")
    print(f"  {'Weight Decay':<15}{'Count':<8}{'Mean %':<10}{'Best %':<10}")
    print(f"  {'─'*15}{'─'*8}{'─'*10}{'─'*10}")
    wd_ranking = sorted(wd_stats.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
    for wd, accs in wd_ranking:
        mean_a = sum(accs) / len(accs)
        print(f"  {wd:<15.0e}{len(accs):<8}{mean_a*100:<10.2f}{max(accs)*100:<10.2f}")

    # --- Best Model per Architecture ---
    print(f"\n{'─'*80}")
    print(f"  BEST MODEL PER ARCHITECTURE")
    print(f"{'─'*80}")
    best_per_arch = {}
    for r in successful:
        if r["arch"] not in best_per_arch or r["accuracy"] > best_per_arch[r["arch"]]["accuracy"]:
            best_per_arch[r["arch"]] = r
    for arch in sorted(best_per_arch.keys()):
        r = best_per_arch[arch]
        print(f"  {arch:<20} → {r['model_id']}  ({r['scheduler']}, wd={r['weight_decay']:.0e})  Acc: {r['accuracy']*100:.2f}%")

    # --- Summary ---
    total_acc = sum(r["accuracy"] for r in successful)
    mean_acc = total_acc / len(successful)
    print(f"\n{'─'*80}")
    print(f"  SUMMARY")
    print(f"{'─'*80}")
    print(f"  Overall mean accuracy:  {mean_acc*100:.2f}%")
    print(f"  Best single model:      {successful[0]['model_id']} ({successful[0]['arch']}, {successful[0]['scheduler']}) → {successful[0]['accuracy']*100:.2f}%")
    print(f"  Best architecture:      {arch_ranking[0][0]} (mean {sum(arch_ranking[0][1])/len(arch_ranking[0][1])*100:.2f}%)")
    print(f"  Best scheduler:         {sched_ranking[0][0]} (mean {sum(sched_ranking[0][1])/len(sched_ranking[0][1])*100:.2f}%)")
    print(f"  Best weight decay:      {wd_ranking[0][0]:.0e} (mean {sum(wd_ranking[0][1])/len(wd_ranking[0][1])*100:.2f}%)")
    print(f"{'='*80}\n")


def main():
    print("Collecting evaluation results...")
    results = collect_results()
    print_stats(results)

    # Save results as JSON
    out_path = Path(str(OUTPUT_DIR)) / "bo_statistics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

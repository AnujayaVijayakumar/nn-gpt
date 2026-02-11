import os
import argparse
import hashlib
import json
import time
from contextlib import contextmanager
import sys
# FIX MODULE PATH: Add repo root to sys.path
# .../ab/gpt/brute/ga/meta_evolution/run_fractal_evolution.py -> .../ -> .../ -> .../ -> .../ -> .../ (5 levels up)
# But safer to just add the root '/a/mm' if we know it, or relative.
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../../../../"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
import socket
import torch
from ab.gpt.brute.ga.meta_evolution.genetic_algorithm import GeneticAlgorithm
from ab.gpt.brute.ga.meta_evolution.FractalNet_evolvable import SEARCH_SPACE, generate_model_code_string
from ab.gpt.util.Eval import Eval
from ab.gpt.iterative_pipeline.gpu_memory_manager import get_gpu_memory_info

import logging

# Configure logging to be simpler (remove timestamps for cleaner output)
logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# This is the folder where unique fractal models will be saved
ARCH_DIR = os.path.join(BASE_DIR, 'ga_fractal_arch') 
STATS_DIR = os.path.join(BASE_DIR, 'stats')
CHECKPOINT = 'fractal_ga_ckpt.pkl'

os.makedirs(ARCH_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)


seen_hashes = set()

def get_hash(s): return hashlib.md5(s.encode()).hexdigest()

def fitness_function(chromosome):
    try:
        # 1. Generate Source Code
        code_str = generate_model_code_string(chromosome)
        code_hash = get_hash(code_str)
        
        # Deduplication
        if code_hash in seen_hashes: return 0.0
        
        # 2. Save Model File to ARCH_DIR
        model_name = f"fractal_ga_{code_hash[:8]}"
        filepath = os.path.join(ARCH_DIR, f"{model_name}.py")
        
        # Save to execution dir
        with open(filepath, 'w') as f: 
            f.write(code_str)
            

            
        # 3. Evaluate
        eval_prm = {
            'lr': chromosome['lr'],
            'momentum': chromosome['momentum'],
            'batch': 32, 
            'epoch': 1, # Short epochs for Meta-Evaluation
            'transform': "norm_256_flip" 
        }
        
        stats_path = os.path.join(STATS_DIR, f"stats_{model_name}")
        
        evaluator = Eval(
            model_source_package=ARCH_DIR,
            task='img-classification',
            dataset='cifar-10',
            metric='acc',
            prm=eval_prm,
            save_to_db=False,
            prefix=model_name,
            save_path=stats_path
        )
        
        start_time = time.time()
        # with suppress_output(): 
        res = evaluator.evaluate(filepath)
        end_time = time.time()
        
        acc = 0.0
        if isinstance(res, dict): acc = res.get('accuracy', 0.0)
        elif isinstance(res, (float, int)): acc = float(res)
        elif isinstance(res, tuple): acc = float(res[1])
        
        seen_hashes.add(code_hash)
        
        # Capture GPU and System Stats using Standard Utility
        total_gb, used_gb, free_gb = get_gpu_memory_info()
        gpu_info = {
            "total_memory_gb": total_gb,
            "used_memory_gb": used_gb,
            "free_memory_gb": free_gb,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        }

        stats_data = {
            "model_name": model_name,
            "hash": code_hash,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": end_time - start_time,
            "accuracy": acc * 100,
            "chromosome": chromosome,
            "gpu_stats": gpu_info,
            "hostname": socket.gethostname()
        }

        # Merge full evaluation results from training_summary.json if available
        # The return value 'res' is often incomplete.
        summary_path = os.path.join(os.getcwd(), 'out', 'training_summary.json')
        full_res = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    full_res = json.load(f)
                    stats_data.update(full_res)
            except Exception as e:
                print(f"Failed to read training summary: {e}")
        elif isinstance(res, dict):
             # Fallback to result dict if file missing
             full_res = res
             stats_data.update(res)

        # Save stats dictionary to a JSON file
        model_stats_dir = os.path.join(STATS_DIR, model_name)
        os.makedirs(model_stats_dir, exist_ok=True)
        
        # Save aggregate stats
        stats_file_json = os.path.join(model_stats_dir, "stats.json")
        # with open(stats_file_json, 'w') as f:
        #     json.dump(stats_data, f, indent=4)
            
        # Save per-epoch stats if available
        # Check both top-level keys and nested structures which vary by Eval version
        epoch_data = full_res.get('epoch_details')
        if not epoch_data and 'learning_curves' in full_res:
             # Construct epoch details from learning curves if strictly separated
             lc = full_res['learning_curves']
             epochs = lc.get('epochs', [])
             train_loss = lc.get('train_loss', [])
             test_loss = lc.get('test_loss', [])
             epoch_data = []
             for i, ep in enumerate(epochs):
                 epoch_data.append({
                     'epoch': ep,
                     'train_loss': train_loss[i] if i < len(train_loss) else None,
                     'test_loss': test_loss[i] if i < len(test_loss) else None
                 })

        if epoch_data:
            for epoch_stat in epoch_data:
                ep_num = epoch_stat.get('epoch', 'unknown')
                
                # Inject UID/Hash into the epoch file
                ep_stat_copy = epoch_stat.copy()
                if 'uid' in full_res:
                    ep_stat_copy['uid'] = full_res['uid']
                else:
                    ep_stat_copy['uid'] = code_hash
                
                ep_file = os.path.join(model_stats_dir, f"{ep_num}.json")
                with open(ep_file, 'w') as f:
                    json.dump(ep_stat_copy, f, indent=4)

        # Store accuracy in chromosome for later "best" retrieval
        print(f"  Model Path: {filepath}")
        print(f"  Stats Path: {stats_file_json}")
        print("\n" + "="*50)
        print(f" >>> FITNESS SCORE (Accuracy): {acc * 100:.4f}% <<<")
        print("="*50 + "\n")
        chromosome['accuracy'] = acc * 100
        return acc * 100 
        
    except Exception as e:
        print(f"Eval Fail: {e}")
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=3)
    parser.add_argument("--pop", type=int, default=10)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    try:
        ga = GeneticAlgorithm(
            population_size=args.pop,
            search_space=SEARCH_SPACE,
            elitism_count=5,
            mutation_rate=0.2,
            checkpoint_path=CHECKPOINT
        )
        
        best, history = ga.run(args.gens, fitness_function)
        
        # Save Best Architecture
        if best:
             best_code = generate_model_code_string(best['chromosome'])
             best_path = os.path.join(BASE_DIR, "best_fractal_model.py")
             with open(best_path, "w") as f:
                 f.write(best_code)
             # print(f"Saved best model to {best_path}") # Keep silent for meta-score parsing

        # Meta-Score Calculation
        if history:
            avg_imp = (history[-1] - history[0]) if len(history) > 1 else 0
            peak = max(history)
            meta_score = peak + (avg_imp * 1.5) 
        else:
            meta_score = 0.0
            
        print(f"META_SCORE: {meta_score:.4f}")

    except Exception:
        # print(f"CRITICAL GA FAIL: {e}")
        print("META_SCORE: 0.0")
import os
import random
import time
import re
import datetime
import json
import hashlib
import glob
import textwrap
import ast
import shutil

# Local imports
from .mutation import MutationStrategy
from .llm_loader import LocalLLMLoader as LLMLoader
from .rl_rewards import evaluate_code_and_reward

# --- TARGETED MICRO-MUTATION PROMPTS ---
PROMPTS = {
    "mutate_optimizer": """You are an expert Deep Learning Engineer. Rewrite the 'train_setup' method in the Net class.
GOAL: Change the optimizer strategy to improve training convergence.
OPTIONS: Adam, SGD (with momentum), RMSprop, AdamW.
Input Code:
{slot_code}

History of recent attempts:
{history}

Based on the history, write a configuration that is mathematically distinct from previous failures.
Output ONLY the code block starting with 'def train_setup(self, prm):'."""
}

class LLMMutationStrategy(MutationStrategy):
    def __init__(self, model_path, log_file="dataset/mutation_log.jsonl", q_table_path=None, use_quantization=True):
        self.mutation_rate = 0.9  # High rate as this strategy is called explicitly
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path normalization
        if os.path.isabs(log_file):
            self.log_file = log_file 
        else:
            self.log_file = os.path.join(base_dir, log_file)
        
        self.adapter_save_path = os.path.join(base_dir, "fine_tuned_adapter")
        history_dir = os.path.join(base_dir, "adapter_history")

        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        os.makedirs(history_dir, exist_ok=True)
        
        # Startup Backup of existing adapters
        if os.path.exists(self.adapter_save_path):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(history_dir, f"backup_startup_{timestamp}")
            shutil.copytree(self.adapter_save_path, backup_path)
        
        # Initialize LLM
        self.llm_loader = LLMLoader(model_path, use_quantization, adapter_path=self.adapter_save_path)
        
        # Load Code History (deduplication)
        self.seen_hashes = set()
        fractals_dir = os.path.join(base_dir, "Fractals")
        if os.path.exists(fractals_dir):
            files = glob.glob(os.path.join(fractals_dir, "FracNet_*.py"))
            for fpath in files:
                try:
                    with open(fpath, "r") as f:
                        code = f.read()
                        h = hashlib.md5(code.encode('utf-8')).hexdigest()
                        self.seen_hashes.add(h)
                except Exception:
                    pass
        
        self.mutation_history = []
        self.training_buffer = []
        self.best_fitness_seen = 0.0
        self.session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_class_context(self, class_node):
        """Scrapes method names and self.attributes from class AST."""
        methods = []
        attributes = []
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                methods.append(node.name)
                # Look inside __init__ for layers
                if node.name == "__init__":
                    for sub_node in ast.walk(node):
                        if isinstance(sub_node, ast.Attribute) and isinstance(sub_node.value, ast.Name):
                            if sub_node.value.id == 'self':
                                attributes.append(sub_node.attr)
        return sorted(list(set(methods))), sorted(list(set(attributes)))

    def _cleanup_llm_response(self, raw_text, action):
        """Extracts code block, verifies syntax, and performs safety checks."""
        code = raw_text
        # Markdown extraction
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        code = code.strip()
        if not code:
            return ""

        # Syntax Check
        try:
            ast.parse(code)
        except SyntaxError:
            print(f"[RL-GA] Syntax Error in LLM response.")
            return ""

        # Semantic/Safety Checks
        if action == "mutate_optimizer":
            import torch.optim as optim
            found_optimizers = re.findall(r'(?:optim\.|torch\.optim\.)([a-zA-Z0-9_]+)', code)
            valid_opts = dir(optim)
            for opt in found_optimizers:
                if opt not in valid_opts and opt != "Optimizer":
                    return ""
                    
        return code

    def _extract_slot_ast(self, code, action):
        """
        Finds target code segment.
        Returns:
            segment (str): The code of the function to replace.
            span (tuple): (start_index, end_index) in the file string.
            indent_col (int): The number of spaces of indentation required.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None, None, 0
            
        path = ["Net", "train_setup"] if action == "mutate_optimizer" else []
        if not path: return None, None, 0

        found_class = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == path[0]), None)
        if not found_class: return None, None, 0

        target_node = next((n for n in found_class.body if isinstance(n, ast.FunctionDef) and n.name == path[1]), None)
        
        if target_node:
            segment = ast.get_source_segment(code, target_node)
            
            lines = code.splitlines(keepends=True)
            start_lineno = target_node.lineno - 1
            start_col = target_node.col_offset # This is the exact indentation level
            
            # Calculate start byte offset (Start of the line to replace indentation too)
            start_idx = sum(len(lines[i]) for i in range(start_lineno)) 
            
            # Calculate end byte offset
            end_lineno = target_node.end_lineno - 1
            end_col = target_node.end_col_offset
            end_idx = sum(len(lines[i]) for i in range(end_lineno)) + end_col
            
            return segment, (start_idx, end_idx), start_col
            
        return None, None, 0

    def mutate(self, chromosome, search_space):
        # Apply mutation probability
        if random.random() > self.mutation_rate:
            return chromosome
        
        full_code = chromosome['code']
        current_fitness = chromosome.get('cached_fitness', 0.0)
        action_key = "mutate_optimizer"
        
        # 1. Gather Class Context via AST
        try:
            tree = ast.parse(full_code)
            net_class = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Net"), None)
            if net_class:
                methods, attrs = self._get_class_context(net_class)
            else:
                methods, attrs = [], []
        except:
            methods, attrs = [], []

        # 2. Extract Slot and Indentation
        target_content, span, indent_col = self._extract_slot_ast(full_code, action_key)
        if not target_content: 
            return chromosome

        # 3. Construct Prompt
        relevant_history = [m['summary'] for m in self.mutation_history if m['action'] == action_key]
        history_str = "\n".join(relevant_history[-3:]) if relevant_history else "None"
        
        prompt_instr = PROMPTS[action_key].format(slot_code=target_content, history=history_str)
        context_info = f"Class 'Net' available attributes: {attrs}"

        full_prompt = (
            f"### Context:\n{context_info}\n"
            f"### Task: {prompt_instr}\n"
            f"### Original Code:\n```python\n{target_content}\n```\n"
            "### Instructions:\n"
            "1. Return ONLY the python code for the method.\n"
            "2. Use 'self.parameters()' for the optimizer.\n"
        )

        error_feedback = ""
        attempts_file = os.path.join(os.path.dirname(self.log_file), "mutation_attempts.jsonl")

        for attempt in range(3):
            # Dynamic Temperature
            temp = 0.8 + (attempt * 0.1)
            current_prompt = full_prompt + (f"\n### PREVIOUS ERROR:\n{error_feedback}" if error_feedback else "")
            
            try:
                raw_response = self.llm_loader.generate(current_prompt, max_new_tokens=600, temperature=temp)
                new_slot_code = self._cleanup_llm_response(raw_response, action_key)
            except Exception as e:
                error_feedback = f"Generation failed: {e}"
                continue
            
            if not new_slot_code:
                error_feedback = "Response invalid or empty."
                continue

            # --- INDENTATION FIX ---
            # 1. Generate the exact spacing string required by the AST
            indent_str = " " * indent_col 
            
            # 2. Normalize the code: remove existing common indentation, then strip
            clean_code = textwrap.dedent(new_slot_code).strip()
            
            # 3. Re-apply the correct indentation to every line
            reindented_code = "\n".join([indent_str + line for line in clean_code.splitlines()])
            # -----------------------

            # Reconstruct the full file
            final_code = full_code[:span[0]] + reindented_code + full_code[span[1]:]
            
            # Deduplication
            code_hash = hashlib.md5(final_code.encode('utf-8')).hexdigest()
            if code_hash in self.seen_hashes: 
                error_feedback = "Code identical to a previous attempt."
                continue

            # Log
            attempt_log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "action": action_key,
                "attempt": attempt + 1,
                "response_preview": new_slot_code[:50] + "...",
                "status": "pending",
                "error": None,
                "reward": 0.0,
                "fitness": 0.0
            }
            
            # Evaluate
            print(f"[RL-GA] Evaluating {action_key} (Attempt {attempt+1})...")
            res = evaluate_code_and_reward(
                final_code, 
                prm=chromosome, 
                log_file=self.log_file, 
                prompt_used=current_prompt,
                val_metric_baseline=current_fitness
            )
            
            if res.get('status') == 'error':
                error_feedback = res.get('message', 'Unknown execution error')
                print(f"[RL-GA] Attempt {attempt+1} Failed: {error_feedback}")
                attempt_log_entry['status'] = 'error'
                attempt_log_entry['error'] = error_feedback
                with open(attempts_file, "a") as f: f.write(json.dumps(attempt_log_entry) + "\n")
                continue
            
            # Success Handling
            reward = res.get('reward', 0.0)
            new_fitness = res.get('val_metric', 0.0)
            
            self.seen_hashes.add(code_hash)
            
            attempt_log_entry['reward'] = reward
            attempt_log_entry['fitness'] = new_fitness

            if reward == 0.0:
                 print(f"[RL-GA] Refused (No Improvement). Acc: {new_fitness:.4f}")
                 attempt_log_entry['status'] = 'refused_zero_reward'
                 error_feedback = "Code ran but performance did not improve over baseline."
            else:
                 attempt_log_entry['status'] = 'success'
                 print(f"[RL-GA] Success! Reward: {reward:.4f}, Acc: {new_fitness:.4f}")
            
            with open(attempts_file, "a") as f: f.write(json.dumps(attempt_log_entry) + "\n")

            # Learning Step
            if new_fitness > self.best_fitness_seen:
                self.best_fitness_seen = new_fitness
            
            if reward > 0.0:
                self.mutation_history.append({'action': action_key, 'summary': f"Acc: {new_fitness:.4f}", 'code': new_slot_code})
                self.training_buffer.append({'prompt': full_prompt, 'completion': new_slot_code})
                
                if len(self.training_buffer) >= 4:
                    print("[RL-GA] Updating LLM weights...")
                    self.llm_loader.train_on_buffer(self.training_buffer)
                    self.llm_loader.save_adapters(self.adapter_save_path)
                    self.training_buffer = []

                mutated_ind = chromosome.copy()
                mutated_ind['code'] = final_code
                mutated_ind['cached_fitness'] = new_fitness
                return mutated_ind
            
        return chromosome
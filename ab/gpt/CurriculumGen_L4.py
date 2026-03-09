import json
from pathlib import Path
from os import makedirs

from peft import PeftModel

import ab.nn.api as lemur
import ab.gpt.NNEval as NNEval

from ab.gpt.util.Chatbot import ChatBot
from ab.gpt.util.Const import (
    conf_llm_dir,
    epoch_dir,
    synth_dir,
    new_nn_file,
    new_out_file,
    hp_file,
    transformer_file,
)
from ab.gpt.util.LLM import LLM
from ab.gpt.util.LLMUtil import quantization_config_4bit
from ab.gpt.util.Util import create_file
from ab.nn.util.Util import release_memory
from ab.gpt.util.prompt.NNGenPromptCurriculum import NNGenPrompt

LLM_CONF = "ds_coder_7b_olympic.json"
CURRICULUM_JSON = "/home/anu/PycharmProjects/CVPraktikum/nn-gpt/ab/gpt/conf/prompt/train/NN_gen_Curriculum.json"
TARGET_KEY = "curriculum_L4_very_low_far_k5"

EPOCH = 0
MAX_PROMPTS = 20          # total prompts to fetch from get_raw_dataset
MAX_NEW_TOKENS = 8192
TEMPERATURE = 0.2
TOP_K = 50
TOP_P = 0.9

ONLY_BEST_ACCURACY = True
RUN_EVAL = True
NN_TRAIN_EPOCHS = 1
NN_NAME_PREFIX = None

# Optional: set to a LoRA checkpoint path to generate with a tuned model
LORA_PATH = None

def extract_single_key_cfg(src_json: str, key: str) -> Path:
    """
    Temporary JSON config containing only one curriculum key.
    """
    src = Path(src_json)
    with open(src, "r") as f:
        cfg = json.load(f)

    if key not in cfg:
        raise KeyError(f"Key '{key}' not found in {src_json}. Available keys: {list(cfg.keys())}")

    out_path = src.parent / f"__tmp_{key}.json"
    with open(out_path, "w") as f:
        json.dump({key: cfg[key]}, f, indent=2)

    return out_path

def load_model_and_tokenizer(llm_conf_name: str):
    with open(conf_llm_dir / llm_conf_name, "r") as f:
        llm_cfg = json.load(f)

    token_from_file = llm_cfg.get("token_from_file", False)
    base_model_name = llm_cfg["base_model_name"]
    use_deepspeed = llm_cfg.get("use_deepspeed", False)
    context_length = llm_cfg.get("context_length")
    use_unsloth = llm_cfg.get("use_unsloth", False)
    load_in_4bit = llm_cfg.get("load_in_4bit", True)

    access_token = None
    if token_from_file:
        from ab.nn.util.Const import ab_root_path
        with open(ab_root_path / "token", "r") as f:
            access_token = f.readline().strip()

    model_loader = LLM(
        base_model_name,
        quantization_config_4bit,
        access_token=access_token,
        use_deepspeed=use_deepspeed,
        context_length=context_length,
        training_args=None,
        use_unsloth=use_unsloth,
        load_in_4bit=load_in_4bit,
    )

    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()

    if LORA_PATH:
        print(f"[INFO] Loading LoRA from: {LORA_PATH}")
        model = PeftModel.from_pretrained(model, LORA_PATH, is_trainable=False)
        model = model.merge_and_unload()

    return model, tokenizer, llm_cfg


def save_generation(model_dir: Path, code: str, hp: str, tr: str, full_out: str):
    makedirs(model_dir, exist_ok=True)

    create_file(model_dir, new_out_file, full_out)

    if code and code.strip():
        create_file(model_dir, new_nn_file, code)
    else:
        print(f"[WARN] No <nn> code generated for {model_dir.name}")

    if hp and hp.strip():
        try:
            hp_obj = json.loads(hp.replace("'", '"'))
            with open(model_dir / hp_file, "w") as f:
                json.dump(hp_obj, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to parse/save hp for {model_dir.name}: {e}")

    if tr and tr.strip():
        create_file(model_dir, transformer_file, tr)

def main():
    print(f"[INFO] Target curriculum key: {TARGET_KEY}")

    tmp_cfg_path = extract_single_key_cfg(CURRICULUM_JSON, TARGET_KEY)

    model, tokenizer, llm_cfg = load_model_and_tokenizer(LLM_CONF)

    max_input_length = llm_cfg.get("max_input_length", None)
    max_len = max_input_length or tokenizer.model_max_length

    chat_bot = ChatBot(
        model,
        tokenizer,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )

    prompt_builder = NNGenPrompt(
        max_len=max_len,
        tokenizer=tokenizer,
        prompts_path=str(tmp_cfg_path),
    )

    df = prompt_builder.get_raw_dataset(
        only_best_accuracy=ONLY_BEST_ACCURACY,
        n_training_prompts=MAX_PROMPTS,
    )

    # only generation rows
    if "category" in df.columns:
        df = df[df["category"] == "generation"].copy()

    print(f"[INFO] prompts ready: {len(df)}")
    if df.empty:
        raise RuntimeError("No prompts generated. Check config/query coverage.")

    out_path = epoch_dir(EPOCH)
    models_dir = synth_dir(out_path)
    makedirs(models_dir, exist_ok=True)

    for idx, (_, row) in enumerate(df.iterrows()):
        prompt = row["instruction"]
        model_dir = models_dir / f"L4_{idx}"

        print(f"\n[INFO] Generating sample {idx + 1}/{len(df)} -> {model_dir.name}")

        if max_input_length:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
            )
            if len(rendered) > max_input_length:
                print(f"[WARN] Prompt too long ({len(rendered)} > {max_input_length}), skipping {model_dir.name}")
                continue

        code, hp, tr, full_out = chat_bot.chat(
            prompt,
            engineer_prompt=False,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        save_generation(model_dir, code, hp, tr, full_out)

    release_memory()

    if RUN_EVAL and models_dir.exists():
        print("[INFO] Running evaluation...")
        NNEval.main(NN_NAME_PREFIX, NN_TRAIN_EPOCHS, EPOCH)
        release_memory()

    lemur.data.cache_clear()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
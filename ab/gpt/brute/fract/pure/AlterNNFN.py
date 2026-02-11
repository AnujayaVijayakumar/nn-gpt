import json
import shutil
import itertools
from pathlib import Path

from ab.gpt.util.Const import conf_test_dir, epoch_dir, new_nn_file, synth_dir, fract_dir
from ab.gpt.util.LLM import LLM


def alter(epochs, test_conf, llm_name, gguf_file=None):
    # Load test prompts (kept for compatibility)
    with open(conf_test_dir / test_conf) as f:
        prompt_dict = json.load(f)
    assert isinstance(prompt_dict, dict)

    model_loader = LLM(llm_name, gguf_file=gguf_file)
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    print("Load Model Complete, Start Loop...")

    # Clean old runs
    shutil.rmtree(epoch_dir(), ignore_errors=True)

    # -------- TEMPLATE --------
    PURE_DIR = Path(__file__).resolve().parent
    TEMPLATE_PATH = PURE_DIR / "Fractal_template.py"
    template = TEMPLATE_PATH.read_text()

    # -------- ELEMENT LIST ( $$ ) --------
    element_list = [
        'nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)',
        'nn.MaxPool2d(kernel_size=3, stride=2)',
        'nn.BatchNorm2d(out_channels)',
        'nn.ReLU(inplace=True)',
        'nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()',
    ]

    element_list_str = "['Conv2d', 'MaxPool2d', 'BatchNorm2d', 'ReLU', 'Dropout2d']"

    # -------- BASE SKIP BLOCK ( @@ ) --------
    # PREFIX GROWTH ONLY
    base_skip_block = [
        " nn.Conv2d(in_ch_ij, out_channels, kernel_size=3, stride=1, padding=1, bias=False)",
        "               nn.ReLU(inplace=True)",
        "               nn.BatchNorm2d(out_channels)",
        "               nn.MaxPool2d(kernel_size=3, stride=2)",
    ]
    # -------- GENERATION --------
    max_variants = 1200
    counter = 0

    for epoch in range(epochs):
        out_path = epoch_dir(epoch)
        out_path.mkdir(parents=True, exist_ok=True)

        # num_skip_blocks = 2 → 4 (prefix, NOT repetition)
        for num_skip_blocks in range(2, 5):

            # PREFIX slice (THIS IS THE KEY FIX)
            skip_layers = base_skip_block[:num_skip_blocks]
            skip_block_code = ",\n        ".join(skip_layers)

            # permutations / combinations of $$
            for r in range(2, 6):  # length of element_list sequence
                for perm in itertools.product(element_list, repeat=r):

                    element_code = ",\n        ".join(perm)

                    for N in range(1, 6):
                        for num_columns in range(1, 8):

                            if counter >= max_variants:
                                return

                            model_dir = synth_dir(out_path) / f"B{counter}"
                            model_dir.mkdir(parents=True, exist_ok=True)

                            nn_code = (
                                template
                                .replace("$$", element_code)
                                .replace("@@", skip_block_code)
                                .replace("??", element_list_str)
                                .replace("?1", str(N))
                                .replace("?2", str(num_columns))
                            )

                            (model_dir / new_nn_file).write_text(nn_code)

                            print(f"C{counter}: len={len(perm)}, N={N}, num_columns={num_columns}")
                            counter += 1

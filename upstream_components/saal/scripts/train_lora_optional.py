"""OPTIONAL LoRA SFT adapter example; not executed by the core reference study.

Requires pip install -e '.[train-lora]'. Use *only* a separately reviewed training
JSONL, never sealed evaluation episodes. Requires locally available HF model.
No training job is triggered automatically by runtime or by this repository.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local-base-model", required=True,
                   help="path to trusted local Hugging Face causal LM, not arbitrary remote code")
    p.add_argument("--qualified-train-jsonl", required=True)
    p.add_argument("--admission-manifest", required=True)
    p.add_argument("--admission-public-key", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--max-steps", type=int, default=50)
    args = p.parse_args()
    from saal.admission import verify_manifest
    manifest = json.loads(Path(args.admission_manifest).read_text())
    if not verify_manifest(args.qualified_train_jsonl, manifest, args.admission_public_key):
        raise PermissionError("missing or invalid independent data admission signature")
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    rows = [json.loads(line) for line in Path(args.qualified_train_jsonl).read_text().splitlines() if line]
    if not rows or any(row.get("admitted_by") != "independent_data_authority" for row in rows):
        raise ValueError("training records missing independent admission marker")
    if any(row.get("source_partition") != "train" for row in rows):
        raise ValueError("holdout contamination: train split only")
    tok = AutoTokenizer.from_pretrained(args.local_base_model, local_files_only=True,
                                         trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.local_base_model,
            local_files_only=True, trust_remote_code=False)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                               target_modules=args.target_modules, bias="none", task_type="CAUSAL_LM"))
    def prep(row):
        # SFT seed; not RL and not a proof of beneficial-trait generalization.
        prompt = f"User task: {row['prompt']}\nAssistant response: {row['ideal_response']}"
        tokens = tok(prompt, truncation=True, padding="max_length", max_length=256)
        tokens["labels"] = [x if mask else -100 for x, mask in zip(tokens["input_ids"], tokens["attention_mask"])]
        return tokens
    ds = Dataset.from_list(rows).map(prep, remove_columns=list(rows[0]))
    kwargs = dict(output_dir=args.output_dir, max_steps=args.max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=1e-4, logging_steps=10, save_strategy="no", report_to="none")
    trainer = Trainer(model=model, args=TrainingArguments(**kwargs), train_dataset=ds)
    trainer.train()
    model.save_pretrained(args.output_dir)
    print("Candidate adapter saved. NOT eligible for promotion without external sealed qualification.")


if __name__ == "__main__":
    main()

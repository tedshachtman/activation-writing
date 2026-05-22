"""LoRA fine-tuning baseline for the mini-language consolidation task."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caic.modeling import (
    ForwardCounter,
    get_attention_o_module,
    get_decoder_layers,
    get_mlp_down_module,
    load_model_and_tokenizer,
    set_attention_o_module,
    set_mlp_down_module,
)
from scripts.minilang_write import (
    build_balanced_questions,
    build_questions,
    evaluate_mc,
    format_prompt,
    render_lesson,
)


class LoraLinear(nn.Module):
    """A frozen linear layer plus trainable low-rank adapter."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        self.merged = False
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        hidden = F.linear(x.float(), self.lora_a)
        update = F.linear(hidden, self.lora_b) * self.scaling
        return out + update.to(out.dtype)

    @torch.no_grad()
    def merge_(self) -> None:
        if self.merged:
            return
        update = torch.matmul(self.lora_b, self.lora_a) * self.scaling
        self.base.weight.add_(update.to(device=self.base.weight.device, dtype=self.base.weight.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge_(self) -> None:
        if not self.merged:
            return
        update = torch.matmul(self.lora_b, self.lora_a) * self.scaling
        self.base.weight.sub_(update.to(device=self.base.weight.device, dtype=self.base.weight.dtype))
        self.merged = False


@dataclass
class TrainExample:
    input_ids: list[int]
    labels: list[int]

    @property
    def tokens(self) -> int:
        return len(self.input_ids)

    @property
    def label_tokens(self) -> int:
        return sum(1 for value in self.labels if value != -100)


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def install_lora(
    model: nn.Module,
    layers: list[int],
    rank: int,
    alpha: float,
    write_attention_o: bool,
    write_mlp: bool,
) -> list[LoraLinear]:
    decoder_layers = get_decoder_layers(model)
    wrappers: list[LoraLinear] = []
    for raw_idx in layers:
        idx = raw_idx if raw_idx >= 0 else len(decoder_layers) + raw_idx
        if idx < 0 or idx >= len(decoder_layers):
            raise IndexError(f"Layer index {raw_idx} resolved to {idx}, but model has {len(decoder_layers)} layers.")
        layer = decoder_layers[idx]
        if write_attention_o:
            module = get_attention_o_module(layer)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Expected attention o_proj to be nn.Linear, got {type(module)!r}")
            wrapper = LoraLinear(module, rank=rank, alpha=alpha)
            set_attention_o_module(layer, wrapper)
            wrappers.append(wrapper)
        if write_mlp:
            module = get_mlp_down_module(layer)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Expected MLP down_proj to be nn.Linear, got {type(module)!r}")
            wrapper = LoraLinear(module, rank=rank, alpha=alpha)
            set_mlp_down_module(layer, wrapper)
            wrappers.append(wrapper)
    return wrappers


def lora_parameters(wrappers: list[LoraLinear]) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for wrapper in wrappers:
        params.extend([wrapper.lora_a, wrapper.lora_b])
    return params


def merge_lora(wrappers: list[LoraLinear]) -> None:
    for wrapper in wrappers:
        wrapper.merge_()


def unmerge_lora(wrappers: list[LoraLinear]) -> None:
    for wrapper in wrappers:
        wrapper.unmerge_()


def make_example(tokenizer, prompt: str, answer: str, max_length: int) -> TrainExample:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    if len(input_ids) > max_length:
        trim = len(input_ids) - max_length
        input_ids = input_ids[trim:]
        labels = labels[trim:]
        if all(value == -100 for value in labels):
            raise ValueError("max_length truncated away every supervised answer token")
    return TrainExample(input_ids=input_ids, labels=labels)


def collate(examples: list[TrainExample], pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    width = max(example.tokens for example in examples)
    input_rows = []
    label_rows = []
    mask_rows = []
    for example in examples:
        pad = width - example.tokens
        input_rows.append([pad_id] * pad + example.input_ids)
        label_rows.append([-100] * pad + example.labels)
        mask_rows.append([0] * pad + [1] * example.tokens)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "labels": torch.tensor(label_rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long, device=device),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output", default="runs/minilang_lora_baseline")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lessons", type=int, default=4)
    parser.add_argument("--lesson-examples", type=int, default=8)
    parser.add_argument("--freeze-language-after", type=int, default=None)
    parser.add_argument("--train-source", choices=["trace", "lesson_examples"], default="trace")
    parser.add_argument("--trace-probes", type=int, default=4)
    parser.add_argument("--balanced-trace", action="store_true")
    parser.add_argument("--eval-questions", type=int, default=16)
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--write-attention-o", action="store_true")
    parser.add_argument("--no-write-mlp", dest="write_mlp", action="store_false", default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--target-correct", type=int, default=11)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--chat-template", dest="chat_template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", dest="chat_template", action="store_false")
    return parser.parse_args()


def build_training_examples(args: argparse.Namespace, tokenizer) -> list[TrainExample]:
    examples: list[TrainExample] = []
    build_trace_questions = build_balanced_questions if args.balanced_trace else build_questions
    for lesson_idx in range(args.lessons):
        language_idx = (
            min(lesson_idx, args.freeze_language_after)
            if args.freeze_language_after is not None
            else lesson_idx
        )
        if args.train_source == "trace":
            questions = build_trace_questions(
                args.trace_probes,
                args.seed + lesson_idx * 10_000,
                lesson_idx,
                "trace_translation",
                language_idx=language_idx,
            )
        else:
            questions = build_trace_questions(
                args.lesson_examples,
                args.seed + lesson_idx * 997,
                lesson_idx,
                "lesson_example",
                language_idx=language_idx,
            )
        for question in questions:
            prompt = format_prompt(tokenizer, question, context=None, use_chat_template=args.chat_template)
            examples.append(make_example(tokenizer, prompt, question.answer, args.max_length))
    return examples


def add_metrics(row: dict, prefix: str, metrics: dict) -> None:
    row[f"{prefix}_accuracy"] = metrics["accuracy"]
    row[f"{prefix}_correct"] = metrics["correct"]
    row[f"{prefix}_n"] = metrics["n"]
    row[f"{prefix}_mean_margin"] = metrics["mean_margin"]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"
    train_path = output_dir / "train_metrics.jsonl"
    details_path = output_dir / "eval_details.jsonl"
    for path in (metrics_path, train_path, details_path):
        if path.exists():
            path.unlink()

    started = time.time()
    print("Loading model...", flush=True)
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    counter = ForwardCounter(model).install()
    print("Model loaded; building task...", flush=True)

    lesson_texts = [
        render_lesson(
            idx,
            args.lesson_examples,
            args.seed,
            language_idx=min(idx, args.freeze_language_after)
            if args.freeze_language_after is not None
            else None,
        )
        for idx in range(args.lessons)
    ]
    final_lesson_idx = args.lessons - 1
    final_language_idx = (
        min(final_lesson_idx, args.freeze_language_after)
        if args.freeze_language_after is not None
        else final_lesson_idx
    )
    eval_questions = build_questions(
        args.eval_questions,
        args.seed + 91_000,
        final_lesson_idx,
        "heldout_translation",
        language_idx=final_language_idx,
    )
    full_context = "\n\n".join(lesson_texts)

    model.eval()
    print("Evaluating baseline/context before LoRA install...", flush=True)
    baseline = evaluate_mc(
        model,
        tokenizer,
        eval_questions,
        device,
        context=None,
        max_length=args.max_length,
        use_chat_template=args.chat_template,
    )
    context = evaluate_mc(
        model,
        tokenizer,
        eval_questions,
        device,
        context=full_context,
        max_length=args.max_length,
        use_chat_template=args.chat_template,
    )
    print("Installing LoRA adapters...", flush=True)
    wrappers = install_lora(
        model,
        args.layers,
        args.rank,
        args.alpha,
        write_attention_o=args.write_attention_o,
        write_mlp=args.write_mlp,
    )
    params = lora_parameters(wrappers)
    trainable_params = sum(param.numel() for param in params)
    total_lora_surfaces = len(wrappers)
    row = {
        "stage": "before_train",
        "seconds": time.time() - started,
        "forward_calls": counter.calls,
        "forward_tokens": counter.tokens,
        "trainable_params": trainable_params,
        "lora_surfaces": total_lora_surfaces,
    }
    add_metrics(row, "baseline", baseline)
    add_metrics(row, "context", context)
    append_jsonl(metrics_path, row)

    train_examples = build_training_examples(args, tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    train_token_count = 0
    train_label_token_count = 0
    reached_target = False

    model.train()
    print(f"Training {trainable_params} LoRA params on {len(train_examples)} examples...", flush=True)
    for step in range(1, args.max_steps + 1):
        batch_examples = random.sample(train_examples, k=min(args.batch_size, len(train_examples)))
        batch = collate(batch_examples, pad_id, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_token_count += int(batch["attention_mask"].sum().item())
        train_label_token_count += int((batch["labels"] != -100).sum().item())
        append_jsonl(
            train_path,
            {
                "stage": "train_step",
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "seconds": time.time() - started,
                "forward_calls": counter.calls,
                "forward_tokens": counter.tokens,
                "train_tokens": train_token_count,
                "train_label_tokens": train_label_token_count,
            },
        )
        if step % args.eval_every != 0 and step != args.max_steps:
            continue
        model.eval()
        merge_lora(wrappers)
        edited = evaluate_mc(
            model,
            tokenizer,
            eval_questions,
            device,
            context=None,
            max_length=args.max_length,
            use_chat_template=args.chat_template,
        )
        unmerge_lora(wrappers)
        eval_row = {
            "stage": "eval",
            "step": step,
            "seconds": time.time() - started,
            "forward_calls": counter.calls,
            "forward_tokens": counter.tokens,
            "train_tokens": train_token_count,
            "train_label_tokens": train_label_token_count,
            "train_forward_equiv_tokens_3x": train_token_count * 3,
        }
        add_metrics(eval_row, "baseline", baseline)
        add_metrics(eval_row, "context", context)
        add_metrics(eval_row, "edited", edited)
        eval_row["accuracy_delta"] = edited["accuracy"] - baseline["accuracy"]
        eval_row["internalization_ratio"] = (
            (edited["accuracy"] - baseline["accuracy"])
            / (context["accuracy"] - baseline["accuracy"] + 1e-12)
        )
        append_jsonl(metrics_path, eval_row)
        if edited["correct"] >= args.target_correct:
            reached_target = True
            break
        model.train()

    model.eval()
    merge_lora(wrappers)
    final = evaluate_mc(
        model,
        tokenizer,
        eval_questions,
        device,
        context=None,
        max_length=args.max_length,
        use_chat_template=args.chat_template,
    )
    unmerge_lora(wrappers)
    final_row = {
        "stage": "after_train",
        "target_reached": reached_target,
        "steps": step,
        "seconds": time.time() - started,
        "forward_calls": counter.calls,
        "forward_tokens": counter.tokens,
        "train_tokens": train_token_count,
        "train_label_tokens": train_label_token_count,
        "train_forward_equiv_tokens_3x": train_token_count * 3,
        "trainable_params": trainable_params,
        "lora_surfaces": total_lora_surfaces,
    }
    add_metrics(final_row, "baseline", baseline)
    add_metrics(final_row, "context", context)
    add_metrics(final_row, "edited", final)
    final_row["accuracy_delta"] = final["accuracy"] - baseline["accuracy"]
    final_row["internalization_ratio"] = (
        (final["accuracy"] - baseline["accuracy"])
        / (context["accuracy"] - baseline["accuracy"] + 1e-12)
    )
    append_jsonl(metrics_path, final_row)
    for idx, detail in enumerate(final["details"]):
        append_jsonl(details_path, {"stage": "final", "idx": idx, **detail})
    counter.uninstall()
    print(f"Wrote LoRA metrics to {metrics_path}")
    print(f"Wrote LoRA train metrics to {train_path}")


if __name__ == "__main__":
    main()

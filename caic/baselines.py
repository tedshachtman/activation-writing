"""Optional gradient baselines for CAIC runs."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import torch

from .synthetic import DomainSpec, QuestionRecord, format_prompt


@dataclass
class TrainConfig:
    steps: int = 20
    lr: float = 5e-5
    batch_size: int = 1
    max_length: int = 2048
    seed: int = 0


def set_trainable(model: Any, trainable: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(trainable)


def paper_text_examples(domain: DomainSpec) -> list[tuple[str, str]]:
    paper = domain.render_paper()
    return [(paper, "")]


def qa_examples(domain: DomainSpec, questions: list[QuestionRecord]) -> list[tuple[str, str]]:
    return [
        (format_prompt(record.question, paper=None), f" {record.answer_text}")
        for record in questions
    ]


def _prompt_completion_loss(
    model: Any,
    tokenizer: Any,
    batch: list[tuple[str, str]],
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    rows = []
    labels = []
    for prompt, completion in batch:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
        if len(full_ids) > max_length:
            full_ids = full_ids[-max_length:]
            prompt_ids = prompt_ids[-min(len(prompt_ids), max_length) :]
        label = [-100] * len(full_ids)
        start = min(len(prompt_ids), len(full_ids))
        for idx in range(start, len(full_ids)):
            label[idx] = full_ids[idx]
        if not completion:
            label = full_ids.copy()
        rows.append(torch.tensor(full_ids, dtype=torch.long))
        labels.append(torch.tensor(label, dtype=torch.long))

    input_ids = torch.nn.utils.rnn.pad_sequence(
        rows,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    ).to(device)
    label_ids = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100,
    ).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids, use_cache=False).loss


def train_prompt_completion(
    model: Any,
    tokenizer: Any,
    examples: list[tuple[str, str]],
    device: torch.device,
    config: TrainConfig,
) -> None:
    if not examples:
        return
    rng = random.Random(config.seed)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr)
    for _step in range(config.steps):
        batch = [rng.choice(examples) for _ in range(config.batch_size)]
        optimizer.zero_grad(set_to_none=True)
        loss = _prompt_completion_loss(model, tokenizer, batch, device, config.max_length)
        loss.backward()
        optimizer.step()
    model.eval()


def train_naive_text_baseline(
    model: Any,
    tokenizer: Any,
    domain: DomainSpec,
    device: torch.device,
    config: TrainConfig,
) -> None:
    set_trainable(model, True)
    train_prompt_completion(model, tokenizer, paper_text_examples(domain), device, config)


def prepare_qa_lora_model(
    model: Any,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> Any:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("PEFT is required for the qa_lora baseline. Install peft.") from exc

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def train_qa_lora_baseline(
    model: Any,
    tokenizer: Any,
    domain: DomainSpec,
    questions: list[QuestionRecord],
    device: torch.device,
    config: TrainConfig,
) -> None:
    train_prompt_completion(model, tokenizer, qa_examples(domain, questions), device, config)

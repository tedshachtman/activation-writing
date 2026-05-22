"""Evaluation and causal-patching utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .modeling import clear_active_slot_weights, patched_down_output, set_active_slot_weights_for_prompts
from .synthetic import QuestionRecord, format_prompt


YES_COMPLETION = " Yes"
NO_COMPLETION = " No"


def format_model_prompt(tokenizer: Any, prompt: str, use_chat_template: bool = False) -> str:
    if not use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def format_question_prompt(
    tokenizer: Any,
    question: str,
    paper: str | None = None,
    use_chat_template: bool = False,
) -> str:
    return format_model_prompt(tokenizer, format_prompt(question, paper=paper), use_chat_template)


@dataclass
class EvalResult:
    accuracy: float
    n: int
    correct: int
    mean_margin: float
    predictions: list[str]
    positive_accuracy: float = 0.0
    positive_n: int = 0
    positive_correct: int = 0
    positive_mean_margin: float = 0.0
    negative_accuracy: float = 0.0
    negative_n: int = 0
    negative_correct: int = 0
    negative_mean_margin: float = 0.0

    def to_dict(self, prefix: str) -> dict[str, float | int]:
        return {
            f"{prefix}_accuracy": self.accuracy,
            f"{prefix}_n": self.n,
            f"{prefix}_correct": self.correct,
            f"{prefix}_mean_margin": self.mean_margin,
            f"{prefix}_positive_accuracy": self.positive_accuracy,
            f"{prefix}_positive_n": self.positive_n,
            f"{prefix}_positive_correct": self.positive_correct,
            f"{prefix}_positive_mean_margin": self.positive_mean_margin,
            f"{prefix}_negative_accuracy": self.negative_accuracy,
            f"{prefix}_negative_n": self.negative_n,
            f"{prefix}_negative_correct": self.negative_correct,
            f"{prefix}_negative_mean_margin": self.negative_mean_margin,
        }


@dataclass
class QuestionEval:
    question: str
    answer: bool
    prediction: bool
    correct: bool
    margin: float
    yes_logprob: float
    no_logprob: float
    category: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": "Yes" if self.answer else "No",
            "prediction": "Yes" if self.prediction else "No",
            "correct": self.correct,
            "margin": self.margin,
            "yes_logprob": self.yes_logprob,
            "no_logprob": self.no_logprob,
            "category": self.category,
        }


@torch.no_grad()
def completion_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    completion: str,
    device: torch.device,
    max_length: int = 2048,
) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("Completion produced no additional tokens.")
    completion_start = len(prompt_ids)

    if len(full_ids) > max_length:
        trim = len(full_ids) - max_length
        full_ids = full_ids[trim:]
        completion_start = max(0, completion_start - trim)
    if completion_start <= 0:
        raise ValueError("Prompt was truncated past the completion boundary; increase max_length.")

    input_ids = torch.tensor([full_ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[0]
    log_probs = torch.log_softmax(logits[completion_start - 1 : len(full_ids) - 1], dim=-1)
    target = input_ids[0, completion_start:]
    return float(log_probs.gather(1, target.unsqueeze(1)).sum().cpu())


@torch.no_grad()
def yes_no_logprobs(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_length: int = 2048,
) -> tuple[float, float]:
    yes_ids = tokenizer.encode(YES_COMPLETION, add_special_tokens=False)
    no_ids = tokenizer.encode(NO_COMPLETION, add_special_tokens=False)
    if not yes_ids or not no_ids:
        raise ValueError("Tokenizer returned empty Yes/No completions.")
    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    tokens = {name: value.to(device) for name, value in tokens.items()}
    set_active_slot_weights_for_prompts(model, [prompt])
    try:
        outputs = model(**tokens, use_cache=False)
    finally:
        clear_active_slot_weights(model)
    log_probs = torch.log_softmax(outputs.logits[0, -1], dim=-1)
    return float(log_probs[yes_ids[0]].cpu()), float(log_probs[no_ids[0]].cpu())


def answer_margin(
    yes_logprob: float,
    no_logprob: float,
    answer: bool,
) -> float:
    return (yes_logprob - no_logprob) if answer else (no_logprob - yes_logprob)


@torch.no_grad()
def evaluate_yes_no(
    model: Any,
    tokenizer: Any,
    questions: list[QuestionRecord],
    device: torch.device,
    paper: str | None = None,
    max_length: int = 2048,
    use_chat_template: bool = False,
) -> EvalResult:
    correct = 0
    positive_correct = 0
    negative_correct = 0
    predictions: list[str] = []
    margins: list[float] = []
    positive_margins: list[float] = []
    negative_margins: list[float] = []
    for record in questions:
        prompt = format_question_prompt(tokenizer, record.question, paper=paper, use_chat_template=use_chat_template)
        yes_lp, no_lp = yes_no_logprobs(model, tokenizer, prompt, device, max_length=max_length)
        pred = yes_lp >= no_lp
        predictions.append("Yes" if pred else "No")
        is_correct = pred == record.answer
        correct += int(is_correct)
        margin = answer_margin(yes_lp, no_lp, record.answer)
        margins.append(margin)
        if record.answer:
            positive_correct += int(is_correct)
            positive_margins.append(margin)
        else:
            negative_correct += int(is_correct)
            negative_margins.append(margin)
    n = len(questions)
    positive_n = len(positive_margins)
    negative_n = len(negative_margins)
    return EvalResult(
        accuracy=correct / n if n else 0.0,
        n=n,
        correct=correct,
        mean_margin=sum(margins) / n if n else 0.0,
        predictions=predictions,
        positive_accuracy=positive_correct / positive_n if positive_n else 0.0,
        positive_n=positive_n,
        positive_correct=positive_correct,
        positive_mean_margin=sum(positive_margins) / positive_n if positive_n else 0.0,
        negative_accuracy=negative_correct / negative_n if negative_n else 0.0,
        negative_n=negative_n,
        negative_correct=negative_correct,
        negative_mean_margin=sum(negative_margins) / negative_n if negative_n else 0.0,
    )


@torch.no_grad()
def evaluate_yes_no_with_bias(
    model: Any,
    tokenizer: Any,
    questions: list[QuestionRecord],
    device: torch.device,
    yes_bias: float,
    paper: str | None = None,
    max_length: int = 2048,
    use_chat_template: bool = False,
) -> EvalResult:
    """Evaluate after adding a scalar bias to the Yes-vs-No decision margin."""

    correct = 0
    positive_correct = 0
    negative_correct = 0
    predictions: list[str] = []
    margins: list[float] = []
    positive_margins: list[float] = []
    negative_margins: list[float] = []
    for record in questions:
        prompt = format_question_prompt(tokenizer, record.question, paper=paper, use_chat_template=use_chat_template)
        yes_lp, no_lp = yes_no_logprobs(model, tokenizer, prompt, device, max_length=max_length)
        biased_yes_lp = yes_lp + yes_bias
        pred = biased_yes_lp >= no_lp
        predictions.append("Yes" if pred else "No")
        is_correct = pred == record.answer
        correct += int(is_correct)
        margin = answer_margin(biased_yes_lp, no_lp, record.answer)
        margins.append(margin)
        if record.answer:
            positive_correct += int(is_correct)
            positive_margins.append(margin)
        else:
            negative_correct += int(is_correct)
            negative_margins.append(margin)
    n = len(questions)
    positive_n = len(positive_margins)
    negative_n = len(negative_margins)
    return EvalResult(
        accuracy=correct / n if n else 0.0,
        n=n,
        correct=correct,
        mean_margin=sum(margins) / n if n else 0.0,
        predictions=predictions,
        positive_accuracy=positive_correct / positive_n if positive_n else 0.0,
        positive_n=positive_n,
        positive_correct=positive_correct,
        positive_mean_margin=sum(positive_margins) / positive_n if positive_n else 0.0,
        negative_accuracy=negative_correct / negative_n if negative_n else 0.0,
        negative_n=negative_n,
        negative_correct=negative_correct,
        negative_mean_margin=sum(negative_margins) / negative_n if negative_n else 0.0,
    )


@torch.no_grad()
def fit_scalar_yes_bias(
    model: Any,
    tokenizer: Any,
    questions: list[QuestionRecord],
    device: torch.device,
    paper: str | None = None,
    max_length: int = 2048,
    use_chat_template: bool = False,
    min_bias: float = -20.0,
    max_bias: float = 20.0,
    steps: int = 161,
) -> tuple[float, EvalResult]:
    """Fit a scalar Yes logit/logprob bias on a validation set."""

    if steps < 2:
        raise ValueError("steps must be at least 2")
    raw_margins: list[float] = []
    answers: list[bool] = []
    for record in questions:
        prompt = format_question_prompt(tokenizer, record.question, paper=paper, use_chat_template=use_chat_template)
        yes_lp, no_lp = yes_no_logprobs(model, tokenizer, prompt, device, max_length=max_length)
        raw_margins.append(yes_lp - no_lp)
        answers.append(record.answer)

    def result_for_bias(bias: float) -> EvalResult:
        correct = 0
        positive_correct = 0
        negative_correct = 0
        predictions: list[str] = []
        margins: list[float] = []
        positive_margins: list[float] = []
        negative_margins: list[float] = []
        for raw_margin, answer in zip(raw_margins, answers):
            biased_margin = raw_margin + bias
            pred = biased_margin >= 0.0
            predictions.append("Yes" if pred else "No")
            is_correct = pred == answer
            correct += int(is_correct)
            margin = biased_margin if answer else -biased_margin
            margins.append(margin)
            if answer:
                positive_correct += int(is_correct)
                positive_margins.append(margin)
            else:
                negative_correct += int(is_correct)
                negative_margins.append(margin)
        n = len(answers)
        positive_n = len(positive_margins)
        negative_n = len(negative_margins)
        return EvalResult(
            accuracy=correct / n if n else 0.0,
            n=n,
            correct=correct,
            mean_margin=sum(margins) / n if n else 0.0,
            predictions=predictions,
            positive_accuracy=positive_correct / positive_n if positive_n else 0.0,
            positive_n=positive_n,
            positive_correct=positive_correct,
            positive_mean_margin=sum(positive_margins) / positive_n if positive_n else 0.0,
            negative_accuracy=negative_correct / negative_n if negative_n else 0.0,
            negative_n=negative_n,
            negative_correct=negative_correct,
            negative_mean_margin=sum(negative_margins) / negative_n if negative_n else 0.0,
        )

    best_bias = 0.0
    best_result: EvalResult | None = None
    best_score = (-1.0, -1.0, -float("inf"), -float("inf"))
    for idx in range(steps):
        bias = min_bias + (max_bias - min_bias) * idx / (steps - 1)
        result = result_for_bias(bias)
        score = (result.accuracy, result.negative_accuracy, result.mean_margin, -abs(bias))
        if best_result is None:
            best_bias = bias
            best_result = result
            best_score = score
            continue
        if score > best_score:
            best_bias = bias
            best_result = result
            best_score = score
    assert best_result is not None
    return best_bias, best_result


@torch.no_grad()
def evaluate_yes_no_details(
    model: Any,
    tokenizer: Any,
    questions: list[QuestionRecord],
    device: torch.device,
    paper: str | None = None,
    max_length: int = 2048,
    use_chat_template: bool = False,
) -> list[QuestionEval]:
    rows: list[QuestionEval] = []
    for record in questions:
        prompt = format_question_prompt(tokenizer, record.question, paper=paper, use_chat_template=use_chat_template)
        yes_lp, no_lp = yes_no_logprobs(model, tokenizer, prompt, device, max_length=max_length)
        pred = yes_lp >= no_lp
        rows.append(
            QuestionEval(
                question=record.question,
                answer=record.answer,
                prediction=pred,
                correct=pred == record.answer,
                margin=answer_margin(yes_lp, no_lp, record.answer),
                yes_logprob=yes_lp,
                no_logprob=no_lp,
                category=record.category,
            )
        )
    return rows


@torch.no_grad()
def score_margin_for_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    answer: bool,
    device: torch.device,
    max_length: int = 2048,
) -> float:
    yes_lp, no_lp = yes_no_logprobs(model, tokenizer, prompt, device, max_length=max_length)
    return answer_margin(yes_lp, no_lp, answer)


@torch.no_grad()
def causal_patch_weights(
    model: Any,
    tokenizer: Any,
    questions: list[QuestionRecord],
    layer_indices: list[int],
    student_outputs: dict[int, torch.Tensor],
    content_deltas: dict[int, torch.Tensor],
    device: torch.device,
    max_length: int = 2048,
    floor: float = 0.05,
    use_chat_template: bool = False,
) -> dict[int, torch.Tensor]:
    """Weight examples by answer-margin improvement from final-token patching."""

    prompts = [
        format_question_prompt(tokenizer, record.question, paper=None, use_chat_template=use_chat_template)
        for record in questions
    ]
    base_margins = [
        score_margin_for_prompt(model, tokenizer, prompt, record.answer, device, max_length=max_length)
        for prompt, record in zip(prompts, questions)
    ]
    weights: dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        layer_weights: list[float] = []
        replacements = student_outputs[layer_idx] + content_deltas[layer_idx]
        for row, (prompt, record, base_margin) in enumerate(zip(prompts, questions, base_margins)):
            with patched_down_output(model, layer_idx, replacements[row : row + 1], device):
                patched_margin = score_margin_for_prompt(
                    model,
                    tokenizer,
                    prompt,
                    record.answer,
                    device,
                    max_length=max_length,
                )
            layer_weights.append(max(floor, patched_margin - base_margin + floor))
        weights[layer_idx] = torch.tensor(layer_weights, dtype=torch.float32)
    return weights


@torch.no_grad()
def yes_no_distributions(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    device: torch.device,
    max_length: int = 2048,
    use_chat_template: bool = False,
) -> torch.Tensor:
    rows = []
    for prompt in prompts:
        prompt = format_model_prompt(tokenizer, prompt, use_chat_template=use_chat_template)
        yes_lp, no_lp = yes_no_logprobs(model, tokenizer, prompt, device, max_length=max_length)
        rows.append(torch.softmax(torch.tensor([yes_lp, no_lp], dtype=torch.float32), dim=0))
    return torch.stack(rows, dim=0) if rows else torch.empty(0, 2)


def categorical_kl(p: torch.Tensor, q: torch.Tensor) -> float:
    if p.numel() == 0:
        return 0.0
    eps = 1e-8
    p = torch.clamp(p.float(), min=eps)
    q = torch.clamp(q.float(), min=eps)
    kl = torch.sum(p * (torch.log(p) - torch.log(q)), dim=1)
    return float(kl.mean().cpu())


def internalization_ratio(aic_acc: float, no_doc_acc: float, context_acc: float) -> float:
    denom = context_acc - no_doc_acc
    if denom <= 1e-8:
        return 0.0
    return (aic_acc - no_doc_acc) / denom

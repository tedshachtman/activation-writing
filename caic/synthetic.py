"""Executable synthetic mini-papers for CAIC.

The benchmark intentionally starts with toy domains whose labels come from a
small interpreter. The rendered paper is prose, but held-out answers are
generated from the hidden DSL so evaluation is not self-graded by the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from typing import Iterable, Literal


RuleKind = Literal[
    "follow",
    "require",
    "forbid_adjacent",
    "count_at_least",
    "position",
    "must_include",
    "forbid_any",
]


OPERATORS = [
    "glim",
    "sorn",
    "vek",
    "nalo",
    "brin",
    "tav",
    "mure",
    "zhek",
    "lumo",
    "prax",
    "keld",
    "fesh",
    "dravo",
    "cairn",
    "yelb",
    "rusk",
    "mav",
    "tovin",
    "esk",
    "halm",
]

MARKS = [
    "red",
    "hollow",
    "silver",
    "amber",
    "striped",
    "dull",
    "bright",
    "violet",
    "open",
    "closed",
    "salted",
    "woven",
    "quiet",
    "sharp",
    "flat",
    "green",
    "blue",
    "round",
    "split",
    "clear",
]

DOMAINS = [
    "Blorbian",
    "Varnic",
    "Oltar",
    "Nemril",
    "Cazhen",
    "Drellic",
    "Suvant",
    "Yomari",
    "Pellin",
    "Kavren",
]


@dataclass(frozen=True)
class Predicate:
    """A partial match over chain atoms."""

    op: str | None = None
    mark: str | None = None

    def matches(self, atom: "Atom") -> bool:
        if self.op is not None and atom.op != self.op:
            return False
        if self.mark is not None and atom.mark != self.mark:
            return False
        return True

    def describe(self) -> str:
        if self.mark and self.op:
            return f"{self.mark} {self.op}"
        if self.mark:
            return f"{self.mark} item"
        if self.op:
            return f"{self.op} item"
        return "any item"


@dataclass(frozen=True)
class Atom:
    op: str
    mark: str

    def render(self) -> str:
        return f"{self.mark} {self.op}"


@dataclass(frozen=True)
class RuleSpec:
    kind: RuleKind
    p: Predicate
    q: Predicate | None = None
    count: int | None = None
    exception: Predicate | None = None

    def applies_exception(self, chain: list[Atom]) -> bool:
        return self.exception is not None and any(self.exception.matches(atom) for atom in chain)

    def evaluate(self, chain: list[Atom]) -> tuple[bool, str]:
        if self.applies_exception(chain):
            return True, "exception"

        if self.kind == "follow":
            assert self.q is not None
            for idx, atom in enumerate(chain):
                if self.p.matches(atom):
                    if idx + 1 >= len(chain) or not self.q.matches(chain[idx + 1]):
                        return False, f"{self.p.describe()} was not followed by {self.q.describe()}"
            return True, "ok"

        if self.kind == "require":
            assert self.q is not None
            if any(self.p.matches(atom) for atom in chain):
                if not any(self.q.matches(atom) for atom in chain):
                    return False, f"{self.p.describe()} appeared without {self.q.describe()}"
            return True, "ok"

        if self.kind == "forbid_adjacent":
            assert self.q is not None
            for left, right in zip(chain, chain[1:]):
                if self.p.matches(left) and self.q.matches(right):
                    return False, f"{self.p.describe()} touched {self.q.describe()}"
            return True, "ok"

        if self.kind == "count_at_least":
            assert self.q is not None
            assert self.count is not None
            if any(self.p.matches(atom) for atom in chain):
                hits = sum(1 for atom in chain if self.q.matches(atom))
                if hits < self.count:
                    return False, f"only {hits} {self.q.describe()} entries were present"
            return True, "ok"

        if self.kind == "position":
            assert self.q is not None
            if chain and self.p.matches(chain[-1]) and not self.q.matches(chain[0]):
                return False, f"final {self.p.describe()} did not start with {self.q.describe()}"
            return True, "ok"

        if self.kind == "must_include":
            if not any(self.p.matches(atom) for atom in chain):
                return False, f"missing required {self.p.describe()}"
            return True, "ok"

        if self.kind == "forbid_any":
            if any(self.p.matches(atom) for atom in chain):
                return False, f"forbidden {self.p.describe()} appeared"
            return True, "ok"

        raise ValueError(f"Unknown rule kind: {self.kind}")

    def render(self) -> str:
        suffix = ""
        if self.exception is not None:
            suffix = f" unless the chain contains a {self.exception.describe()}"

        if self.kind == "follow":
            assert self.q is not None
            return f"Every {self.p.describe()} must be immediately followed by a {self.q.describe()}{suffix}."
        if self.kind == "require":
            assert self.q is not None
            return f"If a chain contains a {self.p.describe()}, it must also contain a {self.q.describe()}{suffix}."
        if self.kind == "forbid_adjacent":
            assert self.q is not None
            return f"A {self.p.describe()} may not be immediately followed by a {self.q.describe()}{suffix}."
        if self.kind == "count_at_least":
            assert self.q is not None
            assert self.count is not None
            return f"If a chain contains a {self.p.describe()}, it must include at least {self.count} {self.q.describe()} entries{suffix}."
        if self.kind == "position":
            assert self.q is not None
            return f"If the final item is a {self.p.describe()}, the first item must be a {self.q.describe()}{suffix}."
        if self.kind == "must_include":
            return f"A valid chain must contain at least one {self.p.describe()}{suffix}."
        if self.kind == "forbid_any":
            return f"A valid chain must not contain any {self.p.describe()}{suffix}."
        raise ValueError(f"Unknown rule kind: {self.kind}")


@dataclass
class QuestionRecord:
    question: str
    answer: bool
    chain: list[Atom]
    category: str

    @property
    def answer_text(self) -> str:
        return "Yes" if self.answer else "No"

    def to_dict(self) -> dict:
        out = asdict(self)
        out["answer_text"] = self.answer_text
        return out


@dataclass
class DomainSpec:
    domain_id: str
    title: str
    operators: list[str]
    marks: list[str]
    rules: list[RuleSpec]
    examples: list[QuestionRecord]

    def validate(self, chain: list[Atom]) -> tuple[bool, list[str]]:
        failures: list[str] = []
        for rule in self.rules:
            ok, reason = rule.evaluate(chain)
            if not ok:
                failures.append(reason)
        return not failures, failures

    def render_chain(self, chain: list[Atom]) -> str:
        return ", ".join(atom.render() for atom in chain)

    def render_paper(self) -> str:
        rule_lines = "\n".join(f"{idx + 1}. {rule.render()}" for idx, rule in enumerate(self.rules))
        examples = "\n".join(
            f"- {self.render_chain(ex.chain)} -> {'valid' if ex.answer else 'invalid'}"
            for ex in self.examples
        )
        return (
            f"{self.title}: Chain Validity Notes\n\n"
            f"This note defines a synthetic rule system named {self.title}. "
            f"A chain is a comma-separated sequence of marked operators. "
            f"The operators are {', '.join(self.operators)}. "
            f"The marks are {', '.join(self.marks)}.\n\n"
            f"Rules:\n{rule_lines}\n\n"
            f"Worked examples:\n{examples}\n"
        )

    def make_question(self, chain: list[Atom], category: str, variant: int = 0) -> QuestionRecord:
        valid, _failures = self.validate(chain)
        rendered = self.render_chain(chain)
        phrasings = [
            f"In {self.title}, is this chain valid: {rendered}?",
            f"Using the {self.title} rules, should the chain be accepted: {rendered}?",
            f"Does {rendered} satisfy the {self.title} validity system?",
            f"Would {self.title} classify this chain as valid: {rendered}?",
        ]
        return QuestionRecord(
            question=phrasings[variant % len(phrasings)],
            answer=valid,
            chain=chain,
            category=category,
        )

    def sample_chain(self, rng: random.Random, min_len: int = 3, max_len: int = 6) -> list[Atom]:
        return [
            Atom(op=rng.choice(self.operators), mark=rng.choice(self.marks))
            for _ in range(rng.randint(min_len, max_len))
        ]

    def balanced_questions(
        self,
        count: int,
        rng: random.Random,
        category: str,
        max_attempts: int = 5000,
    ) -> list[QuestionRecord]:
        questions: list[QuestionRecord] = []
        target_true = count // 2
        target_false = count - target_true
        seen: set[str] = set()
        attempts = 0
        while len(questions) < count and attempts < max_attempts:
            attempts += 1
            chain = self.sample_chain(rng)
            rendered = self.render_chain(chain)
            if rendered in seen:
                continue
            seen.add(rendered)
            q = self.make_question(chain, category=category, variant=attempts)
            true_count = sum(1 for item in questions if item.answer)
            false_count = len(questions) - true_count
            if q.answer and true_count >= target_true:
                continue
            if not q.answer and false_count >= target_false:
                continue
            questions.append(q)

        while len(questions) < count:
            chain = self.sample_chain(rng)
            questions.append(self.make_question(chain, category=category, variant=len(questions)))
        return questions

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _predicate_pool(operators: list[str], marks: list[str]) -> list[Predicate]:
    preds: list[Predicate] = []
    preds.extend(Predicate(op=op) for op in operators)
    preds.extend(Predicate(mark=mark) for mark in marks)
    preds.extend(Predicate(op=op, mark=mark) for op in operators for mark in marks)
    return preds


def _sample_rules(rng: random.Random, operators: list[str], marks: list[str]) -> list[RuleSpec]:
    preds = _predicate_pool(operators, marks)
    rules: list[RuleSpec] = []

    p = Predicate(op=operators[0], mark=marks[0])
    q = Predicate(op=operators[1], mark=marks[1])
    exc = Predicate(op=operators[2], mark=marks[2])
    rules.append(RuleSpec(kind="follow", p=p, q=q, exception=exc))

    p = Predicate(op=operators[2])
    q = Predicate(mark=marks[0])
    rules.append(RuleSpec(kind="require", p=p, q=q))

    p = Predicate(mark=marks[1])
    q = Predicate(op=operators[0])
    rules.append(RuleSpec(kind="forbid_adjacent", p=p, q=q, exception=Predicate(mark=marks[2])))

    p = Predicate(mark=marks[2])
    q = Predicate(op=operators[1])
    rules.append(RuleSpec(kind="count_at_least", p=p, q=q, count=2))

    p = rng.choice(preds)
    q = rng.choice([pred for pred in preds if pred != p])
    rules.append(RuleSpec(kind="position", p=p, q=q))

    rng.shuffle(rules)
    return rules


def _sample_easy_rules(_rng: random.Random, operators: list[str], marks: list[str]) -> list[RuleSpec]:
    required = Predicate(op=operators[0], mark=marks[0])
    return [
        RuleSpec(kind="must_include", p=required),
    ]


def _sample_medium_rules(_rng: random.Random, operators: list[str], marks: list[str]) -> list[RuleSpec]:
    required = Predicate(op=operators[0], mark=marks[0])
    forbidden = Predicate(op=operators[1], mark=marks[1])
    exception = Predicate(op=operators[2], mark=marks[2])
    return [
        RuleSpec(kind="must_include", p=required),
        RuleSpec(kind="forbid_any", p=forbidden, exception=exception),
    ]


def generate_domain(seed: int, index: int, difficulty: str = "standard") -> DomainSpec:
    rng = random.Random((seed + 1) * 1009 + index * 9176)
    if difficulty == "easy":
        operators = rng.sample(OPERATORS, 2)
        marks = rng.sample(MARKS, 2)
    elif difficulty == "medium":
        operators = rng.sample(OPERATORS, 3)
        marks = rng.sample(MARKS, 3)
    elif difficulty == "standard":
        operators = rng.sample(OPERATORS, 3)
        marks = rng.sample(MARKS, 3)
    else:
        raise ValueError(f"Unknown domain difficulty: {difficulty}")
    title = f"{rng.choice(DOMAINS)}-{index:03d}"
    if difficulty == "easy":
        rules = _sample_easy_rules(rng, operators, marks)
    elif difficulty == "medium":
        rules = _sample_medium_rules(rng, operators, marks)
    else:
        rules = _sample_rules(rng, operators, marks)

    placeholder = DomainSpec(
        domain_id=f"domain-{index:03d}",
        title=title,
        operators=operators,
        marks=marks,
        rules=rules,
        examples=[],
    )
    examples = placeholder.balanced_questions(5, rng, category="paper_example")
    return DomainSpec(
        domain_id=placeholder.domain_id,
        title=title,
        operators=operators,
        marks=marks,
        rules=rules,
        examples=examples,
    )


def generate_domains(count: int, seed: int = 0, difficulty: str = "standard") -> list[DomainSpec]:
    return [generate_domain(seed, idx, difficulty=difficulty) for idx in range(count)]


def predicate_from_dict(row: dict) -> Predicate:
    return Predicate(op=row.get("op"), mark=row.get("mark"))


def atom_from_dict(row: dict) -> Atom:
    return Atom(op=row["op"], mark=row["mark"])


def rule_from_dict(row: dict) -> RuleSpec:
    q = row.get("q")
    exception = row.get("exception")
    return RuleSpec(
        kind=row["kind"],
        p=predicate_from_dict(row["p"]),
        q=predicate_from_dict(q) if q is not None else None,
        count=row.get("count"),
        exception=predicate_from_dict(exception) if exception is not None else None,
    )


def question_from_dict(row: dict) -> QuestionRecord:
    return QuestionRecord(
        question=row["question"],
        answer=bool(row["answer"]),
        chain=[atom_from_dict(atom) for atom in row.get("chain", [])],
        category=row.get("category", "loaded"),
    )


def domain_from_dict(row: dict) -> DomainSpec:
    return DomainSpec(
        domain_id=row["domain_id"],
        title=row["title"],
        operators=list(row["operators"]),
        marks=list(row["marks"]),
        rules=[rule_from_dict(rule) for rule in row["rules"]],
        examples=[question_from_dict(question) for question in row.get("examples", [])],
    )


def make_candidate_probes(domain: DomainSpec, count: int, seed: int) -> list[QuestionRecord]:
    rng = random.Random(seed)
    return domain.balanced_questions(count, rng, category="candidate_probe")


def make_eval_questions(domain: DomainSpec, count: int, seed: int) -> list[QuestionRecord]:
    rng = random.Random(seed)
    return domain.balanced_questions(count, rng, category="heldout_eval")


def make_inverse_questions(domain: DomainSpec, count: int, seed: int) -> list[QuestionRecord]:
    """Ask rejection-polarity questions to catch simple Yes/No steering."""

    rng = random.Random(seed)
    base = domain.balanced_questions(count, rng, category="inverse_source")
    questions: list[QuestionRecord] = []
    phrasings = [
        f"In {{title}}, should this chain be rejected: {{chain}}?",
        f"Using the {{title}} rules, is this chain invalid: {{chain}}?",
        f"Does {{chain}} fail the {{title}} validity system?",
        f"Would {{title}} classify this chain as not valid: {{chain}}?",
    ]
    for idx, record in enumerate(base):
        rendered = domain.render_chain(record.chain)
        questions.append(
            QuestionRecord(
                question=phrasings[idx % len(phrasings)].format(title=domain.title, chain=rendered),
                answer=not record.answer,
                chain=record.chain,
                category="inverse_polarity",
            )
        )
    return questions


def make_minimal_pair_questions(
    domain: DomainSpec,
    pair_count: int,
    seed: int,
    max_attempts: int = 8000,
) -> list[QuestionRecord]:
    """Generate one-edit chain pairs whose labels differ.

    The pair keeps chain length and most surface tokens fixed, so a method that
    only learns a broad "formal chains often say No" prior should struggle.
    """

    rng = random.Random(seed)
    questions: list[QuestionRecord] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    while len(questions) < pair_count * 2 and attempts < max_attempts:
        attempts += 1
        chain = domain.sample_chain(rng)
        base_valid, _ = domain.validate(chain)
        if not chain:
            continue
        pos = rng.randrange(len(chain))
        replacements: list[Atom] = []
        for op in domain.operators:
            replacements.append(Atom(op=op, mark=chain[pos].mark))
        for mark in domain.marks:
            replacements.append(Atom(op=chain[pos].op, mark=mark))
        rng.shuffle(replacements)
        for atom in replacements:
            if atom == chain[pos]:
                continue
            mutated = list(chain)
            mutated[pos] = atom
            mutated_valid, _ = domain.validate(mutated)
            if mutated_valid == base_valid:
                continue
            left = domain.render_chain(chain)
            right = domain.render_chain(mutated)
            pair_key = tuple(sorted((left, right)))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            pair_idx = len(questions) // 2
            questions.append(domain.make_question(chain, category="minimal_pair", variant=pair_idx * 2))
            questions.append(domain.make_question(mutated, category="minimal_pair", variant=pair_idx * 2 + 1))
            break

    return questions[: pair_count * 2]


def make_near_collision_domain(domain: DomainSpec, seed: int = 0) -> DomainSpec:
    """Create a same-vocabulary rival domain with a conflicting rule table."""

    rng = random.Random(seed)
    operators = list(domain.operators)
    marks = list(domain.marks)
    if len(operators) < 2 or len(marks) < 2:
        raise ValueError("Near-collision domains require at least two operators and two marks.")

    required = Predicate(op=operators[-1], mark=marks[-1])
    forbidden = Predicate(op=operators[0], mark=marks[0])
    rules = [RuleSpec(kind="must_include", p=required)]
    if len(operators) >= 3 and len(marks) >= 3:
        rules.append(
            RuleSpec(
                kind="forbid_any",
                p=forbidden,
                exception=Predicate(op=operators[-1], mark=marks[0]),
            )
        )

    placeholder = DomainSpec(
        domain_id=f"{domain.domain_id}-near-collision",
        title=f"{domain.title}-Rival",
        operators=operators,
        marks=marks,
        rules=rules,
        examples=[],
    )
    examples = placeholder.balanced_questions(5, rng, category="near_collision_example")
    return DomainSpec(
        domain_id=placeholder.domain_id,
        title=placeholder.title,
        operators=operators,
        marks=marks,
        rules=rules,
        examples=examples,
    )


def make_near_collision_questions(
    domain: DomainSpec,
    count: int,
    seed: int,
    include_rival_prompts: bool = False,
    max_attempts: int = 12000,
) -> list[QuestionRecord]:
    """Generate chains whose label flips under a same-vocabulary rival domain.

    By default, returned questions are phrased only in the original domain.
    That makes the bucket fair for single-paper evaluation while ensuring each
    chain is rule-discriminative rather than a generic formal-chain example.
    Set `include_rival_prompts=True` for explicit same-chain/opposite-domain
    routing diagnostics.
    """

    rng = random.Random(seed)
    rival = make_near_collision_domain(domain, seed=seed + 101)
    questions: list[QuestionRecord] = []
    seen: set[str] = set()
    attempts = 0
    while len(questions) < count and attempts < max_attempts:
        attempts += 1
        chain = domain.sample_chain(rng)
        rendered = domain.render_chain(chain)
        if rendered in seen:
            continue
        domain_valid, _ = domain.validate(chain)
        rival_valid, _ = rival.validate(chain)
        if domain_valid == rival_valid:
            continue
        seen.add(rendered)
        questions.append(domain.make_question(chain, category="near_collision", variant=attempts))
        if include_rival_prompts and len(questions) < count:
            questions.append(rival.make_question(chain, category="near_collision_rival", variant=attempts))

    return questions[:count]


def make_gauntlet_questions(
    domain: DomainSpec,
    count_per_bucket: int,
    seed: int,
    include_near_collision: bool = False,
) -> dict[str, list[QuestionRecord]]:
    """Create falsification buckets for rule learning vs answer-prior repair."""

    buckets = {
        "ordinary": make_eval_questions(domain, count_per_bucket, seed + 11),
        "minimal_pair": make_minimal_pair_questions(domain, count_per_bucket // 2, seed + 23),
        "inverse_polarity": make_inverse_questions(domain, count_per_bucket, seed + 37),
    }
    if include_near_collision:
        buckets["near_collision"] = make_near_collision_questions(domain, count_per_bucket, seed + 41)
    return buckets


def format_prompt(question: str, paper: str | None = None) -> str:
    if paper:
        return (
            f"{paper}\n"
            "Use the preceding paper when answering the next question.\n\n"
            f"Question: {question}\n"
            "Answer with Yes or No only.\n"
            "Answer:"
        )
    return (
        f"Question: {question}\n"
        "Answer with Yes or No only.\n"
        "Answer:"
    )


def make_null_document(seed: int, approx_words: int) -> str:
    rng = random.Random(seed)
    null_ops = rng.sample([op for op in OPERATORS if op], 3)
    null_marks = rng.sample([mark for mark in MARKS if mark], 3)
    sentences = [
        "This control note describes a synthetic filing system with no chain-validity rules.",
        f"The listed labels are {', '.join(null_ops)} and the listed tags are {', '.join(null_marks)}.",
        "Items may be sorted alphabetically, copied into ledgers, or ignored by auditors.",
        "No statement in this note defines whether an operator chain is valid or invalid.",
    ]
    words: list[str] = []
    while len(words) < approx_words:
        words.extend(rng.choice(sentences).split())
    return " ".join(words[:approx_words])


def negative_guard_prompts(seed: int, count: int) -> list[str]:
    rng = random.Random(seed)
    templates = [
        "Question: Is ordinary matrix multiplication commutative? Answer with Yes or No only.\nAnswer:",
        "Question: Does water normally freeze at room temperature? Answer with Yes or No only.\nAnswer:",
        "Question: Is Paris the capital of France? Answer with Yes or No only.\nAnswer:",
        "Question: If a story mentions a glim in ordinary English, must it follow any synthetic chain rule? Answer with Yes or No only.\nAnswer:",
        "Question: Does the word hollow always imply a formal operator exception? Answer with Yes or No only.\nAnswer:",
        "Question: Can a person answer a yes-or-no question without using a made-up algebra? Answer with Yes or No only.\nAnswer:",
    ]
    prompts = [templates[idx % len(templates)] for idx in range(count)]
    rng.shuffle(prompts)
    return prompts


def general_guard_questions() -> list[QuestionRecord]:
    """Small labeled sentinel set for concept-smear checks."""

    rows = [
        ("Is ordinary matrix multiplication commutative?", False),
        ("Does water normally freeze at room temperature?", False),
        ("Is Paris the capital of France?", True),
        ("If a story mentions a glim in ordinary English, must it follow any synthetic chain rule?", False),
        ("Does the word hollow always imply a formal operator exception?", False),
        ("Can a person answer a yes-or-no question without using a made-up algebra?", True),
        ("Is the Moon usually described as a star rather than a natural satellite?", False),
        ("Does two plus two equal four?", True),
    ]
    return [
        QuestionRecord(question=question, answer=answer, chain=[], category="general_guard")
        for question, answer in rows
    ]


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

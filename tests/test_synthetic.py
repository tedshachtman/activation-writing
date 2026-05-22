import json
from argparse import Namespace

import torch

from caic.experiment import build_candidate_pool, load_domain_rows, select_write_questions
from caic.synthetic import (
    domain_from_dict,
    generate_domain,
    make_candidate_probes,
    make_eval_questions,
    make_gauntlet_questions,
    make_inverse_questions,
    make_minimal_pair_questions,
    make_near_collision_domain,
    make_near_collision_questions,
)


def test_domain_generation_is_executable_and_balanced():
    domain = generate_domain(seed=7, index=0)
    probes = make_candidate_probes(domain, count=20, seed=11)
    assert len(probes) == 20
    assert any(record.answer for record in probes)
    assert any(not record.answer for record in probes)
    for record in probes:
        valid, _ = domain.validate(record.chain)
        assert valid == record.answer


def test_paper_contains_rules_and_examples():
    domain = generate_domain(seed=3, index=2)
    paper = domain.render_paper()
    assert domain.title in paper
    assert "Rules:" in paper
    assert "Worked examples:" in paper
    eval_questions = make_eval_questions(domain, count=8, seed=9)
    assert len(eval_questions) == 8


def test_medium_domain_has_two_rules_and_exception():
    domain = generate_domain(seed=5, index=1, difficulty="medium")
    assert len(domain.rules) == 2
    assert any(rule.exception is not None for rule in domain.rules)
    probes = make_candidate_probes(domain, count=20, seed=12)
    assert any(record.answer for record in probes)
    assert any(not record.answer for record in probes)


def test_domain_json_roundtrip_preserves_labels():
    domain = generate_domain(seed=3, index=2, difficulty="medium")
    restored = domain_from_dict(json.loads(domain.to_json()))
    questions = make_eval_questions(restored, count=12, seed=9)
    for record in questions:
        valid, _ = restored.validate(record.chain)
        assert valid == record.answer


def test_gauntlet_inverse_questions_flip_label_polarity():
    domain = generate_domain(seed=7, index=0, difficulty="easy")
    questions = make_inverse_questions(domain, count=10, seed=13)
    assert len(questions) == 10
    for record in questions:
        valid, _ = domain.validate(record.chain)
        assert record.answer == (not valid)
        assert record.category == "inverse_polarity"


def test_minimal_pairs_have_flipped_labels_and_same_length():
    domain = generate_domain(seed=7, index=0, difficulty="easy")
    questions = make_minimal_pair_questions(domain, pair_count=4, seed=17)
    assert len(questions) >= 2
    assert len(questions) % 2 == 0
    for left, right in zip(questions[::2], questions[1::2]):
        assert len(left.chain) == len(right.chain)
        assert left.answer != right.answer
        diffs = sum(a != b for a, b in zip(left.chain, right.chain))
        assert diffs == 1


def test_gauntlet_questions_are_bucketed():
    domain = generate_domain(seed=7, index=0, difficulty="easy")
    buckets = make_gauntlet_questions(domain, count_per_bucket=8, seed=19)
    assert set(buckets) == {"ordinary", "minimal_pair", "inverse_polarity"}
    assert buckets["ordinary"]
    assert buckets["minimal_pair"]
    assert buckets["inverse_polarity"]


def test_near_collision_domain_reuses_vocabulary_but_changes_rules():
    domain = generate_domain(seed=7, index=0, difficulty="medium")
    rival = make_near_collision_domain(domain, seed=23)
    assert rival.title != domain.title
    assert rival.operators == domain.operators
    assert rival.marks == domain.marks
    assert rival.rules != domain.rules


def test_near_collision_questions_flip_under_rival_domain():
    domain = generate_domain(seed=7, index=0, difficulty="easy")
    rival = make_near_collision_domain(domain, seed=40 + 101)
    questions = make_near_collision_questions(domain, count=8, seed=40)
    assert questions
    for record in questions:
        domain_valid, _ = domain.validate(record.chain)
        rival_valid, _ = rival.validate(record.chain)
        assert record.answer == domain_valid
        assert domain_valid != rival_valid
        assert record.category == "near_collision"


def test_gauntlet_can_include_near_collision_bucket():
    domain = generate_domain(seed=7, index=0, difficulty="easy")
    buckets = make_gauntlet_questions(domain, count_per_bucket=8, seed=19, include_near_collision=True)
    assert set(buckets) == {"ordinary", "minimal_pair", "inverse_polarity", "near_collision"}
    assert buckets["near_collision"]


def test_load_domain_rows_reuses_saved_eval_questions(tmp_path):
    domain = generate_domain(seed=9, index=0, difficulty="easy")
    eval_questions = make_eval_questions(domain, count=4, seed=44)
    path = tmp_path / "domains.jsonl"
    path.write_text(
        json.dumps(
            {
                "domain": json.loads(domain.to_json()),
                "paper": domain.render_paper(),
                "eval_questions": [record.to_dict() for record in eval_questions],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded_domains, loaded_eval_sets = load_domain_rows(path, 1)
    assert loaded_domains[0].title == domain.title
    assert [record.question for record in loaded_eval_sets[0]] == [
        record.question for record in eval_questions
    ]


def test_build_candidate_pool_can_include_falsification_buckets():
    domain = generate_domain(seed=3, index=0, difficulty="medium")
    args = Namespace(
        seed=3,
        candidate_probes=6,
        candidate_inverse_probes=6,
        candidate_minimal_pair_probes=4,
        candidate_near_collision_probes=4,
    )
    pool = build_candidate_pool(domain, args, paper_idx=0)
    categories = {record.category for record in pool}
    assert len(pool) >= 16
    assert "candidate_probe" in categories
    assert "inverse_polarity" in categories
    assert "minimal_pair" in categories
    assert "near_collision" in categories


def test_select_write_questions_can_force_label_balance():
    domain = generate_domain(seed=3, index=0, difficulty="easy")
    questions = make_candidate_probes(domain, count=20, seed=17)
    keys = torch.randn(len(questions), 4)
    weights = torch.tensor([100.0 if not record.answer else 1.0 for record in questions])
    selected = select_write_questions(
        keys,
        questions,
        k=10,
        weights=weights,
        ridge=1.0,
        balanced=True,
        positive_fraction=0.5,
    )
    positives = sum(1 for idx in selected if questions[idx].answer)
    negatives = len(selected) - positives
    assert positives == 5
    assert negatives == 5

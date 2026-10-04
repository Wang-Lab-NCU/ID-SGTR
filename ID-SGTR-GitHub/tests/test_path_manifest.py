import json
import tempfile
import unittest
from pathlib import Path

from knowledge_graph.experiments.path_manifest import (
    FoldManifest,
    ManifestValidationError,
    PathStep,
    load_jsonl,
    save_jsonl,
)


def _step(
    step: int,
    source: str,
    target: str,
    chunk_id: str,
    *,
    branch: str = "main",
    reverse: bool = False,
) -> PathStep:
    return PathStep(
        step=step,
        branch=branch,
        source_entity=source,
        relation="connects to",
        target_entity=target,
        supporting_chunk_id=chunk_id,
        edge_score=0.75,
        query_score=0.6,
        canonical_source_entity=target if reverse else source,
        canonical_target_entity=source if reverse else target,
        traversal_direction="reverse" if reverse else "forward",
    )


def _manifest(**overrides) -> FoldManifest:
    values = {
        "query_id": "hotpot:7",
        "dataset": "hotpot",
        "context_id": "7",
        "question": "Who connects Alpha to Gamma?",
        "intent_type": "Reasoning",
        "intent_strategy": "connected_core",
        "anchor_entities": ("Alpha",),
        "path_policy": "directed_beam",
        "path_steps": (
            _step(1, "Alpha", "Beta", "c1"),
            _step(2, "Beta", "Gamma", "c2"),
        ),
        "candidate_chunk_ids": ("c1", "c2", "c3"),
        "selected_chunk_ids": ("c1", "c2", "c3"),
        "core_chunk_ids": ("c1", "c2"),
        "peripheral_chunk_ids": ("c3",),
        "score_order_chunk_ids": ("c1", "c3", "c2"),
        "path_order_chunk_ids": ("c1", "c2", "c3"),
        "chunk_scores": (("c3", 0.5), ("c1", 0.9), ("c2", 0.4)),
        "budget": 3,
        "token_budget": 768,
        "source_token_budget": 600,
        "trace_token_budget": 96,
        "total_evidence_token_budget": 768,
        "foldable": True,
        "fallback_reason": "",
        "path_confidence": 0.75,
        "path_continuous": True,
        "branch_complete": True,
        "anchor_margin": 0.12,
        "anchor_source": "lexical_exact",
        "llm_reranker_used": False,
    }
    values.update(overrides)
    return FoldManifest(**values)


class PathManifestTests(unittest.TestCase):
    def test_reverse_step_preserves_canonical_and_traversal_directions(self):
        step = _step(1, "Beta", "Alpha", "c1", reverse=True)
        self.assertEqual(step.canonical_source_entity, "Alpha")
        self.assertEqual(step.canonical_target_entity, "Beta")
        self.assertEqual(step.traversal_source_entity, "Beta")
        self.assertEqual(step.traversal_target_entity, "Alpha")
        step.validate()

    def test_dict_round_trip_and_sha_are_stable(self):
        manifest = _manifest()
        restored = FoldManifest.from_dict(manifest.to_dict())
        self.assertEqual(restored, manifest)
        self.assertEqual(restored.canonical_json(), manifest.canonical_json())
        self.assertEqual(restored.sha256, manifest.sha256)
        self.assertEqual(len(manifest.sha256), 64)

        # Whitespace/key ordering in an intermediate JSON document cannot
        # affect the canonical payload or digest.
        shuffled = json.loads(json.dumps(manifest.to_dict(), indent=4))
        self.assertEqual(FoldManifest.from_dict(shuffled).sha256, manifest.sha256)

    def test_jsonl_round_trip_is_canonical_and_detects_duplicate_queries(self):
        first = _manifest()
        second = _manifest(query_id="hotpot:8", context_id="8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            save_jsonl(path, [first, second])
            loaded = load_jsonl(path)
            self.assertEqual(loaded, [first, second])
            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines()[0],
                first.canonical_json(),
            )
            with self.assertRaisesRegex(ManifestValidationError, "duplicate query_id"):
                save_jsonl(path, [first, first])

    def test_budget_and_id_invariants(self):
        with self.assertRaisesRegex(ManifestValidationError, "exceeds budget"):
            _manifest(budget=2).validate()
        with self.assertRaisesRegex(ManifestValidationError, "subset of candidate"):
            _manifest(selected_chunk_ids=("c1", "missing")).validate()
        with self.assertRaisesRegex(ManifestValidationError, "duplicate IDs"):
            _manifest(candidate_chunk_ids=("c1", "c1", "c2")).validate()

    def test_factorial_orders_are_selected_permutations(self):
        _manifest().validate()
        with self.assertRaisesRegex(ManifestValidationError, "score_order.*permutation"):
            _manifest(score_order_chunk_ids=("c1", "c2", "missing")).validate()
        with self.assertRaisesRegex(ManifestValidationError, "path_order.*permutation"):
            _manifest(path_order_chunk_ids=("c1", "c2")).validate()
        with self.assertRaisesRegex(ManifestValidationError, "descending"):
            _manifest(score_order_chunk_ids=("c1", "c2", "c3")).validate()

    def test_chunk_scores_are_stable_and_cover_selected(self):
        manifest = _manifest(
            chunk_scores={"c3": 0.5, "c1": 0.9, "c2": 0.4},
        )
        self.assertEqual(
            manifest.chunk_scores,
            (("c1", 0.9), ("c2", 0.4), ("c3", 0.5)),
        )
        with self.assertRaisesRegex(ManifestValidationError, "cover every selected"):
            _manifest(chunk_scores=(("c1", 0.9), ("c2", 0.4))).validate()

    def test_source_and_trace_token_budgets_are_bounded_by_total(self):
        self.assertEqual(FoldManifest(query_id="x").source_token_budget, 1400)
        self.assertEqual(FoldManifest(query_id="x").trace_token_budget, 96)
        self.assertEqual(FoldManifest(query_id="x").total_evidence_token_budget, 1536)
        with self.assertRaisesRegex(ManifestValidationError, "exceeds"):
            _manifest(
                source_token_budget=700,
                trace_token_budget=100,
                total_evidence_token_budget=768,
            ).validate()

    def test_step_must_align_with_selected_core_chunk(self):
        with self.assertRaisesRegex(ManifestValidationError, "selected core chunk"):
            _manifest(
                selected_chunk_ids=("c2", "c3"),
                core_chunk_ids=("c2",),
                peripheral_chunk_ids=("c3",),
            ).validate()

    def test_each_branch_is_checked_independently(self):
        comparative = _manifest(
            intent_type="Comparative",
            path_policy="multi_branch_comparison",
            path_steps=(
                _step(1, "Alpha", "A1", "c1", branch="left"),
                _step(1, "Beta", "B1", "c2", branch="right"),
            ),
            core_chunk_ids=("c1", "c2"),
            peripheral_chunk_ids=("c3",),
            path_confidence=0.75,
        )
        comparative.validate()

        broken = _manifest(
            path_steps=(
                _step(1, "Alpha", "Beta", "c1"),
                _step(2, "Wrong", "Gamma", "c2"),
            ),
        )
        with self.assertRaisesRegex(ManifestValidationError, "discontinuous"):
            broken.validate()

    def test_step_numbering_and_canonical_direction_are_validated(self):
        bad_numbering = _manifest(
            path_steps=(
                _step(1, "Alpha", "Beta", "c1"),
                _step(3, "Beta", "Gamma", "c2"),
            ),
        )
        with self.assertRaisesRegex(ManifestValidationError, "contiguous"):
            bad_numbering.validate()

        bad_direction = PathStep(
            step=1,
            branch="main",
            source_entity="Alpha",
            relation="connects",
            target_entity="Beta",
            supporting_chunk_id="c1",
            canonical_source_entity="Beta",
            canonical_target_entity="Alpha",
            traversal_direction="forward",
        )
        with self.assertRaisesRegex(ManifestValidationError, "canonical endpoints"):
            bad_direction.validate()

    def test_non_foldable_manifest_records_reason_and_may_leave_budget_unused(self):
        fallback = FoldManifest(
            query_id="musique:9",
            dataset="musique",
            context_id="9",
            question="Where?",
            intent_type="Reasoning",
            path_policy="graph_naive_fallback",
            candidate_chunk_ids=("a", "b", "c"),
            selected_chunk_ids=("a", "b"),
            core_chunk_ids=("a", "b"),
            budget=3,
            token_budget=512,
            foldable=False,
            fallback_reason="no_continuous_path",
            fallback_to_graph_naive=True,
            path_confidence=0.1,
            path_continuous=False,
        )
        fallback.validate()
        self.assertEqual(fallback.selected_count, 2)
        self.assertEqual(fallback.unused_budget, 1)
        self.assertTrue(fallback.fallback_to_graph_naive)

    def test_foldable_and_fallback_flags_are_consistent(self):
        with self.assertRaisesRegex(ManifestValidationError, "cannot fall back"):
            _manifest(fallback_to_graph_naive=True).validate()
        with self.assertRaisesRegex(ManifestValidationError, "branch_complete"):
            _manifest(branch_complete=False).validate()

    def test_current_versions_and_derived_path_fields_are_enforced(self):
        with self.assertRaisesRegex(ManifestValidationError, "manifest_version"):
            _manifest(manifest_version="1.0").validate()
        with self.assertRaisesRegex(ManifestValidationError, "implementation_version"):
            _manifest(implementation_version="legacy").validate()
        with self.assertRaisesRegex(ManifestValidationError, "path_confidence"):
            _manifest(path_confidence=0.1).validate()
        with self.assertRaisesRegex(ManifestValidationError, "first occurrence"):
            _manifest(
                core_chunk_ids=("c2", "c1"),
                path_order_chunk_ids=("c2", "c1", "c3"),
            ).validate()


if __name__ == "__main__":
    unittest.main()

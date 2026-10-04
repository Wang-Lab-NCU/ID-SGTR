import copy
import unittest

import pandas as pd

from knowledge_graph.experiments.runtime_manifest import (
    RUNTIME_IMPLEMENTATION_VERSION,
    RuntimeManifestBuilder,
)
from knowledge_graph.experiments.question_aligned_runtime import (
    CONSERVATIVE_QUESTION_ALIGNED_IMPLEMENTATION_VERSION,
    FACTORIAL_CLEAN_IMPLEMENTATION_VERSION,
    QUESTION_ALIGNED_IMPLEMENTATION_VERSION,
    QuestionAlignedRuntimeManifestBuilder,
)
from knowledge_graph.experiments.telemetry import (
    QueryTelemetry,
    bind_telemetry,
    record_runtime_execution,
    record_runtime_step,
)


def _chunks():
    return pd.DataFrame([
        {"context_id": "ctx", "chunk_id": "a", "text": "Alpha reaches Bridge."},
        {"context_id": "ctx", "chunk_id": "b", "text": "Bridge reaches Citadel."},
        {"context_id": "ctx", "chunk_id": "c", "text": "Peripheral source."},
    ])


def _graph():
    return pd.DataFrame([
        {
            "context_id": "ctx", "chunk_id": "a",
            "node_1": "Alpha", "edge": "reaches", "node_2": "Bridge",
        },
        {
            "context_id": "ctx", "chunk_id": "b",
            "node_1": "Bridge", "edge": "located in", "node_2": "Citadel",
        },
        {
            "context_id": "ctx", "chunk_id": "c",
            "node_1": "Other", "edge": "located in", "node_2": "Citadel",
        },
    ])


def _candidate(chunk_id, score, hop, position, triple):
    return {
        "chunk_id": chunk_id,
        "score": score,
        "hop": hop,
        "raw_path_position": position,
        "path_position": hop * 1000 + position,
        "triple": triple,
        "topology_trace": [{
            "hop": hop,
            "path_position": position,
            "triple": triple,
        }],
        "is_structural": True,
    }


def _runtime_step(hop, *edges):
    return {
        "hop": hop,
        "active_nodes": [],
        "relevant_nodes": [edge[2] for edge in edges],
        "next_nodes": [edge[2] for edge in edges],
        "chosen_edges": [{
            "source_entity": edge[0],
            "relation": edge[1],
            "target_entity": edge[2],
            "chunk_ids": list(edge[3]),
            "path_position": position,
        } for position, edge in enumerate(edges, start=1)],
    }


def _row():
    return {
        "query_id": "dataset:0",
        "context_id": "ctx",
        "question": "Where is the place reached from Alpha located?",
        "strategy": "(Reasoning, P=0.9) -> Agent-Hop-2",
        "candidate_evidence": [
            _candidate("a", 0.8, 1, 1, "Alpha --[reaches]--> Bridge"),
            _candidate("b", 0.7, 2, 1, "Bridge --[located in]--> Citadel"),
            {"chunk_id": "c", "score": 0.6, "topology_trace": []},
        ],
        "retrieved_evidence": ["a", "b", "c"],
        "runtime_path_steps": [
            _runtime_step(1, ("Alpha", "reaches", "Bridge", ("a",))),
            _runtime_step(2, ("Bridge", "located in", "Citadel", ("b",))),
        ],
        "gold_answer": "Citadel",
        "gold_evidence": ["a", "b"],
        "pred_answer": "anything",
    }


class RuntimeManifestTests(unittest.TestCase):
    def setUp(self):
        self.builder = RuntimeManifestBuilder(
            _chunks(), _graph(), dataset="hotpot", budget=3,
        )

    def test_freezes_continuous_runtime_trace(self):
        manifest = self.builder.build(_row())
        self.assertTrue(manifest.foldable)
        self.assertEqual(manifest.implementation_version, RUNTIME_IMPLEMENTATION_VERSION)
        self.assertEqual(manifest.core_chunk_ids, ("a", "b"))
        self.assertEqual(manifest.peripheral_chunk_ids, ("c",))
        self.assertEqual(manifest.path_order_chunk_ids, ("a", "b", "c"))
        self.assertEqual(set(manifest.score_order_chunk_ids), {"a", "b", "c"})
        self.assertEqual(manifest.path_steps[0].target_entity, "Bridge")
        self.assertEqual(manifest.path_steps[1].source_entity, "Bridge")

    def test_path_order_is_a_real_factorial_intervention(self):
        row = _row()
        row["candidate_evidence"][0]["score"] = 0.4
        row["candidate_evidence"][1]["score"] = 0.3
        row["candidate_evidence"][2]["score"] = 0.9
        manifest = self.builder.build(row)

        self.assertEqual(manifest.score_order_chunk_ids, ("c", "a", "b"))
        self.assertEqual(manifest.path_order_chunk_ids, ("a", "b", "c"))
        self.assertEqual(
            set(manifest.score_order_chunk_ids),
            set(manifest.path_order_chunk_ids),
        )

    def test_gold_and_prediction_are_non_inputs(self):
        left = _row()
        right = copy.deepcopy(left)
        right["gold_answer"] = "adversarial"
        right["gold_evidence"] = ["c"]
        right["pred_answer"] = "different"
        self.assertEqual(
            self.builder.build(left).canonical_json(),
            self.builder.build(right).canonical_json(),
        )

    def test_parallel_multi_chunk_trace_becomes_independent_branches(self):
        row = _row()
        row["candidate_evidence"][1] = _candidate(
            "b", 0.7, 2, 1, "Other --[located in]--> Citadel"
        )
        row["runtime_path_steps"][1] = _runtime_step(
            2, ("Other", "located in", "Citadel", ("c",)),
        )
        manifest = self.builder.build(row)
        self.assertTrue(manifest.foldable)
        self.assertEqual(
            manifest.path_policy, "runtime_executed_frontier_branch_forest"
        )
        self.assertEqual({step.branch for step in manifest.path_steps}, {"A", "B"})
        self.assertEqual(len(manifest.anchor_entities), len(set(manifest.anchor_entities)))
        self.assertEqual(set(manifest.selected_chunk_ids), {"a", "b", "c"})

    def test_wrong_runtime_chunk_is_remapped_to_selected_graph_source(self):
        row = _row()
        row["candidate_evidence"][0] = _candidate(
            "c", 0.8, 1, 1, "Alpha --[reaches]--> Bridge"
        )
        row["runtime_path_steps"][0] = _runtime_step(
            1, ("Alpha", "reaches", "Bridge", ("c", "a")),
        )
        manifest = self.builder.build(row)
        self.assertTrue(manifest.foldable)
        self.assertEqual(manifest.path_steps[0].supporting_chunk_id, "a")

    def test_reverse_runtime_traversal_preserves_graph_canonical_direction(self):
        row = _row()
        row["question"] = "What can be reached backwards from Bridge?"
        row["candidate_evidence"] = [
            _candidate("c", 0.8, 1, 1, "Bridge --[reaches]--> Alpha"),
        ]
        row["retrieved_evidence"] = ["a", "c"]
        row["runtime_path_steps"] = [
            _runtime_step(1, ("Bridge", "reaches", "Alpha", ("a", "c"))),
        ]
        manifest = self.builder.build(row)
        self.assertTrue(manifest.foldable)
        step = manifest.path_steps[0]
        self.assertEqual(step.supporting_chunk_id, "a")
        self.assertEqual(step.traversal_direction, "reverse")
        self.assertEqual(step.canonical_source_entity, "Alpha")
        self.assertEqual(step.canonical_target_entity, "Bridge")

    def test_frame_requires_runtime_telemetry_columns(self):
        with self.assertRaisesRegex(ValueError, "candidate_evidence"):
            self.builder.build_frame(pd.DataFrame([{
                "query_id": "x", "question": "q", "context_id": "ctx",
                "strategy": "Reasoning", "retrieved_evidence": ["a"],
            }]))

    def test_unselected_candidate_trace_is_never_frozen(self):
        row = _row()
        row["candidate_evidence"].append(
            _candidate("c", 0.99, 1, 1, "Other --[located in]--> Citadel")
        )
        manifest = self.builder.build(row)
        rendered_relations = {
            (step.source_entity, step.relation, step.target_entity)
            for step in manifest.path_steps
        }
        self.assertNotIn(
            ("Other", "located in", "Citadel"), rendered_relations,
        )

    def test_runtime_step_telemetry_records_only_supplied_chosen_edges(self):
        telemetry = QueryTelemetry("q")
        with bind_telemetry(telemetry):
            record_runtime_step(
                2,
                ["Alpha"],
                ["Bridge"],
                ["Bridge"],
                [{
                    "u": "Alpha", "rel": "reaches", "v": "Bridge",
                    "chunk_ids": ["a", "a"], "path_position": 4,
                }],
            )
        self.assertEqual(len(telemetry.runtime_path_steps), 1)
        step = telemetry.runtime_path_steps[0]
        self.assertEqual(step["hop"], 2)
        self.assertEqual(step["chosen_edges"][0]["chunk_ids"], ["a"])

    def test_executed_frontier_recovers_unparsed_runtime_choice(self):
        row = _row()
        row["runtime_path_steps"] = [
            {
                "hop": 1, "active_nodes": ["Alpha"],
                "relevant_nodes": [], "next_nodes": [], "chosen_edges": [],
                "executed_nodes": ["Bridge"],
                "executed_edges": _runtime_step(
                    1, ("Alpha", "reaches", "Bridge", ("a",))
                )["chosen_edges"],
            },
            {
                "hop": 2, "active_nodes": ["Bridge"],
                "relevant_nodes": [], "next_nodes": [], "chosen_edges": [],
                "executed_nodes": ["Citadel"],
                "executed_edges": _runtime_step(
                    2, ("Bridge", "located in", "Citadel", ("b",))
                )["chosen_edges"],
            },
        ]
        manifest = self.builder.build(row)
        self.assertTrue(manifest.foldable)
        self.assertEqual(manifest.core_chunk_ids, ("a", "b"))

    def test_runtime_execution_updates_only_matching_hop(self):
        telemetry = QueryTelemetry("q")
        with bind_telemetry(telemetry):
            record_runtime_step(1, ["Alpha"], [], [], [])
            record_runtime_execution(1, ["Bridge"], [{
                "u": "Alpha", "rel": "reaches", "v": "Bridge",
                "chunk_ids": ["a"], "path_position": 2,
            }])
        self.assertEqual(len(telemetry.runtime_path_steps), 1)
        step = telemetry.runtime_path_steps[0]
        self.assertEqual(step["executed_nodes"], ["Bridge"])
        self.assertEqual(step["executed_edges"][0]["chunk_ids"], ["a"])

    def test_question_aligned_selector_replaces_irrelevant_final_window(self):
        row = _row()
        row["retrieved_evidence"] = ["c"]
        row["candidate_evidence"][2]["score"] = 0.99
        builder = QuestionAlignedRuntimeManifestBuilder(
            _chunks(), _graph(), dataset="hotpot", budget=2,
        )

        manifest = builder.build(row)

        self.assertTrue(manifest.foldable)
        self.assertEqual(
            manifest.implementation_version,
            QUESTION_ALIGNED_IMPLEMENTATION_VERSION,
        )
        self.assertEqual(manifest.core_chunk_ids, ("a", "b"))
        self.assertEqual(set(manifest.selected_chunk_ids), {"a", "b"})
        self.assertNotIn("c", manifest.selected_chunk_ids)
        self.assertEqual(
            manifest.path_policy,
            "question_aligned_relation_slots",
        )

    def test_question_aligned_selector_is_gold_independent(self):
        left = _row()
        right = copy.deepcopy(left)
        right["gold_answer"] = "adversarial"
        right["gold_evidence"] = ["c"]
        right["pred_answer"] = "different"
        builder = QuestionAlignedRuntimeManifestBuilder(
            _chunks(), _graph(), dataset="hotpot", budget=2,
        )
        self.assertEqual(
            builder.build(left).canonical_json(),
            builder.build(right).canonical_json(),
        )

    def test_question_aligned_failure_preserves_runtime_evidence_set(self):
        row = _row()
        row["question"] = "What happened yesterday?"
        row["retrieved_evidence"] = ["a", "c"]
        row["candidate_evidence"][0]["score"] = 0.10
        row["candidate_evidence"][2]["score"] = 0.99
        builder = QuestionAlignedRuntimeManifestBuilder(
            _chunks(), _graph(), dataset="hotpot", budget=2,
        )

        manifest = builder.build(row)

        self.assertFalse(manifest.foldable)
        self.assertTrue(manifest.fallback_to_graph_naive)
        self.assertEqual(
            manifest.fallback_reason,
            "question_aligned_no_complete_path",
        )
        self.assertEqual(set(manifest.selected_chunk_ids), {"a", "c"})
        # Graph-Naive order remains deterministic score order.
        self.assertEqual(manifest.score_order_chunk_ids, ("c", "a"))
        self.assertEqual(manifest.path_order_chunk_ids, ("c", "a"))
        self.assertTrue(
            set(manifest.selected_chunk_ids).issubset(
                manifest.candidate_chunk_ids,
            )
        )

    def test_conservative_selector_rejects_multiple_replacements(self):
        row = _row()
        row["retrieved_evidence"] = ["c"]
        builder = QuestionAlignedRuntimeManifestBuilder(
            _chunks(), _graph(), dataset="hotpot", budget=2,
            conservative_gate=True,
        )

        manifest = builder.build(row)

        self.assertFalse(manifest.foldable)
        self.assertEqual(
            manifest.implementation_version,
            CONSERVATIVE_QUESTION_ALIGNED_IMPLEMENTATION_VERSION,
        )
        self.assertEqual(
            manifest.fallback_reason,
            "conservative_multiple_replacements",
        )
        self.assertEqual(set(manifest.selected_chunk_ids), {"c"})

    def test_conservative_selector_accepts_one_noncore_replacement(self):
        row = _row()
        row["retrieved_evidence"] = ["a", "c"]
        row["candidate_evidence"][2]["triple"] = ""
        row["candidate_evidence"][2]["topology_trace"] = []
        builder = QuestionAlignedRuntimeManifestBuilder(
            _chunks(), _graph(), dataset="hotpot", budget=2,
            conservative_gate=True,
        )

        manifest = builder.build(row)

        self.assertTrue(manifest.foldable)
        self.assertEqual(set(manifest.selected_chunk_ids), {"a", "b"})
        self.assertEqual(
            manifest.implementation_version,
            CONSERVATIVE_QUESTION_ALIGNED_IMPLEMENTATION_VERSION,
        )

    def test_factorial_clean_selector_preserves_semantic_score_order(self):
        row = _row()
        row["retrieved_evidence"] = ["a", "b"]
        baseline = RuntimeManifestBuilder(
            _chunks(), _graph(), dataset="hotpot", budget=2,
        ).build(row)
        builder = QuestionAlignedRuntimeManifestBuilder(
            _chunks(), _graph(), dataset="hotpot", budget=2,
            conservative_gate=True,
            factorial_clean=True,
        )

        manifest = builder.build(row)

        self.assertTrue(manifest.foldable)
        self.assertEqual(
            manifest.score_order_chunk_ids,
            baseline.score_order_chunk_ids,
        )
        self.assertEqual(
            manifest.implementation_version,
            FACTORIAL_CLEAN_IMPLEMENTATION_VERSION,
        )


if __name__ == "__main__":
    unittest.main()

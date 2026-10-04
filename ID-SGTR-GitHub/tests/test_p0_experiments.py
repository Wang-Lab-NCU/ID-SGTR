import unittest
import json
import tempfile
from pathlib import Path

import pandas as pd

from knowledge_graph.experiments.evidence import EvidenceAssembler, EvidenceItem
from knowledge_graph.experiments.metrics import evaluate_predictions, paired_bootstrap
from knowledge_graph.experiments.telemetry import (
    QueryTelemetry,
    TrackedChatModel,
    bind_telemetry,
    record_evidence,
)
from knowledge_graph.experiments.gold import annotate_gold_evidence
from knowledge_graph.experiments.controlled import ControlledAblationRunner


class _Response:
    content = "Final Answer: yes"
    usage_metadata = {"input_tokens": 8, "output_tokens": 3}


class _Model:
    def invoke(self, value):
        return _Response()


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            EvidenceItem("a", "A text", 0.2, 2, "A -> B"),
            EvidenceItem("b", "B text", 0.9, 1, "B -> C"),
            EvidenceItem("a", "duplicate", 0.1, 3, "duplicate"),
        ]

    def test_variants_share_deduplicated_budget(self):
        assembler = EvidenceAssembler(budget=2, random_seed=7)
        topology = assembler.select(self.items, "topology_folding", query_id="q")
        score = assembler.select(self.items, "source_score", query_id="q")
        self.assertEqual({item.chunk_id for item in topology}, {"a", "b"})
        self.assertEqual({item.chunk_id for item in score}, {"a", "b"})
        self.assertEqual([item.chunk_id for item in topology], ["b", "a"])
        self.assertEqual([item.chunk_id for item in score], ["b", "a"])

    def test_random_order_is_reproducible(self):
        assembler = EvidenceAssembler(budget=2, random_seed=7)
        first = assembler.select(self.items, "source_random", query_id="q")
        second = assembler.select(self.items, "source_random", query_id="q")
        self.assertEqual(first, second)

    def test_shared_budget_is_frozen_by_relevance_before_reordering(self):
        items = [
            EvidenceItem("h1a", "h1a", 0.9, 1, "a -> b", hop=1),
            EvidenceItem("h1b", "h1b", 0.8, 2, "b -> c", hop=1),
            EvidenceItem("h2", "h2", 0.2, 1, "c -> d", hop=2),
            EvidenceItem("h3", "h3", 0.1, 1, "d -> e", hop=3),
        ]
        assembler = EvidenceAssembler(budget=3, random_seed=7)
        variants = ["triple_only", "source_score", "source_random", "topology_folding"]
        selected = [assembler.select(items, variant, query_id="q") for variant in variants]
        expected = {"h1a", "h1b", "h2"}
        self.assertTrue(all({item.chunk_id for item in group} == expected for group in selected))

    def test_late_distractor_does_not_evict_more_relevant_evidence(self):
        items = [
            EvidenceItem("gold", "gold", 0.9, 3, "A -> B", hop=1),
            EvidenceItem("near", "near", 0.8, 1, "B -> C", hop=2),
            EvidenceItem("late", "late", 0.1, 1, "X -> Y", hop=4),
        ]
        selected = EvidenceAssembler(budget=2).select(items, "topology_folding")
        self.assertEqual({item.chunk_id for item in selected}, {"gold", "near"})

    def test_cross_hop_structural_backing_beats_one_hop_distractors(self):
        items = [
            EvidenceItem("noise", "noise", 0.9, 1, "X -> Y", hop=1),
            EvidenceItem("bridge", "bridge", 0.3, 1, "A -> B", hop=2),
            EvidenceItem("bridge", "bridge", 0.3, 1, "B -> C", hop=3),
        ]
        selected = EvidenceAssembler(
            budget=1, topology_support_weight=0.7
        ).select(items, "topology_folding")
        self.assertEqual([item.chunk_id for item in selected], ["bridge"])

    def test_weighted_support_matches_shared_smoke_selection_rule(self):
        items = [
            EvidenceItem("semantic", "", 0.465, 1, "A -> B", hop=1),
            EvidenceItem(
                "bridge", "", 0.316, 1, "B -> C", hop=2,
                topology_trace=((2, 1, "B -> C"), (3, 1, "C -> D"), (4, 1, "D -> E")),
            ),
            EvidenceItem("noise", "", 0.446, 2, "X -> Y", hop=1),
        ]
        selected = EvidenceAssembler(
            budget=2, topology_support_weight=0.08
        ).select(items, "topology_folding")
        self.assertEqual({item.chunk_id for item in selected}, {"semantic", "bridge"})

    def test_duplicate_source_merges_cross_hop_topology_trace(self):
        items = [
            EvidenceItem("a", "same source", 0.8, 1, "A -> B", hop=1),
            EvidenceItem("a", "same source", 0.8, 2, "B -> C", hop=2),
        ]
        rendered, selected = EvidenceAssembler(budget=1).render(
            items, "topology_folding"
        )
        self.assertEqual(selected, ["a"])
        self.assertIn("[Hop 1 Path 1] A -> B", rendered[0])
        self.assertIn("[Hop 2 Path 2] B -> C", rendered[0])
        self.assertEqual(rendered[0].count("[Source a]"), 1)


class MetricTests(unittest.TestCase):
    def test_f1_is_not_below_em(self):
        summary, _ = evaluate_predictions([
            {"prediction": "The Eiffel Tower", "gold": "Eiffel Tower"},
            {"prediction": "Paris, France", "gold": "Paris"},
        ])
        self.assertGreaterEqual(summary["f1"], summary["em"])

    def test_paired_bootstrap_is_deterministic(self):
        first = paired_bootstrap([1, 1, 0], [0, 0, 0], samples=100, seed=1)
        second = paired_bootstrap([1, 1, 0], [0, 0, 0], samples=100, seed=1)
        self.assertEqual(first, second)


class TelemetryTests(unittest.TestCase):
    def test_call_roles_and_usage_are_counted(self):
        telemetry = QueryTelemetry("q")
        TrackedChatModel(_Model(), telemetry, "answer").invoke("prompt")
        TrackedChatModel(_Model(), telemetry, "auxiliary").invoke("prompt")
        self.assertEqual(telemetry.total_llm_calls, 2)
        self.assertEqual(telemetry.answer_calls, 1)
        self.assertEqual(telemetry.auxiliary_calls, 1)
        self.assertEqual(telemetry.input_tokens, 16)
        self.assertFalse(telemetry.token_count_estimated)

    def test_final_prompt_evidence_is_separate_from_access_union(self):
        telemetry = QueryTelemetry("q")
        with bind_telemetry(telemetry):
            record_evidence(["stage0"])
            record_evidence(["hop1", "hop2"])
        self.assertEqual(telemetry.retrieved_evidence, ["hop1", "hop2"])
        self.assertEqual(telemetry.accessed_evidence, ["stage0", "hop1", "hop2"])


class GoldEvidenceTests(unittest.TestCase):
    def test_hotpot_titles_map_to_chunk_ids(self):
        raw = [{"question": "Who?", "supporting_facts": [["Doc A", 0], ["Doc B", 1]]}]
        subset = pd.DataFrame([{"question": "Who?", "context_id": "7"}])
        chunks = pd.DataFrame([
            {"context_id": "7", "chunk_id": "a", "title": "Doc A"},
            {"context_id": "7", "chunk_id": "b", "title": "Doc B"},
        ])
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.json"
            raw_path.write_text(json.dumps(raw), encoding="utf-8")
            annotated = annotate_gold_evidence(
                subset, chunks, dataset="hotpot", raw_path=raw_path
            )
        self.assertEqual(annotated.loc[0, "gold_evidence"], ["a", "b"])
        self.assertTrue(annotated.loc[0, "gold_mapping_complete"])


class ControlledReplayTests(unittest.TestCase):
    def test_variants_replay_the_same_candidate_manifest(self):
        chunks = pd.DataFrame([
            {"chunk_id": "a", "text": "A text"},
            {"chunk_id": "b", "text": "B text"},
            {"chunk_id": "c", "text": "C text"},
        ])
        row = {
            "query_id": "q",
            "question": "Who?",
            "candidate_evidence": str([
                {
                    "chunk_id": "a", "score": 0.2,
                    "path_position": 1002, "triple": "A -> B",
                    "topology_trace": [
                        {"hop": 1, "path_position": 2, "triple": "A -> B"},
                        {"hop": 1, "path_position": 3, "triple": "B -> C"},
                    ],
                },
                {"chunk_id": "b", "score": 0.9, "path_position": 1001, "triple": "B -> C"},
                {"chunk_id": "c", "score": 0.1, "path_position": 2001, "triple": "C -> D"},
            ]),
        }
        runner = ControlledAblationRunner(_Model(), chunks, budget=2)
        replay_items = runner._items(row, "topology_folding")
        item_a = next(item for item in replay_items if item.chunk_id == "a")
        self.assertEqual(len(item_a.topology_trace), 2)
        topology = runner.run_row(row, "topology_folding")
        score = runner.run_row(row, "source_score")
        self.assertEqual(set(topology["retrieved_evidence"]), {"a", "b"})
        self.assertEqual(set(score["retrieved_evidence"]), {"a", "b"})
        self.assertEqual(topology["selected_evidence_set"], score["selected_evidence_set"])
        self.assertEqual(topology["fixed_candidate_count"], score["fixed_candidate_count"])
        self.assertEqual(topology["answer_calls"], 1)

    def test_binary_answer_sentence_is_canonicalized(self):
        class _VerboseYesResponse:
            content = "Final Answer: Yes, both entities are American."
            usage_metadata = {"input_tokens": 8, "output_tokens": 6}

        class _VerboseYesModel:
            def invoke(self, value):
                return _VerboseYesResponse()

        chunks = pd.DataFrame([{"chunk_id": "a", "text": "A text"}])
        row = {
            "query_id": "q",
            "question": "Are they the same?",
            "candidate_evidence": str([
                {"chunk_id": "a", "score": 1.0, "hop": 1, "path_position": 1}
            ]),
        }
        result = ControlledAblationRunner(
            _VerboseYesModel(), chunks, budget=1
        ).run_row(row, "source_score")
        self.assertEqual(result["pred_answer"], "yes")


if __name__ == "__main__":
    unittest.main()

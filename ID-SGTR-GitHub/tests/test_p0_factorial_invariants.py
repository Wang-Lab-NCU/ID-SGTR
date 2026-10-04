import inspect
import unittest
from unittest.mock import patch

import pandas as pd

from knowledge_graph.experiments.folding_renderers import FoldingVariant
from knowledge_graph.experiments.local_controlled import LocalControlledV2Runner
from knowledge_graph.experiments.path_manifest import FoldManifest, PathStep
from knowledge_graph.experiments.run_p0 import build_parser


class _Response:
    def __init__(self, content, reasoning=""):
        self.content = content
        self.additional_kwargs = {"reasoning": reasoning} if reasoning else {}
        self.response_metadata = {}
        self.usage_metadata = {"input_tokens": 8, "output_tokens": 3}


class _Reasoner:
    def invoke(self, value):
        return _Response("Final Answer: Citadel", "Alpha reaches Citadel.")


class _Formatter:
    def invoke(self, value):
        return _Response("Final Answer: Citadel")


def _manifest():
    return FoldManifest(
        query_id="q:0",
        dataset="hotpot",
        context_id="q",
        question="Where does Alpha lead?",
        intent_type="Reasoning",
        intent_strategy="(Reasoning, P=0.90)",
        anchor_entities=("Alpha",),
        path_policy="directed_relation_path",
        path_steps=(
            PathStep(1, "main", "Alpha", "leads to", "Bridge", "a", 0.8, 0.7),
            PathStep(2, "main", "Bridge", "is in", "Citadel", "b", 0.7, 0.6),
        ),
        candidate_chunk_ids=("a", "b", "c"),
        selected_chunk_ids=("a", "b", "c"),
        core_chunk_ids=("a", "b"),
        peripheral_chunk_ids=("c",),
        score_order_chunk_ids=("c", "a", "b"),
        path_order_chunk_ids=("a", "b", "c"),
        chunk_scores=(("a", 0.7), ("b", 0.6), ("c", 0.9)),
        budget=3,
        source_token_budget=200,
        trace_token_budget=80,
        total_evidence_token_budget=300,
        token_budget=300,
        foldable=True,
        fallback_to_graph_naive=False,
        fallback_reason="",
        path_confidence=0.75,
        path_continuous=True,
        branch_complete=True,
        anchor_margin=0.5,
        anchor_source="query_exact",
    ).validate()


class FactorialReplayTests(unittest.TestCase):
    def setUp(self):
        chunks = pd.DataFrame([
            {"context_id": "q", "chunk_id": "a", "text": "Alpha reaches Bridge."},
            {"context_id": "q", "chunk_id": "b", "text": "Bridge is in Citadel."},
            {"context_id": "q", "chunk_id": "c", "text": "A related source."},
        ])
        self.runner = LocalControlledV2Runner(
            _Reasoner(), _Formatter(), chunks,
        )
        self.row = {
            "query_id": "q:0",
            "context_id": "q",
            "question": "Where does Alpha lead?",
            "answer": "Citadel",
            "gold_evidence": "['a', 'b']",
        }
        self.manifest = _manifest()

    def test_four_cells_share_manifest_and_selected_sources(self):
        rows = {
            variant.value: self.runner.run_row(
                self.row, self.manifest, variant,
            )
            for variant in FoldingVariant
        }
        hashes = {row["manifest_sha256"] for row in rows.values()}
        sets = {tuple(row["selected_evidence_set"]) for row in rows.values()}
        source_hashes = {row["source_text_sha256"] for row in rows.values()}
        manifest_order = {row["manifest_order_changed"] for row in rows.values()}
        self.assertEqual(len(hashes), 1)
        self.assertEqual(len(sets), 1)
        self.assertEqual(len(source_hashes), 1)
        self.assertEqual(manifest_order, {True})
        self.assertEqual(
            rows["graph_naive"]["rendered_evidence_order"],
            rows["trace_score_source"]["rendered_evidence_order"],
        )
        self.assertEqual(
            rows["path_order_source"]["rendered_evidence_order"],
            rows["topology_folding_v2"]["rendered_evidence_order"],
        )

    def test_output_contains_complete_v2_audit_schema(self):
        row = self.runner.run_row(
            self.row, self.manifest, "topology_folding_v2",
        )
        required = {
            "implementation_version", "manifest_version", "manifest_sha256",
            "path_policy", "path_length", "path_continuous",
            "path_confidence", "foldable", "folding_fallback",
            "folding_fallback_reason", "selected_count", "unused_budget",
            "core_chunk_ids", "peripheral_chunk_ids", "trace_count",
            "trace_tokens", "source_tokens", "evidence_tokens",
            "order_changed", "branch_complete", "anchor_margin",
            "manifest_order_changed", "anchor_source", "source_text_sha256",
            "folding_renderer_version",
        }
        self.assertFalse(required.difference(row))
        self.assertEqual(row["answer_calls"], 2)
        self.assertEqual(row["pred_answer"], "Citadel")

    def test_code_finalizer_uses_exactly_one_llm_call(self):
        with patch.dict("os.environ", {"ID_SGTR_CODE_FINALIZE": "true"}):
            row = self.runner.run_row(
                self.row, self.manifest, "topology_folding_v2",
            )
        self.assertEqual(row["answer_calls"], 1)
        self.assertEqual(row["pred_answer"], "Citadel")
        self.assertEqual(row["finalization_policy"], "strict_code_final_answer_marker")
        self.assertEqual(row["formatter_prompt_sha256"], "")

    def test_v2_runner_api_cannot_retrieve_or_accept_gold(self):
        parameters = inspect.signature(self.runner.run_row).parameters
        self.assertNotIn("query_vector", parameters)
        self.assertNotIn("graph", parameters)
        self.assertNotIn("gold_evidence", parameters)

    def test_cli_exposes_manifest_protocol(self):
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("build-path-manifest", help_text)
        self.assertIn("validate-manifest", help_text)
        self.assertIn("controlled-local-v2", help_text)
        args = parser.parse_args([
            "controlled-local-v2",
            "--subset", "subset.csv",
            "--chunks", "chunk.csv",
            "--manifest", "manifest.jsonl",
            "--dataset", "hotpot",
            "--variant", "topology_folding_v2",
            "--output", "result.csv",
        ])
        self.assertEqual(args.variant, "topology_folding_v2")
        lossless_args = parser.parse_args([
            "controlled-local-v2",
            "--subset", "subset.csv",
            "--chunks", "chunk.csv",
            "--manifest", "manifest.jsonl",
            "--dataset", "hotpot",
            "--variant", "topology_folding_lossless",
            "--output", "result.csv",
        ])
        self.assertEqual(
            lossless_args.variant,
            "topology_folding_lossless",
        )
        compare_args = parser.parse_args([
            "compare",
            "--a", "adjusted-a.csv",
            "--b", "scored-b.csv",
            "--metric", "f1",
            "--precomputed",
        ])
        self.assertTrue(compare_args.precomputed)
        runtime_args = parser.parse_args([
            "build-runtime-manifest",
            "--reference", "capture.csv",
            "--chunks", "chunk.csv",
            "--graph", "graph.csv",
            "--dataset", "hotpot",
            "--selection-policy", "question-aligned",
            "--beam-width", "32",
            "--output", "manifest.jsonl",
        ])
        self.assertEqual(runtime_args.selection_policy, "question-aligned")
        self.assertEqual(runtime_args.beam_width, 32)


if __name__ == "__main__":
    unittest.main()

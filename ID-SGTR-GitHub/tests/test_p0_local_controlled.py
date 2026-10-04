import unittest

import numpy as np
import pandas as pd

from knowledge_graph.experiments.local_controlled import (
    LocalCandidate,
    LocalControlledRunner,
    LocalUniverseIndex,
)


class _Response:
    def __init__(self, content, reasoning=""):
        self.content = content
        self.additional_kwargs = {"reasoning": reasoning} if reasoning else {}
        self.response_metadata = {}
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 5}


class _Reasoner:
    def invoke(self, value):
        return _Response("Final Answer: Alpha", "The evidence identifies Alpha.")


class _Formatter:
    def invoke(self, value):
        return _Response("Final Answer: Alpha")


def _fixture():
    chunks = pd.DataFrame([
        {"context_id": "q", "chunk_id": "a", "title": "Alpha", "text": "Alpha text"},
        {"context_id": "q", "chunk_id": "b", "title": "Beta", "text": "Beta text"},
        {"context_id": "q", "chunk_id": "c", "title": "Gamma", "text": "Gamma text"},
        {"context_id": "q", "chunk_id": "d", "title": "Delta", "text": "Delta text"},
        {"context_id": "other", "chunk_id": "x", "title": "Outside", "text": "Outside text"},
    ])
    vectors = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([0.9, 0.1], dtype=np.float32),
        "c": np.array([0.7, 0.3], dtype=np.float32),
        "d": np.array([0.1, 0.9], dtype=np.float32),
        "x": np.array([1.0, 0.0], dtype=np.float32),
    }
    embeddings = pd.DataFrame([
        {
            "context_id": row.context_id,
            "chunk_id": row.chunk_id,
            "embedding": vectors[row.chunk_id],
            "title_embedding": vectors[row.chunk_id],
        }
        for row in chunks.itertuples(index=False)
    ])
    graph = pd.DataFrame([
        {"context_id": "q", "chunk_id": "a", "node_1": "Alpha", "node_2": "Bridge", "edge": "links"},
        {"context_id": "q", "chunk_id": "b", "node_1": "Bridge", "node_2": "Beta", "edge": "links"},
        {"context_id": "q", "chunk_id": "c", "node_1": "Gamma", "node_2": "Leaf", "edge": "links"},
        {"context_id": "q", "chunk_id": "d", "node_1": "Beta", "node_2": "Delta", "edge": "links"},
        {"context_id": "other", "chunk_id": "x", "node_1": "Outside", "node_2": "World", "edge": "links"},
    ])
    return chunks, embeddings, graph


class LocalUniverseTests(unittest.TestCase):
    def setUp(self):
        self.chunks, self.embeddings, self.graph = _fixture()
        self.index = LocalUniverseIndex(
            self.chunks, self.embeddings, self.graph,
            budget=3, seed=42, loose_threshold=0.1, strict_threshold=0.5,
        )
        self.query = np.array([1.0, 0.0], dtype=np.float32)

    def test_complete_context_and_budget(self):
        candidates, selected = self.index.select(
            "q", self.query, "source_score", query_id="q:0",
        )
        self.assertEqual({item.chunk_id for item in candidates}, {"a", "b", "c", "d"})
        self.assertNotIn("x", {item.chunk_id for item in candidates})
        self.assertEqual(len(selected), 3)

    def test_score_and_random_share_ids(self):
        _, score = self.index.select("q", self.query, "source_score", query_id="q:0")
        _, random_order = self.index.select("q", self.query, "source_random", query_id="q:0")
        self.assertEqual({item.chunk_id for item in score}, {item.chunk_id for item in random_order})
        _, repeated = self.index.select("q", self.query, "source_random", query_id="q:0")
        self.assertEqual([item.chunk_id for item in random_order], [item.chunk_id for item in repeated])

    def test_triple_and_topology_share_ids(self):
        _, triples = self.index.select("q", self.query, "triple_only", query_id="q:0")
        _, topology = self.index.select("q", self.query, "topology_folding", query_id="q:0")
        self.assertEqual({item.chunk_id for item in triples}, {item.chunk_id for item in topology})

    def test_topology_support_does_not_override_semantic_relevance(self):
        def candidate(chunk_id, semantic_score, support_count, position):
            return LocalCandidate(
                chunk_id=chunk_id,
                text=chunk_id,
                title=chunk_id,
                semantic_score=semantic_score,
                title_score=semantic_score,
                structural=True,
                support_count=support_count,
                hop=1,
                path_position=position,
                topology_trace=(),
            )

        candidates = [
            candidate("relevant", 0.48, 1, 3),
            candidate("central_distractor", 0.43, 3, 1),
            candidate("first", 0.53, 2, 2),
            candidate("second", 0.51, 2, 4),
        ]
        selected = self.index._topology_selection(candidates)
        self.assertEqual(
            {item.chunk_id for item in selected},
            {"first", "second", "relevant"},
        )

    def test_structural_anchor_hops_are_finite_across_components(self):
        candidates = self.index._context_candidates("q", self.query)
        structural_hops = [item.hop for item in candidates if item.structural]
        self.assertTrue(structural_hops)
        self.assertLess(max(structural_hops), 10_000)

    def test_only_graph_connected_core_is_structural(self):
        candidates = {
            item.chunk_id: item
            for item in self.index._context_candidates("q", self.query)
        }
        self.assertTrue(candidates["a"].structural)
        self.assertTrue(candidates["b"].structural)
        self.assertFalse(candidates["c"].structural)

    def test_classifier_comparison_uses_semantic_branches(self):
        candidates = {
            item.chunk_id: item
            for item in self.index._context_candidates(
                "q", self.query, "Comparative"
            )
        }
        self.assertTrue(candidates["a"].structural)
        self.assertTrue(candidates["b"].structural)
        self.assertTrue(candidates["c"].structural)
        self.assertFalse(candidates["d"].structural)
        self.assertEqual(
            self.index.topology_policy("Comparative"),
            "multi_branch_comparison",
        )

    def test_oracle_only_uses_gold_and_budget(self):
        _, selected = self.index.select(
            "q", self.query, "oracle", query_id="q:0",
            gold_evidence="['d', 'b', 'x', 'a']",
        )
        self.assertEqual([item.chunk_id for item in selected], ["d", "b", "a"])

    def test_runner_records_two_calls_and_full_manifest(self):
        runner = LocalControlledRunner(_Reasoner(), _Formatter(), self.index)
        result = runner.run_row({
            "query_id": "q:0",
            "question": "Who?",
            "answer": "Alpha",
            "context_id": "q",
            "gold_evidence": "['a']",
        }, self.query, "source_score")
        self.assertEqual(result["pred_answer"], "Alpha")
        self.assertEqual(result["answer_calls"], 2)
        self.assertEqual(result["local_universe_size"], 4)
        self.assertEqual(len(result["candidate_evidence"]), 4)
        self.assertLessEqual(len(result["retrieved_evidence"]), 3)


if __name__ == "__main__":
    unittest.main()

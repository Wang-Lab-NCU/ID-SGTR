import unittest

import numpy as np
import pandas as pd

from knowledge_graph.experiments.topology_folding_v2 import (
    FolderConfig,
    TopologyFolderV2,
    _relation_slot_groups,
    _tokens,
)


def _vectors(values):
    return np.asarray(values, dtype=np.float32)


def _fixture():
    chunks = pd.DataFrame([
        {"context_id": "reason", "chunk_id": "a", "title": "Alpha", "text": "Alpha reaches Bridge."},
        {"context_id": "reason", "chunk_id": "b", "title": "Bridge", "text": "Bridge is located in Citadel."},
        {"context_id": "reason", "chunk_id": "p", "title": "Peripheral", "text": "Bridge is also related to Peripheral."},
        {"context_id": "reason", "chunk_id": "u", "title": "Unrelated", "text": "A very similar but disconnected distractor."},
        {"context_id": "comp", "chunk_id": "ca", "title": "Alpha", "text": "Alpha was born in France."},
        {"context_id": "comp", "chunk_id": "cb", "title": "Beta", "text": "Beta was born in France."},
        {"context_id": "comp", "chunk_id": "cu", "title": "Noise", "text": "Noise."},
    ])
    vectors = {
        "a": _vectors([0.96, 0.04]),
        "b": _vectors([0.86, 0.14]),
        "p": _vectors([0.66, 0.34]),
        "u": _vectors([0.995, 0.005]),
        "ca": _vectors([0.92, 0.08]),
        "cb": _vectors([0.90, 0.10]),
        "cu": _vectors([0.20, 0.80]),
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
        {"context_id": "reason", "chunk_id": "a", "node_1": "Alpha", "node_2": "Bridge", "edge": "Alpha leads to Bridge"},
        # Four unrelated relations in the same source must never be copied as a trace bundle.
        {"context_id": "reason", "chunk_id": "a", "node_1": "Alpha", "node_2": "Noise One", "edge": "unrelated catalog fact"},
        {"context_id": "reason", "chunk_id": "a", "node_1": "Alpha", "node_2": "Noise Two", "edge": "another unrelated fact"},
        {"context_id": "reason", "chunk_id": "a", "node_1": "Alpha", "node_2": "Noise Three", "edge": "third unrelated fact"},
        {"context_id": "reason", "chunk_id": "a", "node_1": "Alpha", "node_2": "Noise Four", "edge": "fourth unrelated fact"},
        {"context_id": "reason", "chunk_id": "b", "node_1": "Bridge", "node_2": "Citadel", "edge": "Bridge is located in Citadel"},
        {"context_id": "reason", "chunk_id": "p", "node_1": "Bridge", "node_2": "Peripheral", "edge": "Bridge has a peripheral note"},
        {"context_id": "reason", "chunk_id": "u", "node_1": "Other", "node_2": "Distractor", "edge": "disconnected located distractor"},
        {"context_id": "comp", "chunk_id": "ca", "node_1": "Alpha", "node_2": "France", "edge": "Alpha was born in France"},
        {"context_id": "comp", "chunk_id": "cb", "node_1": "Beta", "node_2": "France", "edge": "Beta was born in France"},
        {"context_id": "comp", "chunk_id": "cu", "node_1": "Other", "node_2": "Noise", "edge": "irrelevant"},
    ])
    return chunks, embeddings, graph


class TopologyFoldingV2Tests(unittest.TestCase):
    def setUp(self):
        chunks, embeddings, graph = _fixture()
        self.config = FolderConfig(
            budget=3,
            loose_threshold=0.20,
            strict_threshold=0.60,
            beam_width=12,
            max_path_steps=2,
            min_edge_score=0.05,
            path_confidence_threshold=0.05,
            anchor_threshold=0.20,
        )
        self.folder = TopologyFolderV2(
            chunks, embeddings, graph, dataset="synthetic", config=self.config,
        )
        self.query_vector = _vectors([1.0, 0.0])

    def _reasoning_manifest(self):
        return self.folder.build_manifest(
            "reason:0",
            "reason",
            "Where is the place reached from Alpha located?",
            self.query_vector,
            "Reasoning",
        )

    def test_reasoning_path_is_relation_level_and_continuous(self):
        manifest = self._reasoning_manifest()
        self.assertTrue(manifest.foldable)
        self.assertEqual(
            [(step.source_entity, step.target_entity) for step in manifest.path_steps],
            [("Alpha", "Bridge"), ("Bridge", "Citadel")],
        )
        self.assertTrue(manifest.path_continuous)
        self.assertEqual([step.step for step in manifest.path_steps], [1, 2])

    def test_only_traversed_relations_become_trace_steps(self):
        manifest = self._reasoning_manifest()
        relations = [step.relation for step in manifest.path_steps]
        self.assertEqual(len(relations), 2)
        self.assertNotIn("unrelated catalog fact", relations)
        for step in manifest.path_steps:
            self.assertIn(step.supporting_chunk_id, manifest.core_chunk_ids)

    def test_peripheral_is_adjacent_and_disconnected_high_score_is_excluded(self):
        manifest = self._reasoning_manifest()
        self.assertIn("p", manifest.peripheral_chunk_ids)
        self.assertNotIn("u", manifest.selected_chunk_ids)
        self.assertEqual(set(manifest.selected_chunk_ids), {"a", "b", "p"})

    def test_budget_may_remain_unused_without_semantic_filler(self):
        chunks, embeddings, graph = _fixture()
        folder = TopologyFolderV2(
            chunks,
            embeddings,
            graph,
            config=FolderConfig(
                budget=3,
                loose_threshold=0.20,
                strict_threshold=0.999,
                max_path_steps=2,
                min_edge_score=0.05,
                path_confidence_threshold=0.05,
                anchor_threshold=0.20,
            ),
        )
        manifest = folder.build_manifest(
            "reason:0", "reason",
            "Where is the place reached from Alpha located?",
            self.query_vector, "Reasoning",
        )
        self.assertEqual(set(manifest.selected_chunk_ids), {"a", "b"})
        self.assertEqual(manifest.unused_budget, 1)
        self.assertNotIn("u", manifest.selected_chunk_ids)

    def test_comparative_uses_two_independent_branches(self):
        manifest = self.folder.build_manifest(
            "comp:0", "comp",
            "Were Alpha and Beta born in the same country?",
            self.query_vector, "Comparative",
        )
        self.assertTrue(manifest.foldable)
        self.assertTrue(manifest.branch_complete)
        self.assertEqual({step.branch for step in manifest.path_steps}, {"A", "B"})
        self.assertEqual(set(manifest.core_chunk_ids), {"ca", "cb"})
        for branch in ("A", "B"):
            self.assertTrue([step for step in manifest.path_steps if step.branch == branch])

    def test_no_query_anchor_falls_back_exactly_to_score_order(self):
        manifest = self.folder.build_manifest(
            "reason:1", "reason", "What happened yesterday?",
            self.query_vector, "Reasoning",
        )
        self.assertFalse(manifest.foldable)
        self.assertTrue(manifest.fallback_to_graph_naive)
        self.assertEqual(manifest.fallback_reason, "no_reliable_query_anchor")
        self.assertFalse(manifest.path_steps)
        self.assertEqual(
            manifest.selected_chunk_ids,
            manifest.score_order_chunk_ids,
        )

    def test_manifest_construction_is_gold_independent_by_api(self):
        first = self._reasoning_manifest()
        second = self.folder.build_manifest(
            "reason:0", "reason",
            "Where is the place reached from Alpha located?",
            self.query_vector, "Reasoning",
        )
        self.assertEqual(first.sha256, second.sha256)
        with self.assertRaises(TypeError):
            self.folder.build_manifest(
                "reason:0", "reason",
                "Where is the place reached from Alpha located?",
                self.query_vector, "Reasoning", gold_evidence=["b"],
            )

    def test_reasoning_does_not_accept_a_path_shorter_than_target_depth(self):
        chunks, embeddings, graph = _fixture()
        one_edge_graph = graph.loc[
            graph["context_id"].eq("reason")
            & graph["chunk_id"].eq("a")
            & graph["node_2"].eq("Bridge")
        ]
        folder = TopologyFolderV2(
            chunks, embeddings, one_edge_graph,
            config=self.config,
        )
        manifest = folder.build_manifest(
            "reason:short", "reason",
            "Where is the place reached from Alpha located?",
            self.query_vector, "Reasoning",
        )
        self.assertFalse(manifest.foldable)
        self.assertEqual(manifest.fallback_reason, "no_continuous_relation_path")

    def test_comparative_falls_back_when_budget_cannot_retain_both_branches(self):
        chunks, embeddings, graph = _fixture()
        folder = TopologyFolderV2(
            chunks, embeddings, graph,
            config=FolderConfig(
                budget=1,
                loose_threshold=0.20,
                strict_threshold=0.60,
                max_path_steps=2,
                min_edge_score=0.05,
                path_confidence_threshold=0.05,
                anchor_threshold=0.20,
            ),
        )
        manifest = folder.build_manifest(
            "comp:small", "comp",
            "Were Alpha and Beta born in the same country?",
            self.query_vector, "Comparative",
        )
        self.assertFalse(manifest.foldable)
        self.assertFalse(manifest.branch_complete)
        self.assertEqual(manifest.fallback_reason, "comparative_branch_incomplete")

    def test_nested_of_question_uses_exact_relation_depth(self):
        self.assertEqual(
            TopologyFolderV2._infer_target_steps(
                "Who is the mother of the director of the film?",
                "Reasoning",
                3,
            ),
            2,
        )

    def test_comparative_relation_depth_preserves_temporal_slot(self):
        self.assertEqual(
            TopologyFolderV2._infer_target_steps(
                "Which director of the two films died earlier?",
                "Comparative",
                3,
            ),
            2,
        )
        self.assertEqual(
            TopologyFolderV2._infer_target_steps(
                "Were Scott Derrickson and Ed Wood of the same nationality?",
                "Comparative",
                3,
            ),
            1,
        )

    def test_unicode_entities_are_not_split_or_dropped(self):
        self.assertEqual(_tokens("Ælfgar Adèle Małgorzata Żuławski"), (
            "ælfgar", "adèle", "małgorzata", "żuławski",
        ))

    def test_endpoint_surface_cues_share_one_relation_slot(self):
        groups = _relation_slot_groups(
            "Where was the director of the film born?"
        )
        self.assertIn(frozenset({"birth", "location"}), groups)
        self.assertIn(frozenset({"director"}), groups)
        self.assertEqual(len(groups), 2)
        self.assertEqual(
            TopologyFolderV2._infer_target_steps(
                "Who is the spouse of the Green performer?",
                "Reasoning",
                3,
            ),
            2,
        )

    def test_remaining_relation_slot_beats_repeated_director_edge(self):
        chunks = pd.DataFrame([
            {"context_id": "slot", "chunk_id": "film", "title": "Film", "text": "The film and director."},
            {"context_id": "slot", "chunk_id": "family", "title": "Mother", "text": "Ada is the director's mother."},
        ])
        embeddings = pd.DataFrame([
            {"context_id": "slot", "chunk_id": "film", "embedding": _vectors([0.95, 0.05]), "title_embedding": _vectors([0.95, 0.05])},
            {"context_id": "slot", "chunk_id": "family", "embedding": _vectors([0.55, 0.45]), "title_embedding": _vectors([0.55, 0.45])},
        ])
        graph = pd.DataFrame([
            {"context_id": "slot", "chunk_id": "film", "node_1": "Example Film", "node_2": "Dana Director", "edge": "Example Film was directed by Dana Director."},
            {"context_id": "slot", "chunk_id": "film", "node_1": "Dana Director", "node_2": "Alternate Film", "edge": "Dana Director also directed Alternate Film."},
            {"context_id": "slot", "chunk_id": "family", "node_1": "Ada Mother", "node_2": "Dana Director", "edge": "Ada Mother is the mother of Dana Director."},
        ])
        folder = TopologyFolderV2(
            chunks, embeddings, graph,
            config=FolderConfig(
                budget=3, loose_threshold=0.10, strict_threshold=0.80,
                max_path_steps=3, min_edge_score=0.05,
                path_confidence_threshold=0.05, anchor_threshold=0.20,
            ),
        )
        manifest = folder.build_manifest(
            "slot:0", "slot",
            "Who is the mother of the director of Example Film?",
            _vectors([1.0, 0.0]), "Reasoning",
        )
        self.assertTrue(manifest.foldable)
        self.assertEqual(manifest.path_length, 2)
        self.assertEqual(
            [(step.source_entity, step.target_entity) for step in manifest.path_steps],
            [("Example Film", "Dana Director"), ("Dana Director", "Ada Mother")],
        )
        self.assertEqual(manifest.path_steps[1].traversal_direction, "reverse")

    def test_comparative_nationality_rejects_temporal_distractors(self):
        chunks = pd.DataFrame([
            {"context_id": "nation", "chunk_id": "ey", "title": "Ed year", "text": "A 1994 American film."},
            {"context_id": "nation", "chunk_id": "en", "title": "Ed nationality", "text": "Ed Wood was American."},
            {"context_id": "nation", "chunk_id": "sb", "title": "Scott birth", "text": "Scott was born in 1966."},
            {"context_id": "nation", "chunk_id": "sn", "title": "Scott nationality", "text": "Scott Derrickson is American."},
        ])
        scores = {"ey": 0.95, "en": 0.65, "sb": 0.90, "sn": 0.60}
        embeddings = pd.DataFrame([
            {
                "context_id": "nation", "chunk_id": chunk_id,
                "embedding": _vectors([score, 1 - score]),
                "title_embedding": _vectors([score, 1 - score]),
            }
            for chunk_id, score in scores.items()
        ])
        graph = pd.DataFrame([
            {"context_id": "nation", "chunk_id": "ey", "node_1": "Ed Wood", "node_2": "1994", "edge": "The American film Ed Wood was released in 1994."},
            {"context_id": "nation", "chunk_id": "en", "node_1": "Ed Wood", "node_2": "American", "edge": "Ed Wood was an American director."},
            {"context_id": "nation", "chunk_id": "sb", "node_1": "Scott Derrickson", "node_2": "July 16, 1966", "edge": "Scott Derrickson was born on July 16, 1966."},
            {"context_id": "nation", "chunk_id": "sn", "node_1": "Scott Derrickson", "node_2": "American", "edge": "Scott Derrickson is an American director."},
        ])
        folder = TopologyFolderV2(
            chunks, embeddings, graph,
            config=FolderConfig(
                budget=3, loose_threshold=0.10, strict_threshold=0.99,
                max_path_steps=2, min_edge_score=0.05,
                path_confidence_threshold=0.05, anchor_threshold=0.20,
            ),
        )
        manifest = folder.build_manifest(
            "nation:0", "nation",
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            _vectors([1.0, 0.0]), "Comparative",
        )
        self.assertTrue(manifest.foldable)
        self.assertEqual(
            {step.target_entity for step in manifest.path_steps},
            {"American"},
        )
        self.assertEqual(set(manifest.core_chunk_ids), {"en", "sn"})

    def test_parent_slot_accepts_child_perspective_actress_relation(self):
        chunks = pd.DataFrame([
            {"context_id": "kin", "chunk_id": "film", "title": "Film", "text": "Film director."},
            {"context_id": "kin", "chunk_id": "bio", "title": "Director", "text": "Director and mother."},
        ])
        embeddings = pd.DataFrame([
            {"context_id": "kin", "chunk_id": "film", "embedding": _vectors([0.9, 0.1]), "title_embedding": _vectors([0.9, 0.1])},
            {"context_id": "kin", "chunk_id": "bio", "embedding": _vectors([0.7, 0.3]), "title_embedding": _vectors([0.7, 0.3])},
        ])
        graph = pd.DataFrame([
            {"context_id": "kin", "chunk_id": "film", "node_1": "Example Film", "node_2": "Dana", "edge": "Example Film was directed by Dana."},
            {"context_id": "kin", "chunk_id": "bio", "node_1": "Dana", "node_2": "Ada", "edge": "Dana is the son of actress Ada."},
            {"context_id": "kin", "chunk_id": "bio", "node_1": "Dana", "node_2": "Other Film", "edge": "Dana also directed Other Film."},
        ])
        folder = TopologyFolderV2(
            chunks, embeddings, graph,
            config=FolderConfig(
                budget=3, loose_threshold=0.10, strict_threshold=0.80,
                max_path_steps=3, min_edge_score=0.05,
                path_confidence_threshold=0.05, anchor_threshold=0.20,
            ),
        )
        manifest = folder.build_manifest(
            "kin:0", "kin", "Who is the mother of the director of Example Film?",
            _vectors([1.0, 0.0]), "Reasoning",
        )
        self.assertTrue(manifest.foldable)
        self.assertEqual(
            [(step.source_entity, step.target_entity) for step in manifest.path_steps],
            [("Example Film", "Dana"), ("Dana", "Ada")],
        )

    def test_comparative_missing_typed_branch_fails_closed(self):
        chunks = pd.DataFrame([
            {"context_id": "guard", "chunk_id": "ey", "title": "Ed film", "text": "A 1994 American film."},
            {"context_id": "guard", "chunk_id": "sn", "title": "Scott", "text": "Scott is American."},
        ])
        embeddings = pd.DataFrame([
            {"context_id": "guard", "chunk_id": "ey", "embedding": _vectors([0.9, 0.1]), "title_embedding": _vectors([0.9, 0.1])},
            {"context_id": "guard", "chunk_id": "sn", "embedding": _vectors([0.8, 0.2]), "title_embedding": _vectors([0.8, 0.2])},
        ])
        graph = pd.DataFrame([
            {"context_id": "guard", "chunk_id": "ey", "node_1": "Ed Wood", "node_2": "1994", "edge": "The American film Ed Wood was released in 1994."},
            {"context_id": "guard", "chunk_id": "sn", "node_1": "Scott Derrickson", "node_2": "American", "edge": "Scott Derrickson is an American director."},
        ])
        folder = TopologyFolderV2(
            chunks, embeddings, graph,
            config=FolderConfig(
                budget=3, loose_threshold=0.10, strict_threshold=0.80,
                max_path_steps=2, min_edge_score=0.05,
                path_confidence_threshold=0.05, anchor_threshold=0.20,
            ),
        )
        manifest = folder.build_manifest(
            "guard:0", "guard",
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            _vectors([1.0, 0.0]), "Comparative",
        )
        self.assertFalse(manifest.foldable)
        self.assertTrue(manifest.fallback_to_graph_naive)
        self.assertEqual(manifest.fallback_reason, "comparative_branch_incomplete")


if __name__ == "__main__":
    unittest.main()

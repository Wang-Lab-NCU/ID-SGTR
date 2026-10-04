import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import networkx as nx

from knowledge_graph.experiments.evidence import EvidenceAssembler, EvidenceItem
from knowledge_graph.experiments.pcef_v2 import latest_scores, relations, select_set, stage0_select


def candidates():
    return [
        EvidenceItem('a','Film Alpha was directed by John Smith.',.95,triple='Film Alpha --[directed by]--> John Smith'),
        EvidenceItem('b','Additional useful facts about Film Alpha.',.90),
        EvidenceItem('c','A second description of Film Alpha.',.80),
        EvidenceItem('d','John Smith is the son of Mary Smith.',.77,triple='John Smith --[mother]--> Mary Smith'),
        EvidenceItem('e','Unrelated source.',.1),
    ]


def configure(monkeypatch):
    monkeypatch.setenv('ID_SGTR_PCEF_VERSION','v2')
    monkeypatch.setenv('ID_SGTR_PATH_CONSTRAINED_SELECTION','true')
    monkeypatch.setenv('ID_SGTR_PCEF_PROTECT','2')
    monkeypatch.setenv('ID_SGTR_PCEF_STRUCTURE_WEIGHT','0.20')
    monkeypatch.setenv('ID_SGTR_PCEF_MAX_RELEVANCE_DROP','0.15')
    monkeypatch.delenv('ID_SGTR_PCEF_AUDIT_FILE',raising=False)


def test_selects_grounded_missing_bridge_and_respects_core(monkeypatch):
    configure(monkeypatch)
    out=select_set(candidates(),'Who is the mother of the director of Film Alpha?',3)
    assert [x.chunk_id for x in out]==['a','b','d']


def test_insufficient_relevance_falls_back(monkeypatch):
    configure(monkeypatch)
    items=candidates();items[3]=replace(items[3],score=.11)
    assert [x.chunk_id for x in select_set(items,'Who is the mother of the director of Film Alpha?',3)]==['a','b','c']


def test_gold_flags_do_not_affect_selection(monkeypatch):
    configure(monkeypatch)
    query='Who is the mother of the director of Film Alpha?'
    a=select_set(candidates(),query,3)
    b=select_set([replace(x,is_gold=True) for x in candidates()],query,3)
    assert a==[replace(x,is_gold=False) for x in b]


def test_latest_hop_score_not_historical_max():
    a=EvidenceItem('a','source',.9,hop=1)
    b=replace(a,score=.2,hop=2)
    assert {x.score for x in latest_scores([a,b])}=={.2}


def test_unbacked_and_implicit_relations_rejected():
    assert not relations(EvidenceItem('a','Some text',1,triple='A --[mother]--> B'))
    assert not relations(EvidenceItem('a','Alice Bob',1,triple='Alice --[co-occurs with]--> Bob'))


def test_stage0_graph_respects_context_and_uses_all_candidates(monkeypatch):
    configure(monkeypatch)
    items=candidates();g=nx.Graph()
    g.add_edge('Film Alpha','John Smith',relation='directed by',context_ids={'ctx'},chunk_ids=['a'])
    g.add_edge('John Smith','Mary Smith',relation='mother',context_ids={'ctx'},chunk_ids=['d'])
    engine=SimpleNamespace(G=g,_get_chunk_text=lambda cid:next(x.text for x in items if x.chunk_id==cid))
    ranked=[(x.chunk_id,x.score) for x in items]
    assert stage0_select(engine,['Film Alpha'],'ctx',ranked,3,'Who is the mother of the director of Film Alpha?')==['a','b','d']
    assert stage0_select(engine,['Film Alpha'],'another',ranked,3,'Who is the mother of the director of Film Alpha?')==['a','b','c']


def test_disabled_selector_is_topb_and_legacy_unchanged(monkeypatch):
    configure(monkeypatch);monkeypatch.setenv('ID_SGTR_PATH_CONSTRAINED_SELECTION','false')
    assert [x.chunk_id for x in select_set(candidates(),'mother director Film Alpha',3)]==['a','b','c']
    monkeypatch.delenv('ID_SGTR_PCEF_VERSION')
    assert [x.chunk_id for x in EvidenceAssembler()._freeze_budget(candidates())]==['a','b','c']


def test_six_entrypoints_have_scored_stage0_adapter():
    root=Path(__file__).resolve().parents[1]
    for ds in ['hotpot','2wiki','musique']:
        for mode in ['local','global']:
            source=(root/f'knowledge_graph/{ds}/query_{mode}.py').read_text(encoding='utf-8')
            tree=ast.parse(source)
            assert any(isinstance(n,ast.FunctionDef) and n.name=='_rank_chunks' for n in ast.walk(tree))
            assert 'stage0_select(self, nodes,' in source


def test_half_budget_replacement_limit(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setenv('ID_SGTR_PCEF_PROTECT','1')
    monkeypatch.setenv('ID_SGTR_PCEF_MAX_RELEVANCE_DROP','10')
    monkeypatch.setenv('ID_SGTR_PCEF_STRUCTURE_WEIGHT','100')
    pool=[EvidenceItem(str(i),f'noise{i}',1-i*.01) for i in range(5)]
    pool += [EvidenceItem('x','alpha',.2),EvidenceItem('y','beta',.1)]
    for budget,expected in [(1,0),(3,1),(5,2)]:
        chosen=select_set(pool,'alpha beta',budget)
        baseline={x.chunk_id for x in pool[:budget]}
        assert len(chosen)==budget
        assert len({x.chunk_id for x in chosen}-baseline)==expected
        assert '0' in {x.chunk_id for x in chosen}

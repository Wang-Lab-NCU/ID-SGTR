"""Opt-in PCEF v2: score-preserving Stage0 integration and bounded set search.

Only question, source text, retrieval scores and context-local graph metadata
are used. Gold annotations never enter selection. All audit records precede QA.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import threading
import unicodedata

_LOCK = threading.Lock()
_STOP = set('a an the is are was were be been being of in on at for from to by and or with who what which when where how whose did does do has have had more less same both than'.split())
_GENERIC = {'person', 'human', 'male', 'female', 'united states', 'american', 'english', 'film', 'album', 'united kingdom'}
_GROUPS = [
    {'mother', 'maternal'}, {'father', 'paternal'}, {'spouse', 'married', 'wife', 'husband'},
    {'director', 'directed', 'directing'}, {'author', 'wrote', 'written', 'writer'},
    {'born', 'birth', 'birthplace'}, {'died', 'death'},
    {'nationality', 'citizen', 'citizenship'}, {'located', 'location', 'situated'},
]


def enabled():
    return os.getenv('ID_SGTR_PCEF_VERSION', 'legacy') == 'v2'


def active():
    return enabled() and os.getenv('ID_SGTR_PATH_CONSTRAINED_SELECTION', '').lower() in {'1', 'true', 'yes', 'on'}


def norm(value):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', str(value)).casefold()))


def tokens(value):
    return set(norm(value).split()) - _STOP


def mentions(entity, text):
    e = norm(entity)
    return bool(e) and (' ' + e + ' ') in (' ' + norm(text) + ' ')


def relations(item):
    triples = [str(t[2]) for t in item.topology_trace]
    if item.triple:
        triples.append(item.triple)
    found = set()
    for triple in triples:
        match = re.fullmatch(r'\s*(.*?)\s*--\[(.*?)\]-->\s*(.*?)\s*', triple, re.DOTALL)
        if not match:
            continue
        u, rel, v = match.groups()
        if norm(rel) in {'co occurs with', 'related to', ''}:
            continue
        # Merged graph edges may refer to a different context/source; require
        # both endpoint names in the particular source used for this trace.
        if item.text and mentions(u, item.text) and mentions(v, item.text):
            found.add((norm(u), norm(rel), norm(v)))
    return found


def audit(query, phase, candidates, baseline, selected, **extra):
    output = os.getenv('ID_SGTR_PCEF_AUDIT_FILE', '')
    if not output:
        return
    record = dict(query=query, phase=phase, version='pcef-v2',
                  candidate_ids=[x.chunk_id for x in candidates],
                  scores={x.chunk_id: x.score for x in candidates},
                  baseline_ids=[x.chunk_id for x in baseline],
                  selected_ids=[x.chunk_id for x in selected],
                  selection_changed=set(x.chunk_id for x in baseline) != set(x.chunk_id for x in selected),
                  **extra)
    with _LOCK:
        with open(output, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def latest_scores(items):
    """Avoid carrying the largest score from an obsolete Hop-aware query."""
    latest = {}
    for item in items:
        old = latest.get(item.chunk_id)
        if old is None or item.hop >= old.hop:
            latest[item.chunk_id] = item
    return [replace(item, score=latest[item.chunk_id].score) for item in items]


def select_set(items, query, budget, phase='hop'):
    """Search small B-sized sets; preserve core and cap relevance sacrifice."""
    items = list(items)
    base = sorted(items, key=lambda x: (-x.score, x.chunk_id))[:budget]
    if not active() or len(items) <= budget:
        audit(query, phase, items, base, base, reason='disabled_or_small_pool')
        return base
    values = [x.score for x in items]
    if not all(math.isfinite(s) for s in values):
        raise ValueError('PCEF received non-finite retrieval score')
    lo, hi = min(values), max(values)
    scores = {x.chunk_id: (x.score-lo)/(hi-lo) if hi-lo > 1e-12 else 0.5 for x in items}
    protected = min(budget, max(1, int(os.getenv('ID_SGTR_PCEF_PROTECT', '2'))))
    max_drop = float(os.getenv('ID_SGTR_PCEF_MAX_RELEVANCE_DROP', '0.15'))
    strength = float(os.getenv('ID_SGTR_PCEF_STRUCTURE_WEIGHT', '0.20'))
    qtokens = tokens(query)
    for group in _GROUPS:
        if qtokens & group:
            qtokens |= group
    edges = {x.chunk_id: relations(x) for x in items}
    ents = {cid: {v for edge in es for v in (edge[0],edge[2])} for cid,es in edges.items()}
    occurrence = Counter(e for values in ents.values() for e in values)
    text_tokens = {x.chunk_id: tokens(x.text) for x in items}
    def structure(group):
        union_tokens = set().union(*(text_tokens[x.chunk_id] for x in group))
        coverage = len(qtokens & union_tokens) / max(1, len(qtokens))
        bridges = 0.0
        redundancy = 0.0
        comparative = bool(re.search(r'\b(same|different|both|more|less|older|younger|earlier|later)\b', query.casefold()))
        for a,b in combinations(group, 2):
            ea,eb = edges[a.chunk_id],edges[b.chunk_id]
            shared = ents[a.chunk_id] & ents[b.chunk_id] - _GENERIC
            # Shared endpoints alone are insufficient: relation steps must be
            # different, query-connected, and actually supported by passages.
            for common in shared:
                relevant = any(mentions(e, query) for e in (ents[a.chunk_id] | ents[b.chunk_id]) - {common})
                different_rel = any(x[1] != y[1] and common in (x[0],x[2]) and common in (y[0],y[2]) for x in ea for y in eb)
                slot = any(tokens(e[1]) & qtokens for e in ea | eb)
                if relevant and different_rel and slot:
                    bridges += 1.0 / max(1, occurrence[common]-1)
                    break
            if comparative:
                qa = {e for e in ents[a.chunk_id] if mentions(e,query)}
                qb = {e for e in ents[b.chunk_id] if mentions(e,query)}
                same_property = any((tokens(x[1]) & tokens(y[1]) & qtokens) for x in ea for y in eb)
                if qa and qb and qa-qb and qb-qa and same_property:
                    bridges += 1.0
            ta,tb = text_tokens[a.chunk_id],text_tokens[b.chunk_id]
            redundancy += len(ta & tb) / max(1,len(ta | tb))
        return min(bridges,2.0) + 0.35*coverage - 0.35*redundancy
    base_ids = {x.chunk_id for x in base}
    core = base[:protected]
    rest = [x for x in items if x.chunk_id not in {y.chunk_id for y in core}]
    base_rel = sum(scores[x.chunk_id] for x in base)
    base_struct = structure(base)
    best, best_gain = base, 0.0
    max_replacements = budget // 2
    for tail in combinations(rest, budget-len(core)):
        group = core + list(tail)
        if len({x.chunk_id for x in group} - base_ids) > max_replacements:
            continue
        # Reject exact textual copies under different IDs when alternatives exist.
        texts = [norm(x.text) for x in group]
        if all(texts) and len(set(texts)) < len(texts):
            continue
        rel = sum(scores[x.chunk_id] for x in group)
        drop = base_rel-rel
        if drop > max_drop+1e-12:
            continue
        gain = -drop + strength*(structure(group)-base_struct)
        if gain > best_gain+1e-9:
            best, best_gain = group, gain
    # Keep presentation stable: first isolate set selection, not prompt order.
    best = sorted(best,key=lambda x:(-x.score,x.chunk_id))
    audit(query,phase,items,base,best,objective_gain=best_gain,protected=protected,max_drop=max_drop,max_replacements=max_replacements)
    return best


def stage0_select(engine, nodes, context_id, ranked, limit, query):
    """Build source-grounded, context-local traces before the first QA call."""
    from .evidence import EvidenceItem
    ranked = [(str(cid),float(score)) for cid,score in ranked]
    allowed = {cid for cid,_ in ranked}
    traces = {cid: [] for cid in allowed}
    text = {cid:engine._get_chunk_text(cid) for cid in allowed}
    if active():
        frontier = sorted(set(str(x) for x in nodes))
        visited = set()
        position = 0
        for depth in range(2):
            next_nodes = set()
            for u in frontier:
                if u not in engine.G or u in visited:
                    continue
                visited.add(u)
                qualifying = []
                for v,data in engine.G[u].items():
                    if str(context_id) not in {str(x) for x in data.get('context_ids',[])}:
                        continue
                    ids = allowed & {str(x) for x in data.get('chunk_ids',[])}
                    if ids:
                        qualifying.append((str(v),data,ids))
                for v,data,ids in sorted(qualifying,key=lambda z:z[0])[:64]:
                    rel = str(data.get('relation',''))
                    if norm(rel) in {'co occurs with','related to',''}:
                        continue
                    supported = [cid for cid in sorted(ids) if mentions(u,text[cid]) and mentions(v,text[cid])]
                    if not supported:
                        continue
                    position += 1
                    for cid in supported:
                        traces[cid].append((0,position,f'{u} --[{rel}]--> {v}'))
                    next_nodes.add(v)
            frontier = sorted(next_nodes-visited)
    items = [EvidenceItem(chunk_id=cid,text=text[cid],score=score,
                          path_position=i,topology_trace=tuple(traces[cid]))
             for i,(cid,score) in enumerate(ranked,1)]
    return [x.chunk_id for x in select_set(items,query,limit,phase='stage0')]

"""Факты для обзорного поиска и документы, подтверждающие каждый факт."""

import re
from collections import defaultdict
from itertools import combinations

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.schemas import KnowledgeBase


def _sources(*edges: dict) -> set[str]:
    documents = set()
    for edge in edges:
        for chunk in edge.get("sources", []):
            documents.add(chunk.split("#", 1)[0])
    return documents


def _anchors(question: str, kb: KnowledgeBase) -> list[str]:
    anchors = [f"{code_type(code)}:{code}" for code in find_codes(question)]
    anchors.extend(f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary))
    existing = [node for node in anchors if node in kb.graph]
    return list(dict.fromkeys(existing))


def procedure_component(question: str, kb: KnowledgeBase) -> str | None:
    """Определяет вопрос о регламенте одного известного узла."""
    anchors = _anchors(question, kb)
    asks_about_procedure = re.search(r"регламент|процедур", question.lower())
    if len(anchors) == 1 and anchors[0].startswith("Component:") and asks_about_procedure:
        return anchors[0]
    return None


def anchor_paths(question: str, kb: KnowledgeBase) -> list[list[str]]:
    """Находит короткие пути между известными опорами без учёта направления."""
    undirected = kb.graph.to_undirected()
    paths = []
    for start, end in combinations(_anchors(question, kb), 2):
        paths.extend(nx.all_simple_paths(undirected, start, end, cutoff=3))
    return paths


def community_facts(question: str, kb: KnowledgeBase) -> dict[int, list[tuple[str, set[str]]]]:
    """Готовит типизированные факты и суммы весов без участия модели."""
    graph = kb.graph
    facts: dict[int, list[tuple[str, set[str]]]] = {community.community_id: [] for community in kb.communities}
    owners = {}
    for community in kb.communities:
        for node in community.nodes:
            owners[node] = community.community_id

    lines = {f"ProductLine:{name}" for kind, name in find_names(question, kb.vocabulary) if kind == "ProductLine"}
    lines &= set(graph)
    models = set()
    parts = set()
    path_edges: set[tuple[str, str]] | None = None
    if lines:
        for model, line, edge in graph.edges(data=True):
            if edge.get("type") == "BELONGS_TO" and line in lines:
                models.add(model)

        for model in models:
            for target, edge in graph[model].items():
                if edge.get("type") == "HAS_PART":
                    parts.add(target)

        pending = list(parts)
        while pending:
            part = pending.pop()
            for target, edge in graph[part].items():
                if edge.get("type") == "REPLACED_BY" and target not in parts:
                    parts.add(target)
                    pending.append(target)
    else:
        anchors = _anchors(question, kb)
        component = procedure_component(question, kb)
        if len(anchors) >= 2:
            path_edges = set()
            path_nodes = set()
            for path in anchor_paths(question, kb):
                path_nodes.update(path)
                for first, second in zip(path, path[1:], strict=False):
                    if graph.has_edge(first, second):
                        path_edges.add((first, second))
                    if graph.has_edge(second, first):
                        path_edges.add((second, first))
            pending = [node for node in path_nodes if graph.nodes[node].get("type") == "Part"]
            visited = set()
            while pending:
                part = pending.pop()
                if part in visited:
                    continue
                visited.add(part)
                for target, edge in graph[part].items():
                    if edge.get("type") == "REPLACED_BY":
                        path_edges.add((part, target))
                        pending.append(target)
                    elif edge.get("type") == "SUPPLIED_BY" and "постав" in question.lower():
                        path_edges.add((part, target))
        elif component:
            path_edges = set()
            for error, _, edge in graph.in_edges(component, data=True):
                if edge.get("type") != "INDICATES":
                    continue
                path_edges.add((error, component))
                for target, relation in graph[error].items():
                    if relation.get("type") == "FIXED_BY":
                        path_edges.add((error, target))

    for source, target, edge in sorted(graph.edges(data=True)):
        owner = owners.get(source)
        if owner is None:
            continue
        kind = edge.get("type", "")
        if lines:
            model_fact = source in models and kind in {"BELONGS_TO", "HAS_PART", "FAILED"}
            part_fact = source in parts and kind in {"REPLACED_BY", "SUPPLIED_BY", "IS_A"}
            if not model_fact and not part_fact:
                continue
        elif path_edges is not None and (source, target) not in path_edges:
            continue
        elif path_edges is None and owners.get(target) != owner:
            continue
        weight = edge.get("weight", 1)
        facts[owner].append((f"{source} -{kind}-> {target} (вес {weight})", _sources(edge)))

    for line in sorted(lines):
        by_component: dict[tuple[int, str], int] = defaultdict(int)
        by_model: dict[tuple[int, str], int] = defaultdict(int)
        component_sources: dict[tuple[int, str], set[str]] = defaultdict(set)
        model_sources: dict[tuple[int, str], set[str]] = defaultdict(set)
        for model in sorted(models):
            if not graph.has_edge(model, line) or graph.edges[model, line].get("type") != "BELONGS_TO":
                continue
            owner = owners.get(model)
            if owner is None:
                continue
            membership = graph.edges[model, line]
            for component, edge in graph[model].items():
                if edge.get("type") != "FAILED":
                    continue
                weight = edge.get("weight", 1)
                by_component[owner, component] += weight
                by_model[owner, model] += weight
                component_sources[owner, component].update(_sources(membership, edge))
                model_sources[owner, model].update(_sources(membership, edge))
        for (owner, component), count in sorted(by_component.items()):
            fact = f"{line}: {component}, частичная сумма весов FAILED по моделям линейки = {count}"
            facts[owner].append((fact, component_sources[owner, component]))
        for (owner, model), count in sorted(by_model.items()):
            fact = f"{line}: {model}, частичная сумма весов FAILED по узлам модели = {count}"
            facts[owner].append((fact, model_sources[owner, model]))
    return facts

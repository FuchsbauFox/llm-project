"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph()

    for result in results:
        valid_relations = [relation for relation in result.relations if relation.matches_schema()]
        entities = list(result.entities)
        for relation in valid_relations:
            entities.extend((relation.subject, relation.object))

        for entity in entities:
            if entity.node_id not in graph:
                graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=[])

            sources = graph.nodes[entity.node_id]["sources"]
            if result.chunk_id not in sources:
                sources.append(result.chunk_id)

        for relation in valid_relations:
            subject = relation.subject.node_id
            target = relation.object.node_id

            if not graph.has_edge(subject, target):
                graph.add_edge(subject, target, type=relation.type, weight=0, sources=[])

            edge = graph.edges[subject, target]
            if result.chunk_id not in edge["sources"]:
                edge["sources"].append(result.chunk_id)
                edge["weight"] = len(edge["sources"])

    conflicts = 0
    for subject, target, edge in list(graph.edges(data=True)):
        if edge["type"] != "REPLACED_BY" or subject == target:
            continue

        # Ребро могло быть удалено при обработке обратной связи.
        if not graph.has_edge(subject, target) or not graph.has_edge(target, subject):
            continue

        reverse = graph.edges[target, subject]
        if reverse["type"] != "REPLACED_BY":
            continue

        conflicts += 1

        if edge["weight"] > reverse["weight"]:
            graph.remove_edge(target, subject)
        elif edge["weight"] < reverse["weight"]:
            graph.remove_edge(subject, target)
        else:
            graph.remove_edge(subject, target)
            graph.remove_edge(target, subject)

    graph.graph["conflicts"] = conflicts
    return graph

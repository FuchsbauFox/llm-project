"""Сообщества графа и их сводки. Требование FR-3."""

import re
from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

SUMMARY_PREFIX = "Сводка графа: "
MISSING_SUMMARY = "Сводка недоступна: ответ модели не прошёл проверку."

_SENTENCES = {
    "BELONGS_TO": "Модель {a} входит в линейку {b} (вес {w}).",
    "HAS_PART": "Модель {a} содержит деталь {b} (вес {w}).",
    "IS_A": "Деталь {a} относится к узлу {b} (вес {w}).",
    "INDICATES": "Ошибка {a} указывает на узел {b} (вес {w}).",
    "FIXED_BY": "Ошибка {a} устраняется по регламенту {b} (вес {w}).",
    "REPLACED_BY": "Деталь {a} заменена на деталь {b} (вес {w}).",
    "SUPPLIED_BY": "Деталь {a} поставляется поставщиком {b} (вес {w}).",
    "FAILED": "В модели {a} отказал узел {b} (вес {w}).",
    "TOTAL_FAILED": "В сообществе сумма весов FAILED для узла {b} — {w}.",
}

_ALTERNATIVES = {
    "INDICATES": ("ErrorCode:{a} указывает на узел {b} (вес {w}).",),
    "FIXED_BY": ("ErrorCode:{a} устраняется по регламенту {b} (вес {w}).",),
    "FAILED": ("Модель {a} содержит отказ узла {b} (вес {w}).",),
}


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает граф на сообщества Лувена с фиксированным seed."""
    if not graph:
        return []
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    for source, target, data in graph.edges(data=True):
        weight = data.get("weight", 1)
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)

    # Модулярность не определена при нулевом суммарном весе.
    if undirected.size(weight="weight") == 0:
        groups = [{node} for node in undirected]
    else:
        groups = nx.community.louvain_communities(undirected, weight="weight", seed=7)
    groups = [sorted(group) for group in groups]
    groups.sort(key=lambda group: group[0])
    return [Community(index, group) for index, group in enumerate(groups)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Модель пишет предложения, каждое проверяется по направленному ребру графа."""
    subgraph = graph.subgraph(community.nodes)
    facts = []
    valid_sentences = set()
    totals = Counter()
    for source, target, data in sorted(subgraph.edges(data=True)):
        weight = data.get("weight", 1)
        relation = data.get("type", "")
        facts.append(f"{source} -{relation}-> {target} (вес {weight})")

        template = _SENTENCES.get(relation)
        if template:
            names = {
                "a": subgraph.nodes[source].get("name", source),
                "b": subgraph.nodes[target].get("name", target),
                "w": weight,
            }
            valid_sentences.add(template.format(**names).casefold())
            for alternative in _ALTERNATIVES.get(relation, ()):
                valid_sentences.add(alternative.format(**names).casefold())

        if relation == "FAILED":
            totals[subgraph.nodes[target].get("name", target)] += weight

    for name, total in totals.items():
        valid_sentences.add(_SENTENCES["TOTAL_FAILED"].format(b=name, w=total).casefold())

    if not facts:
        valid_sentences.add("нет связей для обработки.")
        for node in subgraph.nodes:
            valid_sentences.add(f"В сообществе есть {node}.".casefold())
    patterns = "\n".join(f"{kind}: {template}" for kind, template in _SENTENCES.items())
    counts = "\n".join(f"{name}: {total}" for name, total in sorted(totals.items())) or "нет"
    prompt = (
        "Напиши сводку сообщества не более 150 слов: одно предложение на каждую выбранную связь. "
        "Используй только указанные рёбра и готовые численные веса, не объединяй разные связи, "
        "не добавляй выводов. Замени {a}, {b}, {w} в шаблоне именами концов и весом соответствующей связи. "
        "Вместо идентификатора с типом, например Model:AR-210, подставляй имя после двоеточия: AR-210. "
        "Для TOTAL_FAILED используй только готовые суммы. Точно соблюдай направление связи и текст шаблона; "
        "пиши только предложения, без нумерации.\n"
        f"Шаблоны:\n{patterns}\nСуммы FAILED:\n{counts}\nСвязи:\n" + ("\n".join(facts) or "нет")
    )
    if not facts:
        nodes = ", ".join(sorted(subgraph.nodes)) or "нет"
        prompt += (
            f"\nИзолированные узлы: {nodes}. Для каждого напиши предложение точно по форме "
            "«В сообществе есть {идентификатор}.» и добавь «Нет связей для обработки.»"
        )
    response = llm.chat([{"role": "user", "content": prompt}])
    accepted = []
    words = len(SUMMARY_PREFIX.split())
    for candidate in re.split(r"(?<=[.!?])\s+|\n+", response.content.strip()):
        candidate = candidate.strip()
        if candidate.casefold() not in valid_sentences or candidate in accepted:
            continue
        size = len(candidate.split())
        if words + size <= 150:
            accepted.append(candidate)
            words += size
    if not accepted:
        return MISSING_SUMMARY
    return SUMMARY_PREFIX + " ".join(accepted)

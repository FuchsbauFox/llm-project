"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
import re
from collections import defaultdict
from itertools import combinations

import networkx as nx

from assistant.graphrag.communities import MISSING_SUMMARY, SUMMARY_PREFIX
from assistant.graphrag.search_evidence import anchor_paths, community_facts, discontinued_parts, procedure_component
from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "refused": {"type": "boolean"}},
    "required": ["answer", "refused"],
}

_MALFORMED_CODE = re.compile(
    r"(?<![\w-])(?:AR|BZ|GS|SP|P|E)(?:--+\d{2,4}|-\d{1,4}-\d{1,4})[A-Z]?(?!\w)",
    re.IGNORECASE,
)
_UNSUPPORTED_INSTALLATION = re.compile(
    r"\bне\s+(?:устанавлива\w*|установлен\w*|ставится|применяется)\b", re.IGNORECASE
)


def _refusal() -> Answer:
    return Answer("В базе знаний не найдено достаточно данных для ответа.", refused=True)


def _documents(chunks) -> set[str]:
    return {chunk.split("#", 1)[0] for chunk in chunks}


def _grounded_summary(text: str) -> bool:
    return text.startswith((SUMMARY_PREFIX, MISSING_SUMMARY, "Факты графа:"))


def _explicit_refusal(text: str) -> bool:
    lowered = text.lower()
    sentences = [part.strip() for part in re.split(r"[.!?]+\s+|\n+", lowered) if part.strip()]
    if not sentences:
        return False
    affirmative = False
    for sentence in sentences[:-1]:
        has_fact = re.match(r"(?:известно|деталь|модель|ошибка|поставщик|ответ\s*[:—])\b", sentence)
        missing_fact = re.match(
            r"(?:деталь|модель|ошибка|поставщик)\s+\S+\s+не\s+"
            r"(?:найден|найдена|указан|указана|известен|известна)",
            sentence,
        )
        if has_fact and not missing_fact:
            affirmative = True
            break
    conclusion = re.search(
        r"(?:следовательно|итог|вывод|ответ)\s*[:,—]?[^.!?]{0,180}"
        r"(?:невозможно|нельзя|недостаточно|не может быть определен|не может быть дан|"
        r"ответ[^.!?]{0,60}отсутствует|ответить невозможно)",
        sentences[-1],
    )
    if conclusion and not affirmative:
        return True
    patterns = (
        r"^вопрос\b[^.!?]{0,160}\bне\s+может\s+быть\s+(?:отвеч\w*|ответ\w*|реш\w*)\b",
        r"^(?:ответ:\s*)?(?:в базе знаний\s+)?(?:данных|сведений|информации)\s+"
        r"(?:недостаточно|не найдено|нет)\b",
        r"^данные\s+[^.!?]{0,160}\b(?:отсутствуют|не найдены)\b",
        r"^(?:данных|сведений|информации)\s+о\s+[^.!?]{0,160}\b(?:нет|недостаточно|не найдено)\b",
        r"^(?:недостаточно|нет)\s+(?:данных|сведений|информации)\b",
        r"^в\s+(?:предоставленных|имеющихся)\s+данных\s+нет\s+(?:информации|сведений|данных)\b",
    )
    if not any(re.match(pattern, lowered) for pattern in patterns):
        return False
    # Положительная часть после противопоставления может быть полезным частичным ответом.
    if re.search(r"\b(?:но|однако|зато)\s+(?!не\b|нет\b|нельзя\b|недостаточно\b)", lowered):
        return False
    if len(sentences) == 1:
        return True
    if any(re.match(r"(?:известно|деталь|модель|ошибка|поставщик)\b", part) for part in sentences[1:]):
        return False
    return bool(
        re.search(
            r"нельзя|невозможно|недостаточ|нет\s+(?:данных|информации|сведений)|"
            r"не\s+(?:указа|указыва|найден|извест|позвол|определ|содерж)|отсутству",
            sentences[-1],
        )
    )


def _model_answer(content: str, sources: set[str], facts: list[str] | None = None) -> Answer:
    """Принимает структурированный отказ; для старых ответов распознаёт только явный полный отказ."""
    text = content.strip()
    refused = False
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        pass
    else:
        if not isinstance(data, dict):
            return _refusal()
        if not isinstance(data.get("answer"), str) or type(data.get("refused")) is not bool:
            return _refusal()
        text = data["answer"].strip()
        refused = data["refused"]
    if not text:
        return _refusal()
    # Запасной вариант для неструктурированного ответа не должен отбрасывать частичный ответ.
    if refused or _explicit_refusal(text):
        return Answer("В базе знаний не найдено достаточно данных для ответа.", sorted(sources), facts or [], True)
    return Answer(text, sorted(sources), facts or [])


def _check_supplier(question: str, answer: Answer, kb: KnowledgeBase) -> Answer:
    """Не переносит поставщика старой детали на другую без ребра SUPPLIED_BY."""
    parts = [code for code in find_codes(question) if code_type(code) == "Part"]
    suppliers = [name for kind, name in find_names(answer.text, kb.vocabulary) if kind == "Supplier"]
    if answer.refused or "постав" not in question.lower() or len(parts) != 1:
        return answer
    part = f"Part:{parts[0]}"
    if part not in kb.graph:
        return _refusal()
    candidates = {part}
    if re.search(r"замен|актуальн|текущ", question.lower()):
        pending = [part]
        while pending:
            node = pending.pop()
            for target, data in kb.graph[node].items():
                if data.get("type") == "REPLACED_BY" and target not in candidates:
                    candidates.add(target)
                    pending.append(target)

        # Поставщик старой детали не обязательно поставляет её замену.
        candidates = {
            node
            for node in candidates
            if not any(data.get("type") == "REPLACED_BY" for data in kb.graph[node].values())
        }
    supplier_edges = []
    for node in candidates:
        for target, data in kb.graph[node].items():
            if data.get("type") == "SUPPLIED_BY":
                supplier_edges.append((node, target))
    # Для прямого вопроса о поставщике отсутствие ребра означает, что ответа в графе нет.
    if not supplier_edges and re.match(r"^\s*кто\s+постав", question, re.IGNORECASE):
        return Answer("В базе знаний не найдено подтверждения поставщика указанной детали.", refused=True)
    if not suppliers:
        return answer
    for supplier in suppliers:
        target = f"Supplier:{supplier}"
        supported = any((node, target) in supplier_edges for node in candidates)
        if not supported:
            return Answer("В базе знаний не найдено подтверждения поставщика указанной детали.", refused=True)
    return answer


def _check_current_model_supplier(question: str, answer: Answer, kb: KnowledgeBase) -> Answer:
    """Проверяет поставщика последней детали для указанной модели и узла."""
    if answer.refused or "постав" not in question.lower():
        return answer
    if not re.search(r"сейчас|текущ|актуальн|замен", question.lower()):
        return answer

    models = [f"Model:{code}" for code in find_codes(question) if code_type(code) == "Model"]
    components = [f"Component:{name}" for kind, name in find_names(question, kb.vocabulary) if kind == "Component"]
    suppliers = [name for kind, name in find_names(answer.text, kb.vocabulary) if kind == "Supplier"]
    if len(models) != 1 or len(components) != 1 or not suppliers or models[0] not in kb.graph:
        return answer

    graph = kb.graph
    model, component = models[0], components[0]
    installed = []
    for part, edge in graph[model].items():
        if edge.get("type") != "HAS_PART" or not graph.has_edge(part, component):
            continue
        if graph.edges[part, component].get("type") == "IS_A":
            installed.append(part)
    if len(installed) != 1:
        return _refusal()

    part = installed[0]
    chain = [(model, part), (part, component)]
    visited = set()
    while part not in visited:
        visited.add(part)
        replacements = [target for target, edge in graph[part].items() if edge.get("type") == "REPLACED_BY"]
        if not replacements:
            break
        if len(replacements) != 1:
            return _refusal()
        target = replacements[0]
        chain.append((part, target))
        part = target
    else:
        return _refusal()

    supplier_edges = [(part, target) for target, edge in graph[part].items() if edge.get("type") == "SUPPLIED_BY"]
    if len(supplier_edges) != 1:
        return _refusal()
    supplier_node = supplier_edges[0][1]
    supplier_name = graph.nodes[supplier_node].get("name", supplier_node.split(":", 1)[-1])

    # Используем только цепочку, уже включённую в факты local-поиска.
    chain.append(supplier_edges[0])
    facts = set(answer.facts)
    for source, target in chain:
        source_name = graph.nodes[source].get("name", source)
        target_name = graph.nodes[target].get("name", target)
        relation = graph.edges[source, target].get("type", "")
        if f"{source_name} -{relation}-> {target_name}" not in facts:
            return _refusal()

    if all(name == supplier_name for name in suppliers):
        return answer

    part_name = graph.nodes[part].get("name", part.split(":", 1)[-1])
    text = f"По данным графа актуальная деталь — {part_name}, её поставщик — {supplier_name}."
    return Answer(text, answer.sources, answer.facts)


def _check_codes(question: str, answer: Answer) -> Answer:
    if answer.refused:
        return answer
    if _MALFORMED_CODE.search(answer.text):
        return Answer("В базе знаний не найдено подтверждения указанного кода.", refused=True)
    supported = set(find_codes(question + "\n" + "\n".join(answer.facts)))
    if any(code not in supported for code in find_codes(answer.text)):
        return Answer("В базе знаний не найдено подтверждения указанного кода.", refused=True)
    return answer


def has_unconfirmed_installation_claim(text: str) -> bool:
    """Возвращает отрицательное утверждение, которое не выражается рёбрами графа."""
    return bool(_UNSUPPORTED_INSTALLATION.search(text))


def _filter_installation_claims(answer: Answer) -> Answer:
    """В схеме нет отрицательных связей об установке деталей."""
    if answer.refused or not has_unconfirmed_installation_claim(answer.text):
        return answer
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer.text)
    supported = [
        sentence for sentence in sentences if sentence.strip() and not has_unconfirmed_installation_claim(sentence)
    ]
    if not supported:
        return Answer("В базе знаний не найдено подтверждения для ответа.", answer.sources, answer.facts, True)
    return Answer(" ".join(supported), answer.sources, answer.facts)


def _check_replacements(question: str, answer: Answer) -> Answer:
    """Проверяет актуальную деталь по выбранным фактам, а не по ответу модели."""
    if answer.refused:
        return answer
    part_question = re.search(r"детал|запчаст|заказ|замен|актуальн|текущ", question, re.I)
    recommendation = re.compile(
        r"закаж\w*|заказ\w*|использ\w*|рекоменду\w*|установ(?:ить|ите)|"
        r"постав(?:ить|ьте)|замен(?:ите|ить|и)\s+на", re.I
    )
    part_recommendation = recommendation.search(answer.text) or re.search(r"замен", answer.text, re.I)
    if not part_question and not part_recommendation:
        return answer
    replacements: dict[str, set[str]] = defaultdict(set)
    for fact in answer.facts:
        match = re.fullmatch(r"(P-\d{4}[A-Z]?) -REPLACED_BY-> (P-\d{4}[A-Z]?)", fact)
        if match:
            replacements[match.group(1)].add(match.group(2))
    if not replacements:
        return answer

    current_question = bool(re.search(r"заказ|замен|актуальн|текущ", question, re.I)) or bool(part_recommendation)
    question_parts = {code for code in find_codes(question) if code in replacements}
    answer_parts = {code for code in find_codes(answer.text) if code in replacements}
    starts = question_parts or answer_parts
    if not starts:
        roots = set(replacements) - {target for targets in replacements.values() for target in targets}
        has_model = any(code_type(code) == "Model" for code in find_codes(question))
        if len(roots) != 1 or not has_model:
            if current_question:
                return Answer(
                    "В базе знаний не удалось однозначно определить замену.", answer.sources, answer.facts, True
                )
            return answer
        starts = roots

    terminals = {}
    for start in starts:
        part = start
        visited = set()
        while part in replacements:
            if part in visited or len(replacements[part]) != 1:
                return Answer(
                    "В базе знаний не удалось однозначно определить замену.", answer.sources, answer.facts, True
                )
            visited.add(part)
            part = next(iter(replacements[part]))
        terminals[start] = part

    codes_in_answer = set(find_codes(answer.text))
    missing = {start: end for start, end in terminals.items() if end not in codes_in_answer}
    wrong_recommendation = False
    if part_recommendation:
        for match in recommendation.finditer(answer.text):
            following = find_codes(answer.text[match.end() : match.end() + 80])
            if following and following[0] in replacements:
                wrong_recommendation = True
                break
        for part in replacements:
            pattern = rf"\b{re.escape(part)}\s+(?:следует|нужно|необходимо)?\s*(?:заказ\w*|использ\w*|установить)"
            if re.search(pattern, answer.text, re.I):
                wrong_recommendation = True
                break
    if (current_question and "постав" not in question.lower() and missing) or wrong_recommendation:
        choices = set(terminals.values())
        if len(choices) != 1:
            return Answer("В базе знаний не удалось однозначно определить замену.", answer.sources, answer.facts, True)
        current = next(iter(choices))
        return Answer(f"По данным графа актуальная деталь — {current}.", answer.sources, answer.facts)
    if missing:
        notes = " ".join(f"Замена детали {old} — {new}." for old, new in sorted(missing.items()))
        return Answer(f"{answer.text} {notes}", answer.sources, answer.facts)
    return answer


def _repair_malformed_codes(question: str, answer: Answer, llm: LLMClient) -> Answer:
    """Просит исправить неверный формат кода, не угадывая его самостоятельно."""
    if answer.refused or not _MALFORMED_CODE.search(answer.text):
        return answer
    prompt = (
        "В ответе неверно записан код. Исправь ответ по вопросу и фактам, не придумывая кодов. "
        "Если данных недостаточно, верни refused=true. Верни JSON с полями answer и refused.\n"
        f"Вопрос: {question}\nФакты:\n" + "\n".join(answer.facts) + f"\nОтвет: {answer.text}"
    )
    response = llm.chat([{"role": "user", "content": prompt}], json_schema=ANSWER_SCHEMA)
    corrected = _model_answer(response.content, set(answer.sources), answer.facts)
    if _MALFORMED_CODE.search(corrected.text):
        return _refusal()
    return corrected


def _check_global_claims(answer: Answer) -> Answer:
    """В данной схеме нет ни отрицательных связей, ни статуса снятия с производства."""
    if answer.refused:
        return answer
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer.text.lower()):
        if re.search(r"нет данных|недостаточно|неизвестно|не подтвержден|не доказан|не указано", sentence):
            continue
        negative = re.search(
            r"\bне\s+(?:ставится|установлен[а-яё]*|снят[а-яё]*|"
            r"устраняется|применяется)\b",
            sentence,
        )
        discontinued = re.search(r"\bснят(?:а|о|ы)?\s+с\s+производств", sentence)
        if negative or discontinued:
            return _refusal()
    return answer


def _selected_edges(answer: Answer) -> list[tuple[str, str, str]]:
    edges = []
    for line in re.split(r"\s*;\s*|\n", "\n".join(answer.facts)):
        match = re.fullmatch(r"(.+?) -([A-Z_]+)-> (.+?) \(вес [^()]+\)", line)
        if match:
            edges.append(match.groups())
    return edges


def _check_global_procedures(question: str, answer: Answer, kb: KnowledgeBase) -> Answer:
    """Регламент узла подтверждается только двумя выбранными связями одной ошибки."""
    if answer.refused or not re.search(r"регламент|процедур", question.lower()):
        return answer
    components = {f"Component:{name}" for kind, name in find_names(question, kb.vocabulary) if kind == "Component"}
    procedures = {f"Procedure:{code}" for code in find_codes(answer.text) if code_type(code) == "Procedure"}
    if len(components) != 1 or not procedures:
        return answer
    edges = _selected_edges(answer)
    errors = {source for source, kind, target in edges if kind == "INDICATES" and target in components}
    supported = {target for source, kind, target in edges if kind == "FIXED_BY" and source in errors}
    if procedures <= supported:
        return answer
    return _refusal()


def _check_global_component_suppliers(question: str, answer: Answer, kb: KnowledgeBase) -> Answer:
    """Поставщик узла модели должен следовать из выбранных связей, включая замены."""
    if answer.refused or "постав" not in question.lower():
        return answer
    models = {f"Model:{code}" for code in find_codes(question) if code_type(code) == "Model"}
    components = {f"Component:{name}" for kind, name in find_names(question, kb.vocabulary) if kind == "Component"}
    suppliers = {f"Supplier:{name}" for kind, name in find_names(answer.text, kb.vocabulary) if kind == "Supplier"}
    if len(models) != 1 or not suppliers:
        return answer
    edges = _selected_edges(answer)
    if not components:
        errors = {f"ErrorCode:{code}" for code in find_codes(question) if code_type(code) == "ErrorCode"}
        if len(errors) != 1:
            return answer
        components = {target for source, kind, target in edges if source in errors and kind == "INDICATES"}
        if not components:
            return _refusal()
    elif len(components) != 1:
        return answer

    installed = {target for source, kind, target in edges if source in models and kind == "HAS_PART"}
    parts = {
        source for source, kind, target in edges if source in installed and kind == "IS_A" and target in components
    }
    pending = list(parts)
    while pending:
        part = pending.pop()
        for source, kind, target in edges:
            if source == part and kind == "REPLACED_BY" and target not in parts:
                parts.add(target)
                pending.append(target)
    if re.search(r"сейчас|текущ|актуальн|замен", question.lower()):
        parts -= {source for source, kind, _ in edges if source in parts and kind == "REPLACED_BY"}
    supported = {target for source, kind, target in edges if source in parts and kind == "SUPPLIED_BY"}
    if suppliers <= supported:
        return answer
    return _refusal()


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает по путям между опорами или по окрестности единственной опоры."""
    graph = kb.graph
    codes = find_codes(question)
    code_nodes = [f"{code_type(code)}:{code}" for code in codes]
    if any(node not in graph for node in code_nodes):
        return _refusal()
    anchors = []
    for node in code_nodes:
        if node not in anchors:
            anchors.append(node)
    for kind, name in find_names(question, kb.vocabulary):
        node = f"{kind}:{name}"
        if node in graph and node not in anchors:
            anchors.append(node)
    if not anchors:
        return _refusal()

    undirected = graph.to_undirected()
    edges: set[tuple[str, str]] = set()
    if len(anchors) == 1:
        nodes = set(nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=2))
        edges.update(graph.subgraph(nodes).edges())
    else:
        path_nodes: set[str] = set()
        for start, end in combinations(anchors, 2):
            for path in nx.all_simple_paths(undirected, start, end, cutoff=3):
                path_nodes.update(path)
                for source, target in zip(path, path[1:], strict=False):
                    if graph.has_edge(source, target):
                        edges.add((source, target))
                    if graph.has_edge(target, source):
                        edges.add((target, source))
        # Обходим направленные цепочки замен, в том числе разветвления и циклы.
        pending = [node for node in path_nodes if graph.nodes[node].get("type") == "Part"]
        visited: set[str] = set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            for target, data in graph[node].items():
                if data.get("type") == "REPLACED_BY":
                    edges.add((node, target))
                    pending.append(target)
                elif data.get("type") == "SUPPLIED_BY" and "постав" in question.lower():
                    edges.add((node, target))

    # Замены идут первыми, чтобы они не потерялись при ограничении контекста.
    ordered = sorted(edges, key=lambda edge: (graph.edges[edge].get("type") != "REPLACED_BY", edge))
    facts: list[str] = []
    sources: set[str] = set()
    length = 0
    for source, target in ordered:
        data = graph.edges[source, target]
        source_name = graph.nodes[source].get("name", source)
        target_name = graph.nodes[target].get("name", target)
        fact = f"{source_name} -{data.get('type', '')}-> {target_name}"
        line_length = len(f"{len(facts) + 1}. {fact}\n")
        if length + line_length > 8000:
            continue
        length += line_length
        facts.append(fact)
        sources.update(_documents(data.get("sources", [])))
    if not facts:
        return _refusal()
    numbered = "\n".join(f"{index}. {fact}" for index, fact in enumerate(facts, 1))
    prompt = (
        "Ответь на вопрос только по перечисленным фактам. Если данных недостаточно, скажи об этом. "
        "Направление стрелки важно. Значения связей: INDICATES — ошибка указывает на узел; "
        "FIXED_BY — ошибка устраняется по регламенту; HAS_PART — модель содержит деталь; "
        "IS_A — деталь относится к типу узла; REPLACED_BY — старая деталь заменена новой; "
        "SUPPLIED_BY — деталь поставляет поставщик; BELONGS_TO — модель входит в линейку; "
        "FAILED — в модели отказал узел. Если вопрос о текущей детали, следуй цепочке "
        "REPLACED_BY до последней детали. Поиск не оформляет заказы и не знает остатки и цены: "
        "не утверждай, что заказ оформлен, и не указывай количество, наличие или цену.\n"
        "Верни JSON: answer — текст ответа; refused=true, только если ответить совсем невозможно. "
        "Если часть ответа известна, сообщи её и укажи ограничения, refused=false.\n"
        f"Вопрос: {question}\nФакты:\n{numbered}"
    )
    response = llm.chat([{"role": "user", "content": prompt}], json_schema=ANSWER_SCHEMA)
    answer = _model_answer(response.content, sources, facts)
    answer = _repair_malformed_codes(question, answer, llm)
    answer = _check_current_model_supplier(question, answer, kb)
    answer = _check_supplier(question, _check_codes(question, answer), kb)
    answer = _filter_installation_claims(answer)
    return _check_replacements(question, answer)


def _bundle_facts(facts: list[tuple[str, set[str]]]) -> list[tuple[str, set[str]]]:
    """Объединяет связанные факты сообщества и их источники в один пакет."""
    texts = []
    sources = set()
    for fact, documents in facts:
        texts.append(fact)
        sources.update(documents)
    return [("; ".join(texts), sources)]


def _add_comparison_totals(partials: list[str]) -> None:
    """Суммирует только веса из принятых ответов map."""
    totals: dict[tuple[str, str, str], int] = defaultdict(int)
    pattern = re.compile(
        r"(ProductLine:[^:]+): ((?:Component|Model):.+?), "
        r"частичная сумма весов FAILED (по моделям линейки|по узлам модели) = (\d+)"
    )
    for fact in partials:
        for line in re.split(r"\s*;\s*|\n", fact):
            match = pattern.fullmatch(line)
            if match:
                product, entity, scope, count = match.groups()
                totals[product, entity, scope] += int(count)

    for (line, entity, scope), count in sorted(totals.items()):
        partials.append(f"{line}: {entity}, сумма принятых частичных весов FAILED {scope} = {count}")


def _reduce_prompt(question: str, partials: list[str]) -> str:
    """Собирает запрос для объединения принятых фактов."""
    question_lower = question.lower()
    prompt = (
        "Ответь по фактам, объединив частичные ответы. Направление связей важно: "
        "Part — деталь, Model — модель, Component — вид узла; не смешивай эти типы. "
        "REPLACED_BY направлена от старой детали к новой. "
        "INDICATES означает, что ошибка указывает на узел, а не наоборот. "
        "Для регламента узла нужны обе связи одной ошибки: INDICATES и FIXED_BY. "
        "Если спрашивают актуальную деталь, проследи все принятые связи REPLACED_BY до конца цепочки. "
        "Замена сама по себе не подтверждает отдельную связь IS_A или SUPPLIED_BY новой детали. "
        "Вес относится к связи и равен числу подтверждающих чанков, а не является свойством узла. "
        "FAILED означает отказ узла в модели, а не снятие детали с производства. "
        "FAILED не означает, что узел отсутствует в модели. Отсутствие связи в данных не доказывает "
        "отрицательного факта: не утверждай, что деталь не установлена или регламент не применяется. "
        "В графе нет отдельного признака снятия с производства: REPLACED_BY подтверждает только замену. "
        "Более частые отказы не означают большую отказоустойчивость. "
        "Не приписывай поставщика другой детали и не смешивай линейки. "
        "Для принадлежности детали линейке нужны связи модели с деталью и линейкой. "
        "Суммы весов FAILED подсчитаны кодом только по принятым фактам, которые могут быть неполными; "
        "не считай число моделей вместо весов и не утверждай полноту данных. "
        "Поиск не оформляет заказы и не знает остатки и цены; не подтверждай заказ и не указывай количество. "
        "Отвечай кратко, используя только необходимые подтверждённые факты, без дополнительных предположений. "
        "Не придумывай связи. Верни JSON: answer — текст, refused=true только если ответить совсем невозможно; "
        "если известна часть ответа, укажи её и ограничения, refused=false.\n"
        f"Вопрос: {question}\nЧастичные ответы:\n" + "\n".join(partials)
    )
    if "постав" in question_lower and re.search(r"сейчас|текущ|актуальн|замен", question_lower):
        prompt += (
            "\nЕсли известна цепочка замен, назови поставщика последней детали в цепочке. "
            "Поставщик старой детали может отличаться. Модель может быть связана со старой деталью, "
            "а применимость новой подтверждается цепочкой замен."
        )
    elif "детал" in question_lower and re.search(r"сейчас|текущ|актуальн", question_lower):
        prompt += (
            "\nНазови текущую деталь и цепочку её замен. Не приписывай каждой детали отдельную связь IS_A, "
            "если она прямо не указана в фактах."
        )
    if re.search(r"модел", question_lower) and re.search(r"снят[а-яё]*\s+с\s+производств", question_lower):
        prompt += (
            "\nСведения о снятии с производства в вопросе не подтверждены графом. "
            "Не повторяй и не подтверждай их; ответь только на вопрос по имеющимся связям."
        )
    if re.search(r"регламент|процедур", question_lower):
        prompt += "\nСвязь FIXED_BY означает, что ошибку устраняют по регламенту; не называй это фиксацией ошибки."
    return prompt


def _map_prompt(
    question: str,
    summary: str,
    facts: list[tuple[str, set[str]]],
    paths: list[list[str]],
    *,
    comparison: bool,
    statistic: bool,
    procedure: bool,
) -> str:
    """Готовит вопрос модели по одному сообществу и его фактам."""
    numbered = "\n".join(f"{i}: {fact}" for i, (fact, _) in enumerate(facts, 1)) or "нет"
    prompt = (
        "Выбери факты, которые помогают ответить на вопрос. Направление и типы сущностей важны: "
        "Model — модель, Part — деталь; REPLACED_BY направлена от старой детали к новой. "
        "Сравнивай только суммы FAILED, относящиеся к нужной линейке. "
        "Верни JSON: answer — только номера выбранных фактов через запятую; score от 1 до 10. "
        "Если подходящих фактов нет, верни пустой answer и score 0. Не придумывай номера.\n"
        f"Вопрос: {question}\nСводка: {summary}\nПроверенные факты:\n{numbered}"
    )
    if comparison and any("сумма весов FAILED" in fact for fact, _ in facts):
        prompt += "\nДля сравнения выбери все подходящие суммы, а не только наибольшую."
        if statistic:
            prompt += " Здесь единственный допустимый номер пакета — 1; выбери его целиком."
    if paths and facts:
        prompt += (
            "\nЭти факты лежат на путях между сущностями вопроса или в цепочках замен деталей. "
            "Здесь единственный допустимый номер пакета — 1; он обозначает все связи данного сообщества. "
            "Выбери пакеты, необходимые для соединения сущностей и определения актуальной детали."
        )
        if "постав" in question.lower():
            prompt += " В пакет также включены подтверждённые связи деталей с поставщиками."
        prompt += (
            "\nПути опор (порядок соседства, не направление связей):\n"
            + "\n".join(" ↔ ".join(path) for path in paths)
            + "\nПути служат только контекстом. Выбирай исключительно номера проверенных фактов этого "
            "сообщества; если их нет, верни score 0."
        )
    elif procedure and facts:
        prompt += (
            "\nПакет содержит связи ошибки с указанным узлом и её регламентом. "
            "Здесь единственный допустимый номер пакета — 1; выбери его целиком, "
            "если он помогает ответить на вопрос."
        )
    return prompt


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Объединяет полезные ответы модели на отдельные сводки сообществ."""
    partials: list[str] = []
    sources: set[str] = set()
    question_lower = question.lower()
    evidence = community_facts(question, kb)
    unknown_code = any(f"{code_type(code)}:{code}" not in kb.graph for code in find_codes(question))
    comparison = re.search(r"чаще|реже|больше|меньше|наибол|наимен|частот|сравн", question_lower)
    names = find_names(question, kb.vocabulary)
    has_product_line = any(kind == "ProductLine" for kind, _ in names)
    path_search = len(find_codes(question)) + len(names) >= 2 and not has_product_line
    paths = anchor_paths(question, kb) if path_search else []
    procedure_search = procedure_component(question, kb) is not None
    statistic = ""
    if comparison and re.search(r"отказ|неисправ", question_lower):
        if re.search(r"\bмодел[а-яё]*", question_lower):
            statistic = "частичная сумма весов FAILED по узлам модели"
        elif re.search(r"\b(?:узел|узла|узлы|узлов|узле|узлом|компонент[а-яё]*)\b", question_lower):
            statistic = "частичная сумма весов FAILED по моделям линейки"
    relevant_statistics: set[int] = set()
    accepted_statistics: set[int] = set()
    for community in kb.communities:
        grounded = _grounded_summary(community.summary)
        available = evidence[community.community_id]
        if grounded:
            # Связи одного сообщества на одном пути предлагаются как единый факт-пакет:
            # модель не теряет замену, выбрав только одно звено принятого сообщества.
            if (paths or procedure_search) and available:
                available = _bundle_facts(available)
            elif statistic:
                selected = [(fact, documents) for fact, documents in available if statistic in fact]
                if selected:
                    relevant_statistics.add(community.community_id)
                    available = _bundle_facts(selected)
                else:
                    available = []
            prompt = _map_prompt(
                question,
                community.summary,
                available,
                paths,
                comparison=bool(comparison),
                statistic=bool(statistic),
                procedure=procedure_search,
            )
        else:
            prompt = (
                f"Ответь по сводке в JSON answer и score от 0 до 10.\nВопрос: {question}\nСводка: {community.summary}"
            )
        response = llm.chat([{"role": "user", "content": prompt}], json_schema=MAP_SCHEMA)
        if unknown_code:
            continue
        try:
            item = json.loads(response.content)
            answer, score = item["answer"], item["score"]
            if not isinstance(answer, str) or not answer.strip() or type(score) is not int or not 0 < score <= 10:
                continue
        except (ValueError, KeyError, TypeError):
            continue
        if grounded:
            selected = answer.strip().strip("[]")
            if not re.fullmatch(r"\s*\d+(?:\s*[,;]\s*\d+)*\s*", selected):
                continue
            indices = []
            for value in re.findall(r"\d+", selected):
                index = int(value)
                if index not in indices:
                    indices.append(index)
            for index in indices:
                if 1 <= index <= len(available):
                    fact, documents = available[index - 1]
                    partials.append(fact)
                    sources.update(documents)
                    if statistic and community.community_id in relevant_statistics:
                        accepted_statistics.add(community.community_id)
        else:
            partials.append(answer)
            for _, _, data in kb.graph.subgraph(community.nodes).edges(data=True):
                sources.update(_documents(data.get("sources", [])))
    if not partials:
        return _refusal()
    if comparison:
        _add_comparison_totals(partials)
    prompt = _reduce_prompt(question, partials)
    response = llm.chat([{"role": "user", "content": prompt}], json_schema=ANSWER_SCHEMA)
    answer = _model_answer(response.content, sources, partials)
    answer = _repair_malformed_codes(question, answer, llm)
    parts, evidence_sources = discontinued_parts(question, kb, partials)
    if parts:
        text = "По подтверждённым документам сняты с производства: " + ", ".join(parts)
        text += ". Перечень может быть неполным."
        return Answer(text, sorted(sources | evidence_sources), partials)
    if all(_grounded_summary(community.summary) for community in kb.communities):
        answer = _check_global_claims(_check_codes(question, answer))
        answer = _check_global_procedures(question, answer, kb)
        answer = _check_global_component_suppliers(question, answer, kb)
    if statistic and relevant_statistics - accepted_statistics and not answer.refused:
        selected_facts = [line for fact in partials for line in re.split(r"\s*;\s*|\n", fact) if statistic in line]
        text = "По принятым сообществам доступны только частичные данные: "
        text += "; ".join(selected_facts) + ". Для полного сравнения данных недостаточно."
        return Answer(text, sorted(sources), partials)
    return _check_supplier(question, answer, kb)

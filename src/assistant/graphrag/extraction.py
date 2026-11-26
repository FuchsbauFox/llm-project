"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json
import re

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

PROMPT = """Извлеки из текста сущности и связи между ними. Верни JSON по заданной схеме:
сущности в entities, связи в relations. Указывай только факты из текста; соседние
упоминания сами по себе не означают связь.

Типы сущностей: Model - модель, ProductLine - линейка, Part - деталь,
Component - вид узла, ErrorCode - код ошибки, Procedure - регламент,
Supplier - поставщик. Номера бюллетеней и отчетов не являются сущностями.

Допустимые связи и их направления:
Model --BELONGS_TO--> ProductLine - модель входит в линейку;
Model --HAS_PART--> Part - деталь установлена в модели;
Part --IS_A--> Component - деталь относится к виду узла;
ErrorCode --INDICATES--> Component - ошибка указывает на отказ узла;
ErrorCode --FIXED_BY--> Procedure - ошибка устраняется по регламенту;
Part --REPLACED_BY--> Part - старая деталь заменена новой;
Part --SUPPLIED_BY--> Supplier - поставщик указанной детали;
Model --FAILED--> Component - в отчете зафиксирован отказ узла модели.

В руководстве разбирай каждую строку состава: для детали укажи модель, вид узла
и поставщика. Например, для модели AR-210, детали P-1042, пускового реле и
поставщика ХолодКомплект нужны три связи: AR-210 HAS_PART P-1042,
P-1042 IS_A пусковое реле и P-1042 SUPPLIED_BY ХолодКомплект. Также связывай
каждую указанную ошибку с ее узлом. В бюллетене указывай замену от старой детали
к новой и поставщика именно той детали, к которой он относится. В регламенте
связывай ошибку с регламентом и узлом. В отчете связывай модель с отказавшим узлом.

Пример: «Деталь P-1042 снята с производства, вместо нее устанавливается P-1042B».
Ответ: {"entities":[{"type":"Part","name":"P-1042"},{"type":"Part","name":"P-1042B"}],
"relations":[{"subject_type":"Part","subject":"P-1042","type":"REPLACED_BY",
"object_type":"Part","object":"P-1042B"}]}.
Примеры не относятся к тексту, который нужно обработать.
"""


def _parse_response(content: str) -> dict:
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("Ответ должен быть объектом")

    entities, relations = data.get("entities"), data.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        raise ValueError("Нужны массивы entities и relations")

    for item in entities:
        if not isinstance(item, dict) or not all(isinstance(item.get(key), str) for key in ("type", "name")):
            raise ValueError("Некорректная сущность")
        if item["type"] not in ENTITY_TYPES:
            raise ValueError("Неизвестный тип сущности")
    keys = ("subject_type", "subject", "type", "object_type", "object")
    for item in relations:
        if not isinstance(item, dict) or not all(isinstance(item.get(key), str) for key in keys):
            raise ValueError("Некорректная связь")
        if (
            item["subject_type"] not in ENTITY_TYPES
            or item["object_type"] not in ENTITY_TYPES
            or item["type"] not in RELATION_SIGNATURES
        ):
            raise ValueError("Неизвестный тип связи или ее конца")

    return data


def _asserted_text(text: str) -> str:
    """Исключает предложения с отрицанием или сомнением перед разбором фактов."""
    text = re.sub(r"(?i)(\b(?:что|будто|якобы)|[,:])\s*\n\s*", r"\1 ", text)
    uncertain = re.compile(
        r"\b(?:неверн\w*|неправд\w*|неправильн\w*|недостоверн\w*|неподтвержд\w*|"
        r"недоказан\w*|неясн\w*|вымысл\w*|ложн\w*|ошибочн\w*|опроверг\w*|отверг\w*|"
        r"отсутств\w*|сомнительн\w*|якобы|"
        r"возможн\w*|вероятн\w*|предполож\w*|может|могут|неизвестн\w*|"
        r"нельзя|невозможн\w*|исключен\w*|исключён\w*|если|лишь\s+при|"
        r"отрица\w*|оспарива\w*|предполага\w*|запрещен\w*|запрещён\w*)\b|"
        r"\b(?:не\s+(?:факт|следует|правда|известно|установлено)|нет\s+основан\w*|"
        r"нет\s+(?:данных|сведен\w*|доказательств\w*|подтвержден\w*|подтверждён\w*|уверенности)|"
        r"при\s+условии)\b|^\s*(?:нет\s*,|-\s*не\b)",
        re.I,
    )
    negated = re.compile(
        r"\b(?:не|нет|без)\b[^,.;\n]{0,100}\b"
        r"(?:поставщик\w*|поставля\w*|замен\w*|являет\w*|входит|установлен\w*|указ\w*|"
        r"применя\w*|показал\w*|подтвержд\w*|доказ\w*|отказ\w*|соответству\w*)\b",
        re.I,
    )
    reference = re.compile(
        r"^\s*(?:однако\s*,?\s*)?(?:это|данное|предыдущее|такое|сказанное|утверждение)\b", re.I
    )
    parts = re.split(r"([.!?\n])", text)
    result = []
    for index in range(0, len(parts), 2):
        sentence = parts[index]
        end = parts[index + 1] if index + 1 < len(parts) else ""
        doubtful = uncertain.search(sentence) or negated.search(sentence) or re.search(r"\bне\s+так\b", sentence, re.I)
        if reference.search(sentence) and doubtful:
            for previous in range(index - 2, -1, -2):
                if parts[previous].strip():
                    result[previous] = " "
                    break
        result.append(" " if end == "?" or uncertain.search(sentence) or negated.search(sentence) else sentence)
        result.append(end)
    return "".join(result)


def _check_relation(text: str, relation: Relation, llm: LLMClient) -> bool:
    """Просит модель проверить связь, которую не распознали явные правила."""
    # Для замен направление особенно важно: на свободных формулировках модель
    # иногда подтверждает оба варианта. Оставляем только разобранные правилами.
    if relation.type == "REPLACED_BY":
        return False
    indicators = {
        "BELONGS_TO": r"входит|принадлеж|относится к линейке",
        "HAS_PART": r"установлен|содержит|состоит|комплекту|оснащ",
        "IS_A": r"являет|относится|представляет|вид узла",
        "INDICATES": r"указывает|означает|сигнализирует",
        "FIXED_BY": r"регламент|устраня|ремонт|процедур",
        "SUPPLIED_BY": r"поставля|поставляет|поставлен|снабжа|снабжен|поставщиком[^.!?\n]*является",
        "FAILED": r"отказ|неисправ",
    }
    indicator = indicators.get(relation.type)
    if not indicator:
        return False

    schema = {
        "type": "object",
        "properties": {"supported": {"type": "boolean"}, "evidence": {"type": "string"}},
        "required": ["supported", "evidence"],
    }
    prompt = (
        "Проверь связь по тексту. supported=true только если текст прямо сообщает связь в указанном направлении. "
        "Упоминание двух сущностей в одном перечне не доказывает связь между ними. "
        "Если связь отрицается, опровергается, предполагается или прямо не указана, ответь false "
        "и evidence пустая строка. Если связь прямо указана, скопируй предложение с этой связью в evidence. "
        "Верни только JSON."
    )
    fact = f"{relation.subject.node_id} {relation.type} {relation.object.node_id}"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Текст: {text}\nСвязь: {fact}"},
    ]
    response = llm.chat(messages, json_schema=schema, max_tokens=80)
    try:
        data = json.loads(response.content)
    except (ValueError, TypeError):
        return False
    if not isinstance(data, dict) or data.get("supported") is not True:
        return False
    evidence = data.get("evidence")
    if not isinstance(evidence, str) or not evidence or evidence not in _asserted_text(text):
        return False
    if not re.search(indicator, evidence, re.I):
        return False
    for entity in (relation.subject, relation.object):
        if entity.type in CODE_TYPES:
            if entity.name not in find_codes(evidence):
                return False
        elif entity.name.casefold() not in evidence.casefold():
            return False
    return True


def _explicit_relations(text: str, entities: dict[str, Entity], names: dict[str, list[str]]) -> list[Relation]:
    """Дополняет ответ только фактами из однозначных формулировок текста."""
    text = _asserted_text(text)
    relations: list[Relation] = []

    def add(subject: Entity, kind: str, obj: Entity, *, require_both: bool = True) -> None:
        if subject.node_id not in entities or (require_both and obj.node_id not in entities):
            return
        relation = Relation(subject, kind, obj)
        if relation not in relations:
            relations.append(relation)

    def code(raw: str, kind: str) -> Entity | None:
        value = canonical_name(kind, raw, names)
        return Entity(kind, value) if value else None

    def named(raw: str, kind: str) -> Entity | None:
        value = next((name for name in names.get(kind, []) if name.casefold() == raw.strip().casefold()), None)
        return Entity(kind, value) if value else None

    membership = re.search(
        r"([A-Za-zА-Яа-я]+[- ]?\d{3})\s+входит в линейку\s*[«\"„“]([^»\"“”]+)[»\"“”]", text, re.I
    )
    if membership:
        model, line = code(membership.group(1), "Model"), named(membership.group(2), "ProductLine")
        if model and line:
            add(model, "BELONGS_TO", line)

    # Строка состава связывает одну модель, один вид узла, деталь и поставщика.
    composition = re.search(r"В\s+([A-Za-zА-Яа-я]+[- ]?\d{3})\s+установлены следующие узлы\.", text, re.I)
    if composition:
        model = code(composition.group(1), "Model")
        row_pattern = (
            r"^\s*-\s*(.+?)\s*[–—-]\s*деталь\s+([A-Za-zА-Яа-я]+[- ]?\d{4}[A-Za-zА-Яа-я]?)"
            r"\s*,\s*поставщик\s*[«\"„“]([^»\"“”]+)[»\"“”]"
        )
        for match in re.finditer(row_pattern, text, re.I | re.M):
            component = named(match.group(1), "Component")
            part = code(match.group(2), "Part")
            supplier = named(match.group(3), "Supplier")
            if model and part:
                add(model, "HAS_PART", part)
            if component and part:
                add(part, "IS_A", component, require_both=False)
            if part and supplier:
                add(part, "SUPPLIED_BY", supplier)

    # Направление задаётся словами «снята ... вместо неё устанавливается».
    replacement = re.search(
        r"деталь\s+([A-Za-zА-Яа-я]+[- ]?\d{4}[A-Za-zА-Яа-я]?)\s*(?:\([^\n)]*\)\s*)?"
        r"снята с производства\.\s*Вместо неё устанавливается деталь\s+"
        r"([A-Za-zА-Яа-я]+[- ]?\d{4}[A-Za-zА-Яа-я]?)(?![A-Za-zА-Яа-я0-9])",
        text, re.I,
    )
    if replacement:
        old, new = (code(value, "Part") for value in replacement.groups())
        if old and new and old != new:
            add(old, "REPLACED_BY", new)

    part_code = r"(?<![A-Za-zА-Яа-я0-9])([A-Za-zА-Яа-я]+[- ]?\d{4}[A-Za-zА-Яа-я]?)(?![A-Za-zА-Яа-я0-9])"
    for match in re.finditer(
        rf"{part_code}\s+(?:заменяет|заменил[ао]?)\s+(?:деталь\s+)?{part_code}", text, re.I
    ):
        new, old = (code(value, "Part") for value in match.groups())
        if old and new and old != new:
            add(old, "REPLACED_BY", new)
    for match in re.finditer(
        rf"{part_code}\s+(?:заменен[ао]?|заменён[ао]?|заменяется|заменили)\s+"
        rf"(?:на\s+(?:деталь\s+)?|деталью\s+){part_code}", text, re.I
    ):
        old, new = (code(value, "Part") for value in match.groups())
        if old and new and old != new:
            add(old, "REPLACED_BY", new)

    for match in re.finditer(
        r"детал[ьи]\s+([A-Za-zА-Яа-я]+[- ]?\d{4}[A-Za-zА-Яа-я]?)\s*\(([^()\n]+)\)", text, re.I
    ):
        part, component = code(match.group(1), "Part"), named(match.group(2), "Component")
        if part and component:
            add(part, "IS_A", component, require_both=False)

    for match in re.finditer(
        r"поставщик детали\s+([A-Za-zА-Яа-я]+[- ]?\d{4}[A-Za-zА-Яа-я]?)\s*[–—-]\s*[«\"„“]([^»\"“”]+)[»\"“”]",
        text, re.I,
    ):
        part, supplier = code(match.group(1), "Part"), named(match.group(2), "Supplier")
        if part and supplier:
            add(part, "SUPPLIED_BY", supplier)

    defect = re.search(
        r"поставщик\s*[«\"„“]([^»\"“”]+)[»\"“”]\s+сообщил о дефекте партии детали\s+"
        r"([A-Za-zА-Яа-я]+[- ]?\d{4}[A-Za-zА-Яа-я]?)(?![A-Za-zА-Яа-я0-9])",
        text, re.I,
    )
    if defect:
        supplier, part = named(defect.group(1), "Supplier"), code(defect.group(2), "Part")
        if part and supplier:
            add(part, "SUPPLIED_BY", supplier)

    for match in re.finditer(
        r"^\s*-\s*([A-Za-zА-Яа-я]+[- ]?\d{2})\s*[–—-][^\n]*?отказ узла\s*[«\"„“]([^»\"“”]+)[»\"“”]",
        text, re.I | re.M,
    ):
        error, component = code(match.group(1), "ErrorCode"), named(match.group(2), "Component")
        if error and component:
            add(error, "INDICATES", component)

    for match in re.finditer(
        r"ошибка\s+([A-Za-zА-Яа-я]+[- ]?\d{2})\s+указывает на отказ узла\s*[«\"„“]([^»\"“”]+)[»\"“”]",
        text, re.I,
    ):
        error, component = code(match.group(1), "ErrorCode"), named(match.group(2), "Component")
        if error and component:
            add(error, "INDICATES", component)

    procedure = re.search(
        r"регламент\s+([A-Za-zА-Яа-я]+[- ]?\d{3})\s+применяется при ошибке\s+"
        r"([A-Za-zА-Яа-я]+[- ]?\d{2})(?![A-Za-zА-Яа-я0-9])",
        text, re.I,
    )
    if procedure:
        repair, error = code(procedure.group(1), "Procedure"), code(procedure.group(2), "ErrorCode")
        if error and repair:
            add(error, "FIXED_BY", repair)

    equipment = re.search(r"оборудование:\s*[^.\n]+", text, re.I)
    failure = re.search(r"диагностика показала отказ узла:\s*([^.,\n]+)", text, re.I)
    if equipment and failure:
        models = [code(value, "Model") for value in find_codes(equipment.group())]
        models = [model for model in models if model]
        component = named(failure.group(1), "Component")
        if len(models) == 1 and component:
            add(models[0], "FAILED", component)

    return relations


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает подтвержденные сущности и связи чанка по схеме графа."""
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": chunk.text}]

    for attempt in range(2):
        response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=4096)
        try:
            data = _parse_response(response.content)
            break
        except (ValueError, TypeError):
            if attempt == 1:
                return ExtractionResult(chunk.chunk_id, chunk.doc_id)
            messages.append({"role": "user", "content": "Ответ не соответствует схеме. Верни корректный JSON."})

    result = ExtractionResult(chunk.chunk_id, chunk.doc_id)
    codes = set(find_codes(chunk.text))
    names = vocabulary()
    entities: dict[str, Entity] = {}

    def grounded(kind: str, raw: str) -> Entity | None:
        name = canonical_name(kind, raw, names)
        if name is None or (kind in CODE_TYPES and name not in codes):
            return None
        if kind not in CODE_TYPES and name.casefold() not in chunk.text.casefold():
            return None
        return Entity(kind, name)

    for item in data["entities"]:
        entity = grounded(item["type"], item["name"])
        if entity is None:
            result.dropped += 1
        else:
            entities[entity.node_id] = entity

    candidates: list[Relation] = []
    for item in data["relations"]:
        subject = grounded(item["subject_type"], item["subject"])
        obj = grounded(item["object_type"], item["object"])
        if subject is None or obj is None or RELATION_SIGNATURES[item["type"]] != (subject.type, obj.type):
            result.dropped += 1
            continue

        entities[subject.node_id] = subject
        entities[obj.node_id] = obj

        candidates.append(Relation(subject, item["type"], obj))

    supported = _explicit_relations(chunk.text, entities, names)
    asserted_text = _asserted_text(chunk.text)

    def mentioned(entity: Entity, text: str) -> bool:
        if entity.type in CODE_TYPES:
            return entity.name in find_codes(text)
        return entity.name.casefold() in text.casefold()

    for relation in candidates:
        if relation not in supported:
            reverse_exists = any(
                item.type == relation.type and item.subject == relation.object and item.object == relation.subject
                for item in supported
            )
            same_sentence = any(
                mentioned(relation.subject, sentence) and mentioned(relation.object, sentence)
                for sentence in re.split(r"[.!?\n]", asserted_text)
            )
            if reverse_exists or not same_sentence:
                result.dropped += 1
                continue
            if not _check_relation(chunk.text, relation, llm):
                result.dropped += 1
                continue

        if relation not in result.relations:
            result.relations.append(relation)

    for relation in supported:
        entities[relation.subject.node_id] = relation.subject
        entities[relation.object.node_id] = relation.object
        if relation not in result.relations:
            result.relations.append(relation)

    result.entities = list(entities.values())
    return result

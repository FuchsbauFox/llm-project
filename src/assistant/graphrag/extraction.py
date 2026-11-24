"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

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
        return Entity(kind, name)

    for item in data["entities"]:
        entity = grounded(item["type"], item["name"])
        if entity is None:
            result.dropped += 1
        else:
            entities[entity.node_id] = entity

    for item in data["relations"]:
        subject = grounded(item["subject_type"], item["subject"])
        obj = grounded(item["object_type"], item["object"])
        if subject is None or obj is None or RELATION_SIGNATURES[item["type"]] != (subject.type, obj.type):
            result.dropped += 1
            continue

        entities[subject.node_id] = subject
        entities[obj.node_id] = obj

        relation = Relation(subject, item["type"], obj)
        if relation not in result.relations:
            result.relations.append(relation)

    result.entities = list(entities.values())
    return result

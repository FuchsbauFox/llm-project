"""Инструменты агента. Требование FR-6."""

import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


_PART = {"type": "string", "description": "Номер детали, например P-1042B"}
_WAREHOUSE = {"type": "string", "enum": WAREHOUSES}
_QUESTION = {"type": "string", "description": "Вопрос для поиска"}
TOOL_SCHEMAS: list[dict] = [
    _schema(
        "search_knowledge",
        "Найти факты о деталях, моделях и ошибках в графе знаний.",
        {"question": _QUESTION},
        ["question"],
    ),
    _schema("summarize_topic", "Обобщить сведения по теме из базы знаний.", {"question": _QUESTION}, ["question"]),
    _schema(
        "check_stock",
        "Проверить остаток детали на складе ERP.",
        {"part_id": _PART, "warehouse": _WAREHOUSE},
        ["part_id", "warehouse"],
    ),
    _schema("get_price", "Узнать цену детали в ERP.", {"part_id": _PART}, ["part_id"]),
    _schema(
        "create_service_order",
        "Создать сервисный заказ после подтверждения пользователя.",
        {
            "part_id": _PART,
            "warehouse": _WAREHOUSE,
            "qty": {"type": "integer", "minimum": 1},
            "model": {"type": "string"},
        },
        ["part_id", "warehouse", "qty", "model"],
    ),
]


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Подключает поиск и ERP; ошибки доступности ERP обрабатывает агент."""

    def search_knowledge(question: str) -> dict:
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize_topic(question: str) -> dict:
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def validate_part(part_id: str) -> str | None:
        if not isinstance(part_id, str) or not re.fullmatch(r"P-\d{4}[A-Z]?", part_id):
            return "Некорректный номер детали"
        return None

    def request_erp(method, **arguments) -> dict:
        try:
            return method(**arguments)
        except ErpNotFound as exc:
            return {"error": f"Данные ERP не найдены: {exc}"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила запрос: {exc}"}

    def check_stock(part_id: str, warehouse: str) -> dict:
        error = validate_part(part_id)
        if not isinstance(warehouse, str) or warehouse not in WAREHOUSES:
            error = "Неизвестный склад"
        if error:
            return {"error": error}
        return request_erp(erp.get_stock, part_id=part_id, warehouse=warehouse)

    def get_price(part_id: str) -> dict:
        error = validate_part(part_id)
        if error:
            return {"error": error}
        return request_erp(erp.get_price, part_id=part_id)

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        error = validate_part(part_id)
        if not isinstance(warehouse, str) or warehouse not in WAREHOUSES:
            error = "Неизвестный склад"
        if type(qty) is not int or qty <= 0:
            error = "Количество должно быть положительным целым числом"
        if not isinstance(model, str) or not model.strip():
            error = "Не указана модель"
        if error:
            return {"error": error}
        return request_erp(erp.create_order, part_id=part_id, warehouse=warehouse, qty=qty, model=model)

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }

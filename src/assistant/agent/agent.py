"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

_SYSTEM = (
    "Ты помощник сервисной службы. Для фактов используй поиск по базе знаний, "
    "для актуальных остатков и цен — ERP. Не выдумывай коды, цены и количество. "
    "При отсутствии данных сообщай об этом. Заказ создаётся только после подтверждения. "
    "Если известна только часть ответа, сообщи её и явно укажи, чего не удалось узнать."
)
_SCHEMA_BY_NAME = {item["function"]["name"]: item["function"]["parameters"] for item in TOOL_SCHEMAS}


def _argument_error(name: str, args: dict) -> str | None:
    schema = _SCHEMA_BY_NAME.get(name)
    if schema is None:
        return "Неизвестный инструмент"
    if not isinstance(args, dict) or "_raw" in args:
        return "Нечитаемые аргументы"
    properties = schema["properties"]
    if set(args) - set(properties) or set(schema["required"]) - set(args):
        return "Неверный набор аргументов"
    for key, value in args.items():
        spec = properties[key]
        if spec["type"] == "string" and (not isinstance(value, str) or not value.strip()):
            return f"Некорректный аргумент {key}"
        if spec["type"] == "integer" and (type(value) is not int or value < spec.get("minimum", 0)):
            return f"Некорректный аргумент {key}"
        if "enum" in spec and value not in spec["enum"]:
            return f"Некорректный аргумент {key}"
        if key == "part_id" and not re.fullmatch(r"P-\d{4}[A-Z]?", value):
            return "Некорректный номер детали"
    return None


def _quantity_named(question: str, qty: int) -> bool:
    """Проверяет количество заказа, а не любое число в вопросе."""
    words = {
        1: ("один", "одна", "одно", "одну"),
        2: ("два", "две"),
        3: ("три",),
        4: ("четыре",),
        5: ("пять",),
        6: ("шесть",),
        7: ("семь",),
        8: ("восемь",),
        9: ("девять",),
        10: ("десять",),
    }
    values = "|".join([str(qty), *words.get(qty, ())])
    number = rf"(?<![\w.,-])(?:{values})(?!\w|[.,]\d)"
    unit = r"(?:шт\.?|штук(?:а|и|у)?|детал(?:ь|и|ей)|единиц(?:а|ы|у)?)(?![а-яё])"
    patterns = (
        (rf"{number}\s*{unit}", False),
        (rf"(?:количеств[оаев]*|кол-во)\s*[:=]?\s*{number}", True),
        (rf"{number}\s+P-\d{{4}}[A-Z]?(?![A-Z0-9])", False),
    )
    for pattern, explicit_quantity in patterns:
        for match in re.finditer(pattern, question, re.I):
            prefix = question[max(0, match.start() - 40) : match.start()].lower()
            if re.search(r"\bне\s*$|\bесли\b[^.!?;]*$|\b(?:остаток|в наличии|на складе|есть)\s*$", prefix):
                continue
            # Остаток на складе сам по себе не является количеством заказа.
            clause = re.split(r"[.!?;\n]", prefix)[-1]
            clause = re.split(r"\b(?:закажи|закажите|заказать|оформи|оформите|оформить)\b", clause)[-1]
            if not explicit_quantity and re.search(r"\b(?:остат\w*|наличи\w*|остал\w*|имеется|на складе)\b", clause):
                continue
            return True
    return False


def _numbers(text: str) -> set[str]:
    """Извлекает числа, не считая цифры в кодах деталей и моделей."""
    text = re.sub(r"\b[A-Za-zА-Яа-яЁё]{1,2}[- ]?\d{2,6}[A-Za-zА-Яа-яЁё]?\b", " ", text)
    values = re.findall(r"(?<!\w)\d+(?:[ \u00a0\u202f]\d{3})*(?:[.,]\d+)?(?!\w)", text)
    return {re.sub(r"[ \u00a0\u202f]", "", value) for value in values}


def _ungrounded(answer: str, question: str, results: list[dict]) -> bool:
    evidence = question + "\n" + json.dumps(results, ensure_ascii=False)
    if re.search(r"(?<![\w-])(?:AR|BZ|GS|SP|P|E)--+\d{2,4}[A-Z]?(?!\w)", answer, re.I):
        return True
    if set(find_codes(answer)) - set(find_codes(evidence)):
        return True
    orders = set(re.findall(r"\bSO-\d+\b", answer, re.IGNORECASE))
    if orders - set(re.findall(r"\bSO-\d+\b", evidence, re.IGNORECASE)):
        return True

    # Номера пунктов списка не являются ценой, остатком или иным фактом.
    answer_without_list_numbers = re.sub(r"(?m)^\s*\d{1,2}[.)]\s+(?=\S)", "", answer)
    if _numbers(answer_without_list_numbers) - _numbers(evidence):
        return True
    number = r"(\d+(?:[ \u00a0\u202f]\d{3})*(?:[.,]\d+)?)"
    price_unit = r"\s*(?:руб(?:лей|ля|ль|\.)?|₽)"
    quantity_unit = r"\s*(?:шт(?:ук[аи]?)?\.?)"
    price_in_question = re.findall(number + price_unit, question, re.I)
    price_in_question += re.findall(r"(?:цена|стоимость)\s*(?:равна|составляет|[:=])?\s*" + number, question, re.I)
    quantity_in_question = re.findall(number + quantity_unit, question, re.I)
    quantity_in_question += re.findall(
        r"(?:остаток|количество)\s*(?:равно|составляет|[:=])?\s*" + number, question, re.I
    )
    prices = {re.sub(r"[ \u00a0\u202f]", "", value) for value in price_in_question}
    prices.update(str(item["price_rub"]) for item in results if "price_rub" in item)
    quantities = {re.sub(r"[ \u00a0\u202f]", "", value) for value in quantity_in_question}
    quantities.update(str(item["qty"]) for item in results if "qty" in item)
    price_claims = re.findall(number + price_unit, answer, re.I)
    quantity_claims = re.findall(number + quantity_unit, answer, re.I)
    return any(re.sub(r"[ \u00a0\u202f]", "", value) not in prices for value in price_claims) or any(
        re.sub(r"[ \u00a0\u202f]", "", value) not in quantities for value in quantity_claims
    )


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью и проверяет вызовы и окончательный ответ."""
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": question}]
    steps: list[ToolStep] = []
    results: list[dict] = []
    sources: list[str] = []
    cache: dict[str, dict] = {}
    erp_unavailable = False
    knowledge_refused = False
    knowledge_answered = False
    successful_order: str | None = None

    def outcome(answer: str, refused: bool = False, **kwargs) -> AgentResult:
        return AgentResult(answer, steps, sources, refused, **kwargs)

    while True:
        if len(steps) >= max_steps:
            known = [str(item.get("answer", "")) for item in results if item.get("answer")]
            known.extend(json.dumps(item, ensure_ascii=False) for item in results if not item.get("answer"))
            detail = " ".join(known)
            if erp_unavailable:
                detail += " Данные ERP получить не удалось."
            return outcome(("Работа прервана: достигнут лимит шагов. " + detail).strip(), not bool(results))

        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())
        if not response.tool_calls:
            answer = response.content.strip()
            if _ungrounded(answer, question, results):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Ответ содержит неподтверждённый код или число. Исправь его, "
                            "используя только вопрос и полученные результаты инструментов. "
                            "Не делай выводов о состоянии ERP без данных из вопроса или инструментов."
                        ),
                    }
                )
                corrected = llm.chat(messages)
                answer = corrected.content.strip()
                if corrected.tool_calls or _ungrounded(answer, question, results):
                    return outcome(
                        "Не удалось подтвердить коды или числа в ответе. В базе знаний не найдено достаточно данных.",
                        True,
                    )
            if not answer:
                answer = "В базе знаний не найдено достаточно данных."
                return outcome(answer, True)
            unavailable_mentioned = re.search(
                r"erp.*(?:недоступ|не удалось)|(?:недоступ|не удалось).*erp", answer, re.I
            )
            if erp_unavailable and not unavailable_mentioned:
                answer += " Данные ERP получить не удалось."
            if successful_order and successful_order not in answer:
                answer += f" Номер созданного заказа: {successful_order}."
            if knowledge_refused:
                if not re.search(r"баз[а-яё]* знани|не найден", answer, re.I):
                    answer += " В базе знаний не найдено достаточно данных."
                # Другие успешные результаты могут составлять частичный ответ.
                partial = knowledge_answered or any("error" not in item and "refused" not in item for item in results)
                return outcome(answer, not partial)
            return outcome(answer, refused=bool(steps) and not results)

        for call in response.tool_calls:
            if len(steps) >= max_steps:
                break
            args = call.arguments
            error = _argument_error(call.name, args)
            if not error and call.name not in tools:
                error = "Неизвестный инструмент"
            if not error and "part_id" in args:
                allowed = set(find_codes(question + "\n" + json.dumps(results, ensure_ascii=False)))
                if args["part_id"] not in allowed:
                    error = "Номер детали не подтверждён; сначала вызовите search_knowledge"
            if not error and call.name == "create_service_order" and not _quantity_named(question, args["qty"]):
                error = "Количество заказа не указано пользователем; уточните количество"
                result = {"error": error}
                steps.append(ToolStep(call.name, args, result=result, error=error))
                return outcome("Укажите, сколько штук нужно заказать.")
            if error:
                result = {"error": error}
            elif call.name == "create_service_order" and not confirm:
                pending = {"tool": call.name, "args": dict(args)}
                steps.append(
                    ToolStep(
                        call.name,
                        args,
                        result={"needs_confirmation": True},
                        error="Ожидается подтверждение пользователя",
                    )
                )
                return outcome(
                    "Подтвердите создание сервисного заказа: " + json.dumps(args, ensure_ascii=False),
                    needs_confirmation=True,
                    pending_action=pending,
                )
            else:
                key = json.dumps([call.name, args], ensure_ascii=False, sort_keys=True)
                if key in cache:
                    result = cache[key]
                else:
                    for attempt in range(3):
                        try:
                            result = tools[call.name](**args)
                            break
                        except ErpUnavailable:
                            if attempt == 2:
                                result = {"error": "Данные ERP получить не удалось: сервис недоступен"}
                                erp_unavailable = True
                        except TypeError as exc:
                            result = {"error": f"Неверные аргументы: {exc}"}
                            break
                    cache[key] = result
                error = result.get("error")
            steps.append(ToolStep(call.name, args if isinstance(args, dict) else {}, result=result, error=error))
            messages.append(tool_message(call, result))
            if not error:
                results.append(result)
                for source in result.get("sources", []):
                    if source not in sources:
                        sources.append(source)
                if call.name in ("search_knowledge", "summarize_topic"):
                    knowledge_refused |= bool(result.get("refused"))
                    knowledge_answered |= not result.get("refused", False)
                if call.name == "create_service_order" and result.get("order_id"):
                    successful_order = str(result["order_id"])

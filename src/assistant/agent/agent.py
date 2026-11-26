"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.graphrag.search import has_unconfirmed_installation_claim
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
        argument = properties[key]
        if argument["type"] == "string" and (not isinstance(value, str) or not value.strip()):
            return f"Некорректный аргумент {key}"
        if argument["type"] == "integer" and (type(value) is not int or value < argument.get("minimum", 0)):
            return f"Некорректный аргумент {key}"
        if "enum" in argument and value not in argument["enum"]:
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


def _without_spaces(value: str) -> str:
    return re.sub(r"[ \u00a0\u202f]", "", value)


def _warehouses_named(text: str) -> set[str]:
    warehouse_patterns = {
        "Москва": r"\bмоскв[а-яё]*\b",
        "Санкт-Петербург": r"\bсанкт[- ]петербург[а-яё]*\b",
        "Казань": r"\bказан[а-яё]*\b",
        "Новосибирск": r"\bновосибирск[а-яё]*\b",
    }
    return {name for name, pattern in warehouse_patterns.items() if re.search(pattern, text, re.I)}


def _order_location_error(question: str, args: dict) -> str | None:
    """Не подставляет модель и склад, которых пользователь не называл."""
    models = {code for code in find_codes(question) if re.fullmatch(r"(?:AR|BZ|GS)-\d+", code)}
    if models != {args["model"]}:
        return "Укажите одну модель оборудования для заказа."
    if _warehouses_named(question) != {args["warehouse"]}:
        return "Укажите один склад для заказа."
    return None


def _order_error(question: str, args: dict) -> str | None:
    parts = [code for code in find_codes(question) if code.startswith("P-")]
    numbers = re.findall(r"(?<!\w)(\d+)\s*(?:шт\.?|штук(?:а|и|у)?)(?![а-яё])", question, re.I)
    quantities = {int(value) for value in numbers}
    quantities.update(value for value in range(1, 11) if _quantity_named(question, value))
    if len(parts) > 1 or len(quantities) > 1:
        return "Укажите одну деталь и её количество для отдельного заказа."
    if not _quantity_named(question, args["qty"]):
        return "Укажите, сколько штук нужно заказать."
    return _order_location_error(question, args)


def _execute_tool(tool: Callable[..., dict], args: dict) -> tuple[dict, bool]:
    """Возвращает результат и признак, что ERP осталась недоступной после повторов."""
    for _ in range(3):
        try:
            return tool(**args), False
        except ErpUnavailable:
            continue
        except TypeError as exc:
            return {"error": f"Неверные аргументы: {exc}"}, False
    return {"error": "Данные ERP получить не удалось: сервис недоступен"}, True


def _mismatched_erp_claim(answer: str, question: str, results: list[dict], number: str) -> bool:
    """Сверяет цену или остаток не только со значением, но и с деталью."""
    fields = {
        "price_rub": (r"\s*(?:руб(?:лей|ля|ль|\.)?|₽)", r"(?:цена|стоимость|стоит)"),
        "qty": (r"\s*(?:шт(?:ук[аи]?)?\.?)", r"(?:остаток|количество)"),
    }

    def claims(text: str, unit: str, label: str) -> set[str]:
        # Код детали не должен превращаться в цену после слова «стоит».
        without_codes = re.sub(r"\bP-\d{4}[A-Z]?\b", "", text, flags=re.I)
        values = re.findall(number + unit, text, re.I)
        values += re.findall(label + r"\s*(?:равна|равно|составляет|[:=])?\s*" + number, without_codes, re.I)
        return {_without_spaces(value) for value in values}

    def supported(parts: set[str], values: set[str], field: str, warehouses: set[str], stock_only: bool) -> bool:
        for part in parts:
            locations = warehouses if field == "qty" and warehouses else {None}
            for warehouse in locations:
                matching = {
                    _without_spaces(str(item[field]))
                    for item in results
                    if item.get("part_id") == part
                    and field in item
                    and (warehouse is None or item.get("warehouse") == warehouse)
                    and not (stock_only and item.get("order_id"))
                }
                if not values <= matching:
                    return False
        return True

    question_parts = {code for code in find_codes(question) if code.startswith("P-")}
    question_warehouses = _warehouses_named(question)
    question_stock = bool(re.search(r"остат|в налич|на складе|доступн", question, re.I))
    for clause in re.split(r"(?<=[.!?;])\s+|\n+", answer):
        parts = {code for code in find_codes(clause) if code.startswith("P-")}
        occurrences = list(re.finditer(r"\bP-\d{4}[A-Z]?\b", clause, re.I))
        warehouses = _warehouses_named(clause)
        for field, (unit, label) in fields.items():
            values = claims(clause, unit, label)
            if not values:
                continue
            stock_only = field == "qty" and bool(re.search(r"остат|в налич|на складе|доступн", clause, re.I))
            if (
                len(parts) == 1
                and parts == question_parts
                and values <= claims(question, unit, label)
                and warehouses <= question_warehouses
                and (not stock_only or question_stock)
            ):
                continue
            if not parts:
                # При нескольких деталях нельзя понять, к какой из них относится число.
                if len(question_parts) > 1:
                    return True
                checked_parts = {
                    item.get("part_id")
                    for item in results
                    if field in item and item.get("part_id") and not (stock_only and item.get("order_id"))
                }
                if stock_only and not checked_parts:
                    return True
                if len(checked_parts) > 1:
                    return True
                if (warehouses or stock_only) and checked_parts and not supported(
                    checked_parts, values, field, warehouses, stock_only
                ):
                    return True
                continue
            # Если каждая деталь имеет своё значение, проверяем их по отдельности.
            segments = []
            for index, match in enumerate(occurrences):
                end = occurrences[index + 1].start() if index + 1 < len(occurrences) else len(clause)
                segments.append(clause[match.start() : end])
            if len(parts) > 1 and len(occurrences) == len(parts) and all(claims(s, unit, label) for s in segments):
                for match, segment in zip(occurrences, segments, strict=True):
                    valid = supported(
                        {match.group().upper()}, claims(segment, unit, label), field, warehouses, stock_only
                    )
                    if not valid:
                        return True
            elif not supported(parts, values, field, warehouses, stock_only):
                return True
    return False


def _ungrounded(answer: str, question: str, results: list[dict]) -> bool:
    evidence = question + "\n" + json.dumps(results, ensure_ascii=False)
    if has_unconfirmed_installation_claim(answer):
        return True
    if re.search(
        r"(?<![\w-])(?:AR|BZ|GS|SP|P|E)(?:--+\d{2,4}|-\d{1,4}-\d{1,4})[A-Z]?(?!\w)",
        answer,
        re.I,
    ):
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
    prices = {_without_spaces(value) for value in price_in_question}
    prices.update(str(item["price_rub"]) for item in results if "price_rub" in item)
    quantities = {_without_spaces(value) for value in quantity_in_question}
    quantities.update(str(item["qty"]) for item in results if "qty" in item)
    price_claims = re.findall(number + price_unit, answer, re.I)
    quantity_claims = re.findall(number + quantity_unit, answer, re.I)
    if _mismatched_erp_claim(answer, question, results, number):
        return True
    for value in price_claims:
        if _without_spaces(value) not in prices:
            return True
    for value in quantity_claims:
        if _without_spaces(value) not in quantities:
            return True
    return False


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
    successful_order_args: dict | None = None

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
            uncertainty = re.fullmatch(
                r"(?:номер детали мне неизвестен|не получилось|не удалось|"
                r"(?:данные )?erp (?:недоступна|получить не удалось)|"
                r"в базе знаний не найдено)[.!]?",
                answer,
                re.I,
            )
            if not results and erp_unavailable:
                return outcome("Данные ERP получить не удалось.", True)
            if not results and any(step.error and "Данные ERP не найдены" in step.error for step in steps):
                return outcome("Данные ERP не найдены.", True)
            if not results and any(step.error and "ERP отклонила запрос" in step.error for step in steps):
                return outcome("ERP отклонила запрос.", True)
            if not results and (not steps or not uncertainty):
                return outcome("В базе знаний не найдено данных.", True)
            if _ungrounded(answer, question, results):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Ответ содержит неподтверждённый факт, код или число. Исправь его, "
                            "используя только вопрос и полученные результаты инструментов. "
                            "Не делай выводов о состоянии ERP без данных из вопроса или инструментов."
                        ),
                    }
                )
                corrected = llm.chat(messages)
                answer = corrected.content.strip()
                if corrected.tool_calls or _ungrounded(answer, question, results):
                    if successful_order:
                        return outcome(
                            f"Заказ {successful_order} создан. Остальные сведения в ответе подтвердить не удалось."
                        )
                    return outcome(
                        "Не удалось подтвердить факты в ответе. В базе знаний не найдено достаточно данных.",
                        True,
                    )
            if not answer:
                if successful_order:
                    return outcome(f"Заказ {successful_order} создан.")
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
                # Другие успешные результаты могут составлять частичный ответ.
                partial = knowledge_answered or any("error" not in item and "refused" not in item for item in results)
                if not partial:
                    refusal = "В базе знаний не найдено."
                    if erp_unavailable:
                        refusal += " Данные ERP получить не удалось."
                    return outcome(refusal, True)
                if not re.search(r"баз[а-яё]* знани|не найден", answer, re.I):
                    answer += " В базе знаний не найдено достаточно данных."
                return outcome(answer)
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
            if not error and call.name == "create_service_order":
                if successful_order and args != successful_order_args:
                    error = "Заказ уже создан; для другого заказа требуется новое подтверждение."
                    steps.append(ToolStep(call.name, args, result={"error": error}, error=error))
                    return outcome(f"Заказ {successful_order} создан. {error}")
                order_error = _order_error(question, args)
                if order_error:
                    steps.append(ToolStep(call.name, args, result={"error": order_error}, error=order_error))
                    return outcome(order_error)
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
                    result, unavailable = _execute_tool(tools[call.name], args)
                    erp_unavailable = erp_unavailable or unavailable
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
                    if result.get("refused"):
                        knowledge_refused = True
                    else:
                        knowledge_answered = True
                if call.name == "create_service_order" and result.get("order_id"):
                    successful_order = str(result["order_id"])
                    successful_order_args = dict(args)

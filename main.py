"""Агент с памятью, инструментами и структурированным выводом.

Эксперт по породам и разведению кошек: факты, контекст пользователя
и заметки по разведению через tools.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


SYSTEM_PROMPT = """Ты умный помощник и обладаешь большими знаниями в разведении кошек.
Подсказывай пользователю факты о породах.

Правила:
- Отвечай только по делу, без шуток, без лишней воды и без разговорных вставок.
- Для любой названной породы всегда вызывай get_cat_facts. Не подменяй это своими знаниями, даже если порода уже была в диалоге.
- get_breeding_notes вызывай только если пользователь спрашивает про разведение, вязку, помёт или генетику.
- get_user_experience вызывай, только если порода не названа и нужно опереться на уровень пользователя.
- В cat_facts клади только формулировки из инструментов или из истории диалога, по одному факту на элемент, без маркеров «-» и без нумерации.
- Если пользователь спрашивает, что было раньше, ответь по памяти диалога коротко и по делу.
"""


@dataclass
class Context:
    user_id: str


@dataclass
class ResponseFormat:
    cat_facts: list[str]


CAT_FACTS: dict[str, list[str]] = {
    "золотая шиншилла": [
        "Золотая шиншилла — окрас британской короткошёрстной, а не отдельная порода.",
        "Шерсть тикированная: каждый волос разделён на зоны цвета, из-за этого мех выглядит «пыльным золотом».",
        "Характер спокойный, контактный, хорошо живёт в квартире.",
        "Глаза обычно зелёные или ореховые, морда округлая, тело плотное.",
    ],
    "сиамская": [
        "Сиамская кошка — ориентальная порода с колорпойнтовым окрасом: тёмные морда, уши, лапы и хвост.",
        "Окрас проявляется на более холодных участках тела, поэтому котята рождаются почти белыми.",
        "Порода очень голосистая, социальная и сильно привязывается к человеку.",
        "Тело стройное, клиновидная голова, глаза голубые миндалевидные.",
    ],
    "мейн-кун": [
        "Мейн-кун — одна из самых крупных домашних пород, взрослый кот может весить 6–9 кг.",
        "Шерсть полудлинная, с подшёрстком, кисточки на ушах и длинный пушистый хвост.",
        "Характер уравновешенный, хорошо уживается с детьми и другими животными.",
        "Породе нужны регулярный груминг и пространство для движения.",
    ],
    "британская": [
        "Британская короткошёрстная — плотное тело, круглая голова, плюшевая шерсть.",
        "Характер независимый, но спокойный, не требует постоянного внимания.",
        "Популярные окрасы: голубой, серебристый, золотая шиншилла, вискас.",
        "Склонна к набору веса, поэтому важен контроль рациона.",
    ],
    "сфинкс": [
        "Канадский сфинкс почти без шерсти, теплообмен идёт через кожу, животному нужно тепло.",
        "Кожу протирают, потому что себум не удерживается шерстью.",
        "Характер очень контактный, «следует за человеком» по квартире.",
        "Уши крупные, тело мускулистое, часто есть лёгкий велюровый налёт.",
    ],
}

BREEDING_NOTES: dict[str, list[str]] = {
    "золотая шиншилла": [
        "В разведении контролируют осветление тикинга и чистоту золотого тона.",
        "Не вяжут животных с выраженным серебристым налётом, если цель — стабильный golden.",
        "Типичный помёт британской линии: 3–5 котят.",
        "Следят за поликистозом почек (PKD) и гипертрофической кардиомиопатией (HCM).",
    ],
    "сиамская": [
        "Первая вязка обычно не раньше 12–14 месяцев, когда кошка физически сформирована.",
        "Помёт часто 4–6 котят, котята рождаются светлыми, пойнты проявляются позже.",
        "Важно тестирование на прогрессирующую атрофию сетчатки и HCM.",
        "Не допускают крайний ориентальный тип, если это ухудшает прикус или дыхание.",
    ],
    "мейн-кун": [
        "К разведению допускают после полного роста: чаще после 1,5–2 лет.",
        "Помёт обычно 3–6 котят, котята крупные, роды требуют наблюдения.",
        "Обязательны тесты HCM, SMA и поликистоза почек в линии.",
        "Следят за тазобедренной дисплазией из-за большого веса породы.",
    ],
    "британская": [
        "Вязку планируют после 12 месяцев, у котов часто позже — после формирования костяка.",
        "Помёт в среднем 3–5 котят.",
        "В линии проверяют PKD и HCM.",
        "Избегают слишком плоского профиля, чтобы не получить проблемы с дыханием и слезотечением.",
    ],
    "сфинкс": [
        "Кожу кошки перед вязкой и родами содержат особенно чистой, чтобы снизить риск раздражений.",
        "Помёт обычно 3–5 котят, котята зябкие, нужен обогрев.",
        "Проверяют HCM: порода в группе риска.",
        "Не используют в разведении животных с залысинами воспалительного характера и хроническими кожными проблемами.",
    ],
}

ALIASES = {
    "золотая шиншилла": "золотая шиншилла",
    "шиншилла": "золотая шиншилла",
    "golden chinchilla": "золотая шиншилла",
    "british golden": "золотая шиншилла",
    "сиамская": "сиамская",
    "сиамские": "сиамская",
    "сиамских": "сиамская",
    "siamese": "сиамская",
    "мейн-кун": "мейн-кун",
    "мейн кун": "мейн-кун",
    "maine coon": "мейн-кун",
    "британская": "британская",
    "британцы": "британская",
    "british": "британская",
    "сфинкс": "сфинкс",
    "sphynx": "сфинкс",
}


def normalize_breed(breed: str) -> str:
    key = " ".join(breed.lower().replace("ё", "е").split())
    if key in ALIASES:
        return ALIASES[key]
    for alias, canonical in ALIASES.items():
        if alias in key or key in alias:
            return canonical
    return key


@tool
def get_cat_facts(breed: str) -> str:
    """Возвращает факты о породе кошек: внешность, характер, содержание."""
    name = normalize_breed(breed)
    facts = CAT_FACTS.get(name)
    if not facts:
        return (
            f"В справочнике нет отдельной карточки для «{breed}». "
            "Уточни породу: золотая шиншилла, сиамская, мейн-кун, британская, сфинкс."
        )
    return "\n".join(facts)


@tool
def get_breeding_notes(breed: str) -> str:
    """Заметки по разведению породы.

    Возвращает возраст допуска к вязке, типичный размер помёта
    и генетические риски, которые проверяют в питомнике.
    """
    name = normalize_breed(breed)
    notes = BREEDING_NOTES.get(name)
    if not notes:
        return (
            f"Отдельных заметок по разведению для «{breed}» нет. "
            "Нужно название породы из справочника."
        )
    return "\n".join(notes)


@tool
def get_user_experience(runtime: ToolRuntime[Context]) -> str:
    """Возвращает уровень опыта пользователя по user_id из контекста."""
    user_id = runtime.context.user_id
    if user_id == "1":
        return (
            "Пользователь — новичок. Ему лучше начинать с спокойных пород "
            "в квартирном содержании: золотая шиншилла или британская короткошёрстная."
        )
    return (
        "Пользователь — опытный заводчик. Можно давать рабочие детали по вязке, "
        "отбору производителей и генетическим тестам."
    )


def build_model():
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    max_tokens = int(os.getenv("MAX_TOKENS", "1500"))
    temperature = float(os.getenv("TEMPERATURE", "0.2"))

    if not api_key:
        raise RuntimeError("В .env не найден OPENAI_API_KEY или OPENAI_KEY")

    return init_chat_model(
        f"openai:{model_name}",
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=60,
    )


def build_agent():
    checkpointer = InMemorySaver(
        serde=JsonPlusSerializer(
            allowed_msgpack_modules=[("__main__", "ResponseFormat")]
        )
    )
    return create_agent(
        model=build_model(),
        system_prompt=SYSTEM_PROMPT,
        tools=[get_user_experience, get_cat_facts, get_breeding_notes],
        context_schema=Context,
        response_format=ToolStrategy(ResponseFormat),
        checkpointer=checkpointer,
    )


def ask(agent, text: str, *, thread_id: str, user_id: str) -> ResponseFormat | None:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},
        context=Context(user_id=user_id),
    )
    return result.get("structured_response")


def print_response(title: str, response: ResponseFormat | None) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    if response is None:
        print("Структурированный ответ не получен.")
        return
    for index, fact in enumerate(response.cat_facts, start=1):
        print(f"{index}. {fact}")


def main() -> None:
    agent = build_agent()

    first = ask(
        agent,
        "Несколько фактов о породе золотая шиншилла",
        thread_id="dialog-1",
        user_id="1",
    )
    print_response("Пользователь 1, запрос 1: золотая шиншилла", first)

    second = ask(
        agent,
        "Какую породу я спрашивал в предыдущем сообщении? Кратко напомни главный факт о ней.",
        thread_id="dialog-1",
        user_id="1",
    )
    print_response("Пользователь 1, запрос 2: проверка памяти диалога", second)

    third = ask(
        agent,
        "Что важно при разведении сиамских кошек: возраст вязки, размер помёта и генетические тесты?",
        thread_id="dialog-2",
        user_id="2",
    )
    print_response("Пользователь 2, новый диалог: сиамские + разведение", third)


if __name__ == "__main__":
    main()

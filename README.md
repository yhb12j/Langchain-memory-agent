# LangChain Memory Agent

Агент на LangChain / LangGraph: эксперт по породам и разведению кошек.

Держит диалог в памяти, вызывает инструменты и возвращает ответ строго списком фактов.

## Что внутри

- модель `gpt-4o-mini` через ProxyAPI
- системный промпт эксперта по разведению кошек
- память диалога через `InMemorySaver` и `thread_id`
- контекст пользователя (`user_id`)
- структурированный вывод `ResponseFormat.cat_facts: list[str]`

### Инструменты

| Tool | Что делает |
| --- | --- |
| `get_cat_facts` | факты о породе: внешность, характер, содержание |
| `get_user_experience` | уровень опыта пользователя из контекста (`user_id`) |
| `get_breeding_notes` | допуск к вязке, размер помёта, генетические риски |

## Запуск

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

В `.env` укажи ключ ProxyAPI. Затем:

```powershell
python main.py
```

Скрипт делает три вызова:

1. пользователь `1` спрашивает про золотую шиншиллу;
2. тот же `thread_id` — агент вспоминает предыдущую породу;
3. пользователь `2` в новом диалоге запрашивает факты и заметки по разведению.

## Структура

```
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

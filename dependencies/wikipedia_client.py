__author__ = "Dong Yihan"

import re
from urllib.parse import quote

import requests


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "multi-agent-fact-checking-local/0.1"


def search_titles(query: str, limit: int = 5):
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
        "utf8": 1,
    }
    data = _get_json(params=params)
    results = data.get("query", {}).get("search", [])
    return [result["title"] for result in results if result.get("title")]


def page_summary(title: str, sentences: int = 2):
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "titles": title,
        "format": "json",
        "redirects": 1,
    }
    data = _get_json(params=params)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract", "")
        if extract:
            return _first_sentences(text=extract, count=sentences)
    return ""


def page_url(title: str):
    return f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"


def _get_json(params: dict):
    try:
        response = requests.get(
            WIKIPEDIA_API_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as err:
        print("Wikipedia request failed:", err)
        return {}
    except ValueError as err:
        print("Wikipedia JSON parsing failed:", err)
        return {}


def _first_sentences(text: str, count: int):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:count])

"""The two pieces of wording every answer shares: how a source is cited and how baht are written."""


def cite(source: dict) -> str:
    page = f" หน้า {source['page']}" if source.get("page") else ""
    return f"{source['doc']} {source['clause']}{page}"


def baht(value: int) -> str:
    return f"{value:,}"

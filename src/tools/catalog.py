import csv
from pathlib import Path

from src.config import settings


def _read_products() -> dict[str, dict]:
    path = settings.data_dir / "products.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["sku"].upper(): row for row in csv.DictReader(handle)}


_PRODUCTS: dict[str, dict] | None = None


def _products() -> dict[str, dict]:
    global _PRODUCTS
    if _PRODUCTS is None:
        path = settings.data_dir / "products.csv"
        if not path.exists():
            _PRODUCTS = {}
        else:
            _PRODUCTS = _read_products()
    return _PRODUCTS


def lookup_product(sku: str) -> dict | None:
    product = _products().get(sku.upper())
    return dict(product) if product else None


def product_skus() -> set[str]:
    return set(_products())


def search_products(text: str) -> list[dict]:
    lowered = text.casefold()
    matches = []
    for product in _products().values():
        title = product["title"].casefold()
        base_title = title.split(",", 1)[0].strip()
        if title in lowered or base_title in lowered:
            matches.append(dict(product))
    return matches

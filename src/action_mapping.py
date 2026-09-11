import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import requests

XIVAPI_ACTION_URL = "https://xivapi.com/action"
DEFAULT_MAPPING_PATH = Path("data/action_id_name.json")


def normalize_action_rows(rows: Iterable[dict]) -> Dict[str, str]:
    """把 XIVAPI Action 记录规范化为字符串 ID 到技能名的映射。"""
    mapping: Dict[str, str] = {}
    for row in rows:
        action_id = row.get("ID")
        name = str(row.get("Name") or "").strip()
        if action_id is not None and name:
            mapping[str(action_id)] = name
    return dict(sorted(mapping.items(), key=lambda item: int(item[0])))


def fetch_action_mapping(
    output_path: str | Path = DEFAULT_MAPPING_PATH,
    page_size: int = 100,
    timeout: int = 60,
) -> Dict[str, str]:
    """分页下载 XIVAPI Action 表并同时保存 JSON/CSV 对照表。"""
    if page_size <= 0:
        raise ValueError("page_size 必须大于 0")

    rows: List[dict] = []
    page = 1
    while True:
        response = requests.get(
            XIVAPI_ACTION_URL,
            params={"page": page, "limit": page_size, "columns": "ID,Name"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("Results", [])
        rows.extend(batch)
        pagination = payload.get("Pagination", {})
        if not batch or not pagination.get("PageNext"):
            break
        page = int(pagination["PageNext"])

    mapping = normalize_action_rows(rows)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = destination.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "name"])
        writer.writerows(mapping.items())
    return mapping


def load_action_mapping(path: str | Path = DEFAULT_MAPPING_PATH) -> Dict[str, str]:
    destination = Path(path)
    if not destination.exists():
        return {}
    payload = json.loads(destination.read_text(encoding="utf-8"))
    return {
        str(action_id): str(name)
        for action_id, name in payload.items()
        if str(name).strip()
    }


if __name__ == "__main__":
    mapping = fetch_action_mapping(page_size=1000)
    print(f"已生成 {len(mapping)} 条英文 Action ID 映射")
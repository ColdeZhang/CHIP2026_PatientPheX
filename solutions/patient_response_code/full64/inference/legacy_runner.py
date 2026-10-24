"""Pinned asset and input helpers only; no runnable model or resume entrypoint."""
import hashlib
import json
from pathlib import Path

REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TEXT_PARAMETERS = 8_953_803_264


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value):
    # Preserve actual TF5 loading-info keys while serializing sets deterministically.
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    raise TypeError("Unsupported evidence value type: " + type(value).__name__)


def read_prompts(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    ids = [row["row_id"] for row in rows]
    if not rows or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Prompts require nonempty, unique string row IDs")
    for row in rows:
        if not row.get("messages") or row["messages"][-1].get("role") != "user":
            raise ValueError("Prompt messages must end in a user message")
        if "answer" in row:
            raise ValueError("Inference prompts must not contain training answers")
    return rows


def verify_model(model_path, source_manifest):
    source = json.loads(Path(source_manifest).read_text())
    if source.get("model") != "Qwen/Qwen3.5-9B" or source.get("hf_revision") != REVISION:
        raise ValueError("Unexpected source model/revision")
    checked = {}
    for item in source["files"]:
        if Path(item["name"]).name != item["name"]:
            raise ValueError("Unexpected nested model filename")
        path = Path(model_path) / item["name"]
        if path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
            raise ValueError("Actual model file differs from pinned source: " + item["name"])
        checked[item["name"]] = {"bytes": item["size"], "sha256": item["sha256"]}
    index = json.loads((Path(model_path) / "model.safetensors.index.json").read_text())
    if not set(index["weight_map"].values()).issubset(checked):
        raise ValueError("Source manifest does not cover all weight shards")
    return checked


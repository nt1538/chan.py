from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class EnvConfig:
    data: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "EnvConfig":
        path = Path(path)
        if path.suffix.lower() == ".json":
            return cls(json.loads(path.read_text(encoding="utf-8")))
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML config requires PyYAML; JSON works without it") from exc
            return cls(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        raise ValueError("Config must be .json, .yaml, or .yml")

    def get(self, dotted_key: str, default: Any = None) -> Any:
        value: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return default
            value = value[part]
        return value

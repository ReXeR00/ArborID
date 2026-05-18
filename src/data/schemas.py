from dataclasses import dataclass


@dataclass
class BarkNetRecord:
    path: str
    species: str
    tree_id: str | None = None
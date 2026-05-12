from __future__ import annotations

from collections.abc import Mapping
from typing import Generic, TypeAlias, TypeVar

DictRow: TypeAlias = Mapping[str, object]

_RowT = TypeVar("_RowT")

class RowFactory(Generic[_RowT]): ...

dict_row: RowFactory[DictRow]

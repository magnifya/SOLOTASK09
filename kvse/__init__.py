"""kvse: a minimal embedded relational storage engine (stdlib only).

Public entry points:

    from kvse import Engine, Transaction, StorageError

    engine = Engine("./kvse_data")
    engine.create_table("items", [{"name": "id", "type": "int", "nullable": False}], "id")
    engine.insert("items", {"id": 1})
"""

from .engine import ConflictError, ConstraintError, Engine, ReadOnlyView, Transaction
from .pager import PAGE_SIZE, Pager, StorageError

__version__ = "0.1.0"

__all__ = ["Engine", "Transaction", "StorageError"]

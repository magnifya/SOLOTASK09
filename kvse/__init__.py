"""kvse: a minimal embedded relational storage engine (stdlib only).

Public entry points:

    from kvse import Engine, Transaction, StorageError, Replica

    engine = Engine("./kvse_data")
    engine.create_table("items", [{"name": "id", "type": "int", "nullable": False}], "id")
    engine.insert("items", {"id": 1})

    replica = Replica.create(engine, "./kvse_replica")
    replica.sync()          # catch up to the newest committed state
    replica.get("items", 1)
"""

from .engine import ConflictError, ConstraintError, Engine, ReadOnlyView, Transaction
from .pager import PAGE_SIZE, Pager, StorageError
from .replica import Replica, ReplicaSession

__version__ = "0.1.0"

__all__ = ["Engine", "Transaction", "StorageError", "Replica", "ReplicaSession"]

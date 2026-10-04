"""Tiny query subset: scan, point get and index range on one column.

The module also owns the two small helpers shared with the engine
(``check_type`` for column types and ``sort_key`` for deterministic row
ordering) so that no import cycle is needed.

Supported predicates are ``[(column, op, value)]`` with ``op`` taken from
``OPS``.  The access path is chosen by the first predicate: when it targets an
indexed column and the operator can be served by a range scan, the index is
used, otherwise the table is scanned.
"""

from __future__ import annotations

from .pager import StorageError

OPS = ("=", "!=", "<", "<=", ">", ">=")
INDEX_OPS = ("=", "<", "<=", ">", ">=")


def check_type(value, type_name):
    """True when ``value`` is a legal value for a column of ``type_name``."""
    if type_name == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "bool":
        return isinstance(value, bool)
    if type_name == "text":
        return isinstance(value, str)
    return False


def sort_key(value):
    """Ordering key that keeps mixed int/str/bool values deterministic."""
    if isinstance(value, bool):
        return (0, int(value), "")
    if isinstance(value, (int, float)):
        return (1, value, "")
    return (2, 0, "" if value is None else str(value))


def compare(op, left, right):
    if op == "=":
        return left == right
    if op == "!=":
        return left != right
    if left is None or right is None:
        return False
    try:
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
    except TypeError:
        raise StorageError("cannot compare %r with %r" % (left, right))
    raise StorageError("unsupported operator %r" % (op,))


def normalize_where(source, table, where):
    """Validate ``where`` and return a list of ``(column, op, value)``."""
    info = source.table_info(table)
    columns = {col["name"]: col for col in info["columns"]}
    if where is None:
        return []
    if not isinstance(where, (list, tuple)):
        raise StorageError("where must be a list of [column, op, value]")
    conds = []
    for item in where:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            column, op, value = item
        elif isinstance(item, dict):
            column = item.get("column", item.get("col"))
            op = item.get("op", "=")
            value = item.get("value")
        else:
            raise StorageError("each where predicate must be [column, op, value]")
        if op not in OPS:
            raise StorageError("unsupported operator %r" % (op,))
        if column not in columns:
            raise StorageError("unknown column %s.%s" % (table, column))
        if value is not None and not check_type(value, columns[column]["type"]):
            raise StorageError(
                "predicate on %s.%s expects %s" % (table, column, columns[column]["type"])
            )
        conds.append((column, op, value))
    return conds


def _choose(info, conds, index_hint):
    indexes = info["indexes"]
    if index_hint and index_hint != "scan":
        column = index_hint if index_hint in indexes else None
        if column is None:
            for name, index_name in indexes.items():
                if index_name == index_hint:
                    column = name
                    break
        if column is None:
            raise StorageError("no usable index named %s on table %s" % (index_hint, info["name"]))
        for cond in conds:
            if cond[0] == column and cond[1] in INDEX_OPS:
                return "index", indexes[column]
        return "scan", None
    if conds and conds[0][1] in INDEX_OPS and conds[0][0] in indexes:
        return "index", indexes[conds[0][0]]
    return "scan", None


def plan(source, table, columns=None, where=None, index_hint=None, limit=None, order_by=None):
    conds = normalize_where(source, table, where)
    info = source.table_info(table)
    access, index_name = _choose(info, conds, index_hint)
    return {
        "table": table,
        "access": access,
        "index": index_name,
        "columns": list(columns) if columns else None,
        "where": [list(c) for c in conds],
        "limit": limit,
        "order_by": order_by,
    }


def explain(source, table, where=None, index_hint=None, **kwargs):
    chosen = plan(source, table, where=where, index_hint=index_hint, **kwargs)
    return {"access": chosen["access"], "index": chosen["index"]}


def execute(source, table, columns=None, where=None, index_hint=None, limit=None, order_by=None):
    conds = normalize_where(source, table, where)
    info = source.table_info(table)
    access, index_name = _choose(info, conds, index_hint)
    if access == "index":
        column, op, value = conds[0]
        if op == "=":
            rows = source.index_get(table, column, value)
        elif op in ("<", "<="):
            rows = source.index_range(table, column, None, value)
        else:
            rows = source.index_range(table, column, value, None)
        key_column = order_by or column
    else:
        rows = source.scan(table)
        key_column = order_by or info["primary_key"]
    if conds:
        rows = [row for row in rows if all(compare(op, row.get(col), val) for col, op, val in conds)]
    if order_by and order_by not in {col["name"] for col in info["columns"]}:
        raise StorageError("unknown column %s.%s" % (table, order_by))
    rows = sorted(rows, key=lambda row: sort_key(row.get(key_column)))
    if limit is not None:
        limit = int(limit)
        if limit < 0:
            raise StorageError("limit must be greater than or equal to 0")
        rows = rows[:limit]
    if columns:
        known = {col["name"] for col in info["columns"]}
        unknown = sorted(col for col in columns if col not in known)
        if unknown:
            raise StorageError("unknown column(s) %s on table %s" % (", ".join(unknown), table))
        rows = [{col: row.get(col) for col in columns} for row in rows]
    return {"rows": rows, "access": access, "index": index_name, "table": table}


def select(source, table, columns=None, where=None, index_hint=None, limit=None, order_by=None):
    """Convenience wrapper returning just the projected rows."""
    return execute(
        source,
        table,
        columns=columns,
        where=where,
        index_hint=index_hint,
        limit=limit,
        order_by=order_by,
    )["rows"]

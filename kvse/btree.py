"""B+ tree indexes: an in-memory tree plus its recoverable paged form.

The engine keeps one :class:`BPlusTree` per indexed column in memory and
serializes the committed entries of every index into dedicated index pages
inside the same WAL commit that carries the table pages, so a crash never
exposes half an index update.

In-memory tree
--------------
Entries are ``(value, pk)`` pairs ordered by :func:`entry_key`, which applies
the engine wide ``sort_key`` ordering first to the indexed value and then to
the primary key, so scans come out ordered by key and then by primary key.
Null values are never inserted.  Inserts split full nodes; entries left behind
by deleted or superseded row versions are filtered by the reader and compacted
away by the next bulk rebuild, so a sparse tree never loses connectivity.

Paged form
----------
Every node occupies exactly one page of the page file, serialized as one JSON
document ``{"format": "kvse-btree-1", "node": "leaf"|"internal", "table",
"column", ...}``:

* leaf: ``"entries": [[value, pk], ...]`` and ``"next"`` (page id of the
  right sibling or ``null``);
* internal: ``"keys": [[value, pk], ...]`` and ``"children": [page_id, ...]``
  with one more child than key; key ``i`` is the smallest entry of child
  ``i + 1``.

Leaves are laid out left to right starting at the index's first page,
followed by the internal levels bottom up; the root is the last page.  The
state blob of a commit names every index's root page in its ``index_refs``,
which is what :func:`read_entries` walks and validates: page ids, node
formats, root to leaf connectivity, the leaf chain, key ordering and
separator consistency.
"""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right

from .pager import StorageError
from .query import sort_key

BTREE_FORMAT = "kvse-btree-1"
DEFAULT_ORDER = 64
# Placeholder page id used while measuring a node whose sibling links are not
# assigned yet; wide enough that the real (smaller) ids always fit.
_MEASURE_PAGE_ID = 2 ** 53


def entry_key(entry):
    """Total ordering key of an index entry ``(value, pk)``."""
    return (sort_key(entry[0]), sort_key(entry[1]))


class _Leaf:
    __slots__ = ("entries", "next")

    def __init__(self):
        self.entries = []
        self.next = None


class _Internal:
    __slots__ = ("keys", "children")

    def __init__(self):
        self.keys = []
        self.children = []


class BPlusTree:
    """An in-memory B+ tree over ``(value, pk)`` entries.

    Inserting an entry whose key is already present is a no-op, mirroring the
    engine's rule that a row version contributes an entry only once.  There is
    no delete: superseded entries are dropped by :meth:`bulk_load` rebuilds.
    """

    def __init__(self, order=DEFAULT_ORDER):
        self.order = max(4, int(order))
        self.root = _Leaf()
        self._size = 0

    def __len__(self):
        return self._size

    # ------------------------------------------------------------- descent
    def _descend(self, key):
        """The leaf that would hold an entry with the combined ``key``."""
        node = self.root
        while isinstance(node, _Internal):
            keys = [entry_key(sep) for sep in node.keys]
            node = node.children[bisect_right(keys, key)]
        return node

    def _leftmost(self):
        node = self.root
        while isinstance(node, _Internal):
            node = node.children[0]
        return node

    # -------------------------------------------------------------- lookup
    def __contains__(self, entry):
        key = entry_key(entry)
        leaf = self._descend(key)
        keys = [entry_key(e) for e in leaf.entries]
        pos = bisect_left(keys, key)
        return pos < len(leaf.entries) and keys[pos] == key

    def __iter__(self):
        node = self._leftmost()
        while node is not None:
            yield from node.entries
            node = node.next

    def iter_range(self, low=None, high=None):
        """Yield entries whose value falls inside ``[low, high]``, in order.

        ``None`` bounds are open; an empty range yields nothing.
        """
        lo = None if low is None else (sort_key(low), ())
        hi = None if high is None else sort_key(high)
        if lo is None:
            node = self._leftmost()
            index = 0
        else:
            node = self._descend(lo)
            keys = [entry_key(e) for e in node.entries]
            index = bisect_left(keys, lo)
        while node is not None:
            entries = node.entries
            while index < len(entries):
                entry = entries[index]
                if hi is not None and sort_key(entry[0]) > hi:
                    return
                yield entry
                index += 1
            node = node.next
            index = 0

    # -------------------------------------------------------------- mutation
    def insert(self, entry):
        """Insert ``entry`` unless an equal key exists; split full nodes."""
        key = entry_key(entry)
        path = []
        node = self.root
        while isinstance(node, _Internal):
            keys = [entry_key(sep) for sep in node.keys]
            index = bisect_right(keys, key)
            path.append((node, index))
            node = node.children[index]
        keys = [entry_key(e) for e in node.entries]
        pos = bisect_left(keys, key)
        if pos < len(node.entries) and keys[pos] == key:
            return False
        node.entries.insert(pos, entry)
        self._size += 1
        if len(node.entries) > self.order:
            self._split_leaf(path, node)
        return True

    def _split_leaf(self, path, leaf):
        mid = len(leaf.entries) // 2
        right = _Leaf()
        right.entries = leaf.entries[mid:]
        leaf.entries = leaf.entries[:mid]
        right.next = leaf.next
        leaf.next = right
        self._insert_in_parent(path, leaf, right.entries[0], right)

    def _insert_in_parent(self, path, left, key, right):
        if not path:
            root = _Internal()
            root.keys = [key]
            root.children = [left, right]
            self.root = root
            return
        parent, index = path.pop()
        parent.keys.insert(index, key)
        parent.children.insert(index + 1, right)
        if len(parent.keys) > self.order:
            mid = len(parent.keys) // 2
            up = parent.keys[mid]
            sibling = _Internal()
            sibling.keys = parent.keys[mid + 1:]
            sibling.children = parent.children[mid + 1:]
            parent.keys = parent.keys[:mid]
            parent.children = parent.children[:mid + 1]
            self._insert_in_parent(path, parent, up, sibling)

    def bulk_load(self, entries):
        """Replace the tree with a compactly built one over ``entries``."""
        ordered = []
        for entry in sorted(entries, key=entry_key):
            if ordered and entry_key(ordered[-1]) == entry_key(entry):
                continue
            ordered.append(entry)
        if not ordered:
            self.root = _Leaf()
            self._size = 0
            return
        leaves = []
        for start in range(0, len(ordered), self.order):
            leaf = _Leaf()
            leaf.entries = ordered[start:start + self.order]
            if leaf.entries:
                if leaves:
                    leaves[-1].next = leaf
                leaves.append(leaf)
        level = leaves
        firsts = [leaf.entries[0] for leaf in leaves]
        while len(level) > 1:
            parents = []
            parent_firsts = []
            width = self.order + 1
            for start in range(0, len(level), width):
                group = level[start:start + width]
                node = _Internal()
                node.children = list(group)
                node.keys = [firsts[start + i] for i in range(1, len(group))]
                parents.append(node)
                parent_firsts.append(firsts[start])
            level, firsts = parents, parent_firsts
        self.root = level[0]
        self._size = len(ordered)


# ---------------------------------------------------------------------------
# Paged serialization
# ---------------------------------------------------------------------------

def _encode(node):
    return json.dumps(node, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _leaf_node(table, column, entries, next_page):
    return {
        "format": BTREE_FORMAT,
        "node": "leaf",
        "table": table,
        "column": column,
        "entries": [list(e) for e in entries],
        "next": next_page,
    }


def _internal_node(table, column, keys, children):
    return {
        "format": BTREE_FORMAT,
        "node": "internal",
        "table": table,
        "column": column,
        "keys": [list(k) for k in keys],
        "children": list(children),
    }


def _pack(items, measure, limit):
    """Group ``items`` greedily so every group's serialized form fits."""
    groups = []
    current = []
    for item in items:
        trial = current + [item]
        if current and len(measure(trial)) > limit:
            groups.append(current)
            current = [item]
        else:
            current = trial
    if current or not groups:
        groups.append(current)
    for group in groups:
        if len(measure(group)) > limit:
            raise StorageError("a single index entry does not fit on one page")
    return groups


def _rebalance(groups, measure, limit):
    """Avoid trailing single child internal nodes where the bytes allow it."""
    if len(groups) > 1 and len(groups[-1]) == 1:
        last = groups.pop()
        combined = groups.pop() + last
        if len(measure(combined)) <= limit:
            groups.append(combined)
        else:
            mid = len(combined) // 2
            groups.extend((combined[:mid], combined[mid:]))
    return groups


def build_pages(table, column, entries, first_page, payload_size):
    """Serialize one index into page payloads starting at ``first_page``.

    ``entries`` must be sorted by :func:`entry_key` with no duplicates.
    Returns ``(payloads, root_page, height)`` where ``payloads[i]`` belongs to
    page ``first_page + i`` and the root is always the last payload.
    """
    entries = [list(e) for e in entries]
    groups = _pack(
        entries,
        lambda group: _encode(_leaf_node(table, column, group, _MEASURE_PAGE_ID)),
        payload_size,
    )
    level = []
    for index, group in enumerate(groups):
        next_page = first_page + index + 1 if index + 1 < len(groups) else None
        level.append({
            "payload": _encode(_leaf_node(table, column, group, next_page)),
            "first": group[0] if group else None,
        })
    levels = [level]
    while len(levels[-1]) > 1:
        prev = levels[-1]
        base = first_page + sum(len(lv) for lv in levels[:-1])
        ids = [base + i for i in range(len(prev))]

        def measure(idxs):
            return _encode(_internal_node(
                table, column,
                [prev[j]["first"] for j in idxs[1:]],
                [ids[j] for j in idxs],
            ))

        child_groups = _rebalance(
            _pack(list(range(len(prev))), measure, payload_size), measure, payload_size
        )
        level = []
        for idxs in child_groups:
            level.append({
                "payload": _encode(_internal_node(
                    table, column,
                    [prev[j]["first"] for j in idxs[1:]],
                    [ids[j] for j in idxs],
                )),
                "first": prev[idxs[0]]["first"],
            })
        levels.append(level)
    payloads = [node["payload"] for level in levels for node in level]
    root_page = first_page + len(payloads) - 1
    return payloads, root_page, len(levels)


def read_entries(read_page, root_page, table, column, page_count):
    """Validate the paged B+ tree rooted at ``root_page``; return its entries.

    ``read_page`` is a pager style page reader that raises :class:`StorageError`
    for missing, torn or corrupt pages, which covers page ids outside the file
    and checksum failures.  Every structural rule of the format is checked
    here: node format and ownership, page id reuse, root to leaf connectivity,
    the leaf sibling chain, key ordering inside and across nodes and the
    separator keys.  Any violation raises :class:`StorageError`.
    """
    if not isinstance(root_page, int) or root_page < 0 or root_page >= page_count:
        raise StorageError("index root page %r is out of range" % (root_page,))
    visited = set()
    leaves = []
    entries = []

    def parse(page_id):
        payload = read_page(page_id).rstrip(b"\x00")
        try:
            node = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise StorageError("index page %d is not valid JSON" % page_id)
        if not isinstance(node, dict) or node.get("format") != BTREE_FORMAT:
            raise StorageError("page %d is not a %s node" % (page_id, BTREE_FORMAT))
        if node.get("table") != table or node.get("column") != column:
            raise StorageError(
                "index page %d belongs to %s.%s, not %s.%s"
                % (page_id, node.get("table"), node.get("column"), table, column)
            )
        return node

    def check_entry(entry, page_id):
        if not isinstance(entry, list) or len(entry) != 2:
            raise StorageError("index page %d holds a malformed entry" % page_id)
        return entry_key(entry)

    def walk(page_id, lo, hi):
        """Walk a subtree within bounds [lo, hi); return its smallest key."""
        if page_id in visited:
            raise StorageError("index page %d is referenced more than once" % page_id)
        visited.add(page_id)
        node = parse(page_id)
        kind = node.get("node")
        if kind == "leaf":
            raw = node.get("entries")
            if not isinstance(raw, list):
                raise StorageError("index page %d has malformed entries" % page_id)
            prev = None
            for entry in raw:
                key = check_entry(entry, page_id)
                if prev is not None and key <= prev:
                    raise StorageError("index page %d is not sorted" % page_id)
                prev = key
                if lo is not None and key < lo:
                    raise StorageError("index page %d holds a key below its range" % page_id)
                if hi is not None and key >= hi:
                    raise StorageError("index page %d holds a key above its range" % page_id)
                entries.append(entry)
            leaves.append((page_id, node.get("next")))
            return entry_key(raw[0]) if raw else None
        if kind == "internal":
            keys = node.get("keys")
            children = node.get("children")
            if not isinstance(keys, list) or not isinstance(children, list):
                raise StorageError("index page %d is malformed" % page_id)
            if not children or len(children) != len(keys) + 1:
                raise StorageError(
                    "index page %d has %d keys but %d children"
                    % (page_id, len(keys), len(children))
                )
            separators = [check_entry(key, page_id) for key in keys]
            for left, right in zip(separators, separators[1:]):
                if left >= right:
                    raise StorageError("index page %d separators are not sorted" % page_id)
            for child in children:
                if not isinstance(child, int) or child < 0 or child >= page_count:
                    raise StorageError(
                        "index page %d points at page %r, which is out of range"
                        % (page_id, child)
                    )
            first = None
            for index, child in enumerate(children):
                child_lo = lo if index == 0 else separators[index - 1]
                child_hi = hi if index == len(children) - 1 else separators[index]
                smallest = walk(child, child_lo, child_hi)
                if smallest is None:
                    raise StorageError("index page %d has an empty subtree" % child)
                if index == 0:
                    first = smallest
                elif smallest != separators[index - 1]:
                    raise StorageError(
                        "index page %d separator does not match its right child" % page_id
                    )
            return first
        raise StorageError("index page %d has unknown node type %r" % (page_id, kind))

    walk(root_page, None, None)
    for index, (page_id, next_page) in enumerate(leaves):
        expected = leaves[index + 1][0] if index + 1 < len(leaves) else None
        if next_page != expected:
            raise StorageError("index leaf chain is broken at page %d" % page_id)
    return entries

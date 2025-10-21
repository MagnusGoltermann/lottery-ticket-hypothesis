"""Compatibility package to allow imports like `from lottery_ticket.x import y`.

This shim aliases top-level modules in this repo (e.g., `foundations`,
`datasets`, `mnist_fc`) under the `lottery_ticket` package namespace
without moving files. It works by importing the sibling modules and
registering them in `sys.modules` as submodules of `lottery_ticket`.
"""

import importlib as _importlib
import sys as _sys

_ALIASES = (
    'datasets',
    'foundations',
    'mnist_fc',
)

for _name in _ALIASES:
  try:
    _module = _importlib.import_module(_name)
  except Exception:  # pragma: no cover - best-effort aliasing
    continue
  _sys.modules[__name__ + '.' + _name] = _module
  globals()[_name] = _module

__all__ = list(_ALIASES)



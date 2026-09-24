"""支持 `python -m kubedocs`，等价于 `python -m kubedocs.cli`。"""
from __future__ import annotations

import sys

from kubedocs.cli import main

if __name__ == "__main__":
    sys.exit(main())

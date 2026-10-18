"""所有处理功能。

导入下面这些文件时，里面带 @register 的功能会自动登记到 REGISTRY。
以后新增一组功能，在这里加一行 import 即可。
"""

from . import checks, cleanup, features, format, level, segment  # noqa: F401
from .base import REGISTRY, Param, Step, register

__all__ = ["REGISTRY", "Param", "Step", "register"]

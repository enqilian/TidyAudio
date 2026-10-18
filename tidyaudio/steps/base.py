"""所有处理功能共用的"模板"。

每个功能都是一个 Step 子类，只需要写两样东西：
  1. params：有哪些可调参数（界面会根据它自动生成输入框、下拉框）
  2. apply()：拿到一段音频 Clip，返回处理后的 Clip 列表

apply() 统一返回"列表"，这样三种功能可以用同一种写法：
  - 普通处理（重采样等）：返回 [处理后的 clip]       → 1 进 1 出
  - 过滤（时长过滤等）：  不合格就返回 []            → 1 进 0 出（丢掉）
  - 切片：               返回 [片段1, 片段2, ...]    → 1 进 N 出
"""

from dataclasses import dataclass
from typing import Any

from ..audio_io import Clip


@dataclass
class Param:
    """一个可调参数。"""

    name: str                       # 代码里用的名字，比如 "target_sr"
    label: str                      # 界面上显示的名字，比如 "Target sample rate (Hz)"
    default: Any                    # 默认值
    choices: list | None = None     # 有固定选项时填这里，界面会显示成下拉框
    min: float | None = None        # 数字参数的最小值
    max: float | None = None        # 数字参数的最大值
    help: str = ""                  # 鼠标悬停时的说明
    # 只在另一个参数取某个值时才显示，比如 {"method": "lufs"}：
    # 选了"响度"方式时，界面才显示"目标响度"这个输入框。
    # 值也可以是列表，比如 {"method": ["rms", "lufs"]} 表示二者之一时显示
    show_if: dict | None = None

    def is_active(self, values: dict) -> bool:
        """按 show_if 判断：在当前这组参数值下，这个参数是否生效（界面上是否显示）。"""
        if not self.show_if:
            return True
        for name, expected in self.show_if.items():
            allowed = expected if isinstance(expected, list) else [expected]
            if values.get(name) not in allowed:
                return False
        return True


class Step:
    """处理功能的基类。"""

    key: str = ""           # 唯一的英文名，配置文件里用，比如 "resample"
    label: str = ""         # 界面上显示的名字
    description: str = ""   # 一句话说明这个功能做什么
    params: list[Param] = []
    advanced: bool = False  # True：放在界面的"高级选项"里，默认折叠、不勾选

    def __init__(self, **values):
        # 没传的参数用默认值；传了不认识的参数直接报错，避免拼错名字没发现
        known = {p.name for p in self.params}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"{self.key}: unknown parameters {sorted(unknown)}")
        self.values = {p.name: values.get(p.name, p.default) for p in self.params}

    def apply(self, clip: Clip) -> list[Clip]:
        raise NotImplementedError


# 所有功能的登记表：{"resample": Resample, ...}
REGISTRY: dict[str, type[Step]] = {}


def register(cls: type[Step]) -> type[Step]:
    """装饰器：在类定义上方写 @register，这个功能就会被登记，界面里自动出现。"""
    if cls.key in REGISTRY:
        raise ValueError(f"Duplicate step key: {cls.key}")
    REGISTRY[cls.key] = cls
    return cls

"""
Minimal pydantic v2 shim — ONLY for verifying logic in a sandbox without pydantic.
Not part of the deliverable; delete it. Your real project has pydantic via FastAPI.
"""
from __future__ import annotations
import sys, types, enum, json
from typing import Any, get_type_hints


class FieldInfo:
    def __init__(self, default=None, default_factory=None, **kw):
        self.default = default
        self.default_factory = default_factory
        self.kw = kw


def Field(default=None, *, default_factory=None, **kw):
    return FieldInfo(default, default_factory, **kw)


def _dump(v, mode="python"):
    if isinstance(v, BaseModel):
        return v.model_dump(mode=mode)
    if isinstance(v, enum.Enum):
        return v.value
    if isinstance(v, list):
        return [_dump(x, mode) for x in v]
    if isinstance(v, dict):
        return {k: _dump(x, mode) for k, x in v.items()}
    return v


class _Meta(type):
    def __new__(mcls, name, bases, ns):
        cls = super().__new__(mcls, name, bases, ns)
        fields = {}
        for b in bases:
            fields.update(getattr(b, "_fields", {}))
        for k, v in list(ns.get("__annotations__", {}).items()):
            if k.startswith("_"):
                continue
            fields[k] = ns.get(k, FieldInfo())
        cls._fields = fields
        return cls


class BaseModel(metaclass=_Meta):
    def __init__(self, **data):
        for name, fi in type(self)._fields.items():
            if name in data:
                setattr(self, name, data[name])
            elif isinstance(fi, FieldInfo):
                if fi.default_factory is not None:
                    setattr(self, name, fi.default_factory())
                else:
                    setattr(self, name, fi.default)
            else:
                setattr(self, name, fi)
        for k, v in data.items():
            if k not in type(self)._fields:
                setattr(self, k, v)

    def model_dump(self, mode="python"):
        return {k: _dump(getattr(self, k, None), mode) for k in type(self)._fields}

    def __repr__(self):
        return f"{type(self).__name__}({self.model_dump()})"


mod = types.ModuleType("pydantic")
mod.BaseModel = BaseModel
mod.Field = Field
mod.VERSION = "2.0-shim"
sys.modules["pydantic"] = mod

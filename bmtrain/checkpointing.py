import torch
from typing import Callable, TypeVar
from functools import wraps
from . import debug

class ScopedDebugTensorList:
    def __init__(self) -> None:
        self._hidden_states = []
    
    @property
    def hidden_states(self):
        return self._hidden_states

    def _set_hidden_states(self, hidden_states):
        self._hidden_states = hidden_states

class ScopedTensorInspectorContext:
    def __init__(self):
        pass
    
    def __enter__(self):
        self.prev_hidden = debug.get("_inspect_hidden_states", [])
        debug.set("_inspect_hidden_states", [])
        self._local_list = ScopedDebugTensorList()
        return self._local_list
    
    def __exit__(self, *args):
        self._local_list._set_hidden_states(debug.get("_inspect_hidden_states", []))
        debug.set("_inspect_hidden_states", self.prev_hidden)
        self.prev_hidden = None

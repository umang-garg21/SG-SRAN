# -*-coding:utf-8 -*-
"""
File:        config_utils.py
Created at:  2025/10/19 13:13:56
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: None
"""
from typing import Dict, Any

# ----------------------------------------------------------------------
# Config Namespace Helper
# ----------------------------------------------------------------------


class ConfigNamespace:
    """Dictionary-to-namespace wrapper with dict-like and attribute access."""

    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = ConfigNamespace(v)
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        return {
            k: v.to_dict() if isinstance(v, ConfigNamespace) else v
            for k, v in self.__dict__.items()
        }

    def __repr__(self):
        # Pretty print for debugging
        return f"ConfigNamespace({self.to_dict()})"

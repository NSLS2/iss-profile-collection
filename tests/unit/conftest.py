"""Load pure startup classes without executing hardware initialization."""

import ast
from pathlib import Path
import uuid

import pandas as pd
import pytest


STARTUP = Path(__file__).resolve().parents[2] / "startup"


def load_class(filename, class_name):
    path = STARTUP / filename
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    namespace = {"pd": pd, "uuid": uuid, "ROOT_PATH_SHARED": "/unused"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[class_name]


@pytest.fixture
def sample_class():
    return load_class("93-batch-manager.py", "Sample")


@pytest.fixture
def scan_manager_class():
    return load_class("91-scan_manager.py", "ScanManager")

import json
import os
from unittest.mock import Mock

import pytest


def scan(kind="step scan", **parameters):
    return {"scan_type": kind, "scan_parameters": parameters}


@pytest.fixture
def manager(scan_manager_class, tmp_path):
    # Avoid __init__, which needs the beamline RunEngine and trajectory manager.
    manager = scan_manager_class.__new__(scan_manager_class)
    manager.scan_dict = {"existing": scan(energy=100, filename="old.txt")}
    manager.json_file_path = str(tmp_path / "scans.json")
    (tmp_path / "scans.json").write_text(json.dumps(manager.scan_dict))
    manager.make_scan_uid = Mock(return_value="new")
    manager.create_trajectory_file = Mock()
    scan_manager_class.check_if_brand_new.__globals__.update(json=json, os=os)
    return manager


def test_matching_scan_reuses_uid_and_ignores_filename_value(manager):
    assert manager.check_if_brand_new(scan(filename="other.txt", energy=100)) == "existing"
    manager.create_trajectory_file.assert_not_called()
    manager.make_scan_uid.assert_not_called()


@pytest.mark.parametrize("new_scan", [
    scan("fly scan", energy=100, filename="old.txt"),
    scan(energy=101, filename="old.txt"),
    scan(energy=100),
    scan(energy=100, filename="old.txt", dwell=1),
])
def test_distinct_scan_creates_trajectory_and_persists(manager, new_scan, tmp_path):
    assert manager.check_if_brand_new(new_scan) == "new"
    manager.create_trajectory_file.assert_called_once_with(new_scan, "new")
    saved = json.loads((tmp_path / "scans.json").read_text())
    backup = json.loads((tmp_path / "scans.bak").read_text())
    assert saved["new"] == new_scan
    assert list(backup) == ["existing"]


def test_parameter_order_does_not_affect_matching(manager):
    manager.scan_dict["existing"] = scan(dwell=1, energy=100)
    assert manager.check_if_brand_new(scan(energy=100, dwell=1)) == "existing"

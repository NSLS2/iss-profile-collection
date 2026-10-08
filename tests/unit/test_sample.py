import json

import pandas as pd
import pytest


def coordinates(x=0):
    return dict(x=x, y=1, z=2, th=3)


def test_samples_own_their_position_data(sample_class):
    first = sample_class("first")
    second = sample_class("second")
    first.position_data["extra"] = []
    assert "extra" not in second.position_data


def test_bulk_positions_preserve_order_and_metadata(sample_class):
    sample = sample_class("sample", coordinates=(coordinates(x) for x in range(100)))
    assert sample.number_of_points == 100
    assert sample.number_of_unexposed_points == 100
    assert sample.position_data.index.tolist() == list(range(100))
    assert sample.position_data["x"].tolist() == list(range(100))
    assert sample.position_data["exposure_time"].tolist() == [0] * 100
    assert len(set(sample.uids)) == 100
    assert sample.index_coordinate_dict(5) == coordinates(5)


@pytest.mark.parametrize("invalid", [None, [], [1, 2, 3, 4], {}, dict(x=1, y=2, z=3, bad=4)])
def test_invalid_coordinates_rejected_without_partial_insert(sample_class, invalid):
    sample = sample_class("sample")
    with pytest.raises(ValueError, match="Coordinates"):
        sample.add_new_positions_from_coordinates([coordinates(), invalid])
    assert sample.number_of_points == 0


def test_append_after_deletion_and_sorting(sample_class):
    sample = sample_class("sample", coordinates=[coordinates(x) for x in range(4)])
    sample.remove_positions([1])
    sample.sort_positions_by(["x"])
    sample.add_new_positions_from_coordinates([coordinates(4), coordinates(5)])
    assert sample.position_data.index.tolist() == [0, 2, 3, 4, 5]


def test_round_trip_string_indexes_and_append(sample_class):
    original = sample_class("sample", comment="comment", coordinates=[coordinates()])
    restored = sample_class.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.name == "sample"
    assert restored.comment == "comment"
    assert restored.uids == original.uids
    restored.add_new_position(coordinates(1))
    assert restored.position_data.index.tolist() == [0, 1]
    restored.update_position_coordinates(0, {"x": 7})
    assert restored.number_of_points == 2
    assert restored.index_coordinate_dict(0) == coordinates(7)


def test_updates_with_copy_on_write_and_nonconsecutive_indexes(sample_class):
    with pd.option_context("mode.copy_on_write", True):
        sample = sample_class("sample", coordinates=[coordinates(x) for x in range(3)])
        sample.remove_positions([1])
        sample.update_position_coordinates(2, dict(x=42, th=7))
        sample.set_exposed(1)
        assert sample.index_coordinate_dict(1) == dict(x=42, y=1, z=2, th=7)
        assert sample.index_exposed(1)
        assert not sample.index_exposed(0)
        assert sample.number_of_unexposed_points == 1


def test_empty_append_is_noop(sample_class):
    sample = sample_class("sample", coordinates=[coordinates()])
    before = sample.position_data.copy(deep=True)
    sample.add_positions(iter(()))
    pd.testing.assert_frame_equal(sample.position_data, before)


def test_add_positions_preserves_existing_uids(sample_class):
    source = sample_class("source", coordinates=[coordinates(2), coordinates(3)])
    target = sample_class("target", coordinates=[coordinates()])
    target.add_positions(source.position_data.to_dict("records"))
    assert target.uids[1:] == source.uids

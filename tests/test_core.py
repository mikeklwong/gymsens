import json
from types import SimpleNamespace

import numpy as np
import pytest

from gym_core import (
    Association, Calibration, ITEM_LABELS, ItemObs, ItemRegistry, PersonObs,
    StationConfig, StationRuntime, box_iou, load_calibration,
    point_in_polygon, rect_polygon_overlap_fraction, source_timestamp,
)
from gym_occupancy import atomic_write_json, item_predictions, validate_resolution


def station(sid="rack", rect=(.1, .1, .5, .6), belongings=None):
    def poly(r):
        x1, y1, x2, y2 = r
        return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    return StationConfig(sid, sid, poly(rect), poly(belongings or rect))


def bottle(x=.25, y=.25, size=.025):
    return ItemObs("bottle", (x, y, x+size, y+size), .9)


def make_ready_person(st):
    rt = StationRuntime(st)
    rt.update(0, {42})
    rt.update(2.99, {42})
    assert rt.state == "unknown"
    rt.update(3, {42})
    assert rt.state == "occupied"
    return rt


def test_floor_point_not_body_center():
    st = station(rect=(.1, .7, .5, .95))
    p = PersonObs(101, (.2, .1, .4, .9))
    assert not point_in_polygon(st.interaction, (.3, .5))
    ass, _ = Association([st]).assign_people([p], 0)
    assert ass[st.id] == {101}


def test_person_near_not_inside_is_not_assigned():
    st = station()
    p = PersonObs(101, (.6, .2, .8, .5))
    ass, _ = Association([st]).assign_people([p], 0)
    assert ass[st.id] == set()


def test_two_zones_only_one_claim():
    a = station("a", (.1,.1,.6,.7))
    b = station("b", (.4,.1,.9,.7))
    p = PersonObs(3, (.38,.2,.5,.5))
    ass, _ = Association([a,b]).assign_people([p], 0)
    assert sum(len(v) for v in ass.values()) <= 1


def test_projection_occlusion_marks_unknown():
    st = station("a", (.2,.2,.5,.8))
    p = PersonObs(3, (.21,.15,.49,.4))  # torso over station, feet outside zone
    # For this artificial bbox bottom is in zone; instead use foot to right.
    p = PersonObs(3, (.22,.2,.62,.7))
    a, occluded = Association([st]).assign_people([p], 0)
    # assigned due foot inside; use feet BELOW polygon and body overlap
    p = PersonObs(3, (.22,.3,.48,.95))
    a, occluded = Association([st]).assign_people([p], 1)
    assert a["a"] == set()
    assert occluded["a"]


def test_item_stationary_requires_observations_and_time():
    registry = ItemRegistry(dwell_seconds=5)
    registry.observe([bottle()], 0)
    assert not registry.stable(6)
    registry.observe([bottle()], 2)
    assert registry.stable(5) == ["bottle"]


def test_item_disappearance_expires():
    registry = ItemRegistry(dwell_seconds=2, ttl_seconds=3)
    registry.observe([bottle()], 0)
    registry.observe([bottle()], 1)
    assert registry.stable(2.5)
    assert not registry.stable(4.1)


def test_objects_do_not_occupy_while_merely_passing():
    rt = StationRuntime(station())
    rt.item_registry.observe([bottle()], 0)
    rt.update(0, set())
    rt.update(2, set())
    assert rt.state == "unknown"
    rt.item_registry.observe([bottle()], 3)
    rt.update(5, set())
    assert rt.state == "occupied"
    assert rt.reason == "belongings_present"


def test_person_three_second_entry():
    st = station()
    rt = make_ready_person(st)
    assert rt.reason == "person_present"


def test_person_exit_debounce_and_free_confirmation():
    rt = make_ready_person(station())
    rt.update(5, set())
    assert rt.state == "occupied"
    rt.update(27.9, set())
    assert rt.state == "occupied"
    rt.update(28.1, set())
    assert rt.state == "unknown"
    rt.update(32.2, set())
    assert rt.state == "free"


def test_item_holds_without_person_then_becomes_unknown():
    rt = make_ready_person(station())
    rt.item_registry.observe([bottle()], 2)
    rt.item_registry.observe([bottle()], 8)
    rt.update(8, set(), item_unattended_limit=10)
    assert rt.state == "occupied"
    assert rt.reason == "belongings_present"
    rt.item_registry.observe([bottle()], 12)
    rt.item_registry.observe([bottle()], 17)
    rt.update(18.1, set(), item_unattended_limit=10)
    assert rt.state == "unknown"
    assert rt.reason == "belongings_unattended_long"


def test_person_claim_overrides_unattended_item_timeout():
    rt = StationRuntime(station())
    rt.item_registry.observe([bottle()], 0)
    rt.item_registry.observe([bottle()], 6)
    rt.update(6, set(), item_unattended_limit=10)
    rt.update(15, {1}, item_unattended_limit=10)
    rt.update(18, {1}, item_unattended_limit=10)
    assert rt.state == "occupied"


def test_occluded_station_not_marked_free():
    rt = StationRuntime(station())
    for t in [0, 3, 10]:
        rt.update(t, set(), occluded=True)
    assert rt.state == "unknown"


def test_clear_startup_requires_confirmation():
    rt = StationRuntime(station())
    rt.update(0, set())
    assert rt.state == "unknown"
    rt.update(4, set())
    assert rt.state == "free"


def test_item_overlap_resolves_single_station():
    a = station("a", (.1,.1,.65,.75))
    b = station("b", (.5,.1,.9,.75))
    items, ambiguous = Association([a,b]).assign_items([bottle(.2,.2)])
    assert len(items["a"]) == 1 and not items["b"]


def test_timestamp_prefers_video_time_and_fallback():
    assert source_timestamp(0, 4, 20, None) == pytest.approx(.15)
    assert source_timestamp(2000, 4, 20, None) == pytest.approx(2)
    assert source_timestamp(0, 5, 20, 2) > 2


def test_calibration_checks_aspect_and_ids():
    calibration = Calibration(3840, 2160, [station()])
    calibration.validate(1920,1080)
    with pytest.raises(ValueError):
        calibration.validate(1080,1920)
    with pytest.raises(ValueError):
        Calibration(3840,2160,[station(),station()]).validate(3840,2160)


def test_save_snapshot_atomic(tmp_path):
    path = str(tmp_path / 'sub' / 'status.json')
    atomic_write_json(path, {'stations': {'a':'occupied'}})
    assert json.loads((tmp_path/'sub'/'status.json').read_text())['stations']['a']=='occupied'
    assert not (tmp_path/'sub'/'status.json.tmp').exists()


def test_camera_4k_gate():
    args = SimpleNamespace(min_width=3840, min_height=2160, allow_low_res_prototype=False)
    assert validate_resolution(np.zeros((2160,3840,3), np.uint8), args) == (3840,2160,True)
    with pytest.raises(ValueError):
        validate_resolution(np.zeros((1080,1920,3), np.uint8),args)
    args.allow_low_res_prototype = True
    assert validate_resolution(np.zeros((1080,1920,3), np.uint8),args)[2] is False


def test_standard_model_keys_are_not_coco_class():
    # Models with a custom 'keys' class can still be added separately.
    assert 'key' in ITEM_LABELS
    assert 'bottle' in ITEM_LABELS


def test_polygon_rectangle_overlap():
    poly = [[0,0],[1,0],[1,1],[0,1]]
    assert rect_polygon_overlap_fraction(poly, (.25,.25,.75,.75)) == pytest.approx(.25)
    assert rect_polygon_overlap_fraction(poly, (2,2,3,3)) == 0


def test_recorded_video_end_to_end_with_mocked_predictions(tmp_path, monkeypatch):
    """Integration test of clock, state, and output without downloading YOLO."""
    import cv2
    import gym_occupancy as runner
    path = tmp_path / 'fixture.avi'
    codec = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(str(path), codec, 5, (320, 180))
    if not writer.isOpened():
        pytest.skip('VideoWriter codec unavailable')
    for _ in range(215):
        writer.write(np.zeros((180,320,3), np.uint8))
    writer.release()
    st = station('rack',(.1,.1,.6,.8))
    calibration = Calibration(320,180,[st])
    count = {'n': 0}

    def fake_person_predictions(model, frame, args):
        count['n'] += 1
        return [PersonObs(10, (.15,.1,.35,.7))] if count['n'] <= 25 else []

    monkeypatch.setattr(runner, 'person_predictions', fake_person_predictions)
    monkeypatch.setattr(runner, 'item_predictions', lambda *args, **kwargs: [])
    args = SimpleNamespace(
        status=str(tmp_path/'status.json'), zones='ignored', show=False,
        min_width=3840,min_height=2160, allow_low_res_prototype=True,
        person_conf=.35, person_imgsz=960,item_conf=.35,item_imgsz=960,
        item_interval=1.5, enter_seconds=3, exit_seconds=25,
        free_confirm_seconds=4, item_unattended_limit=300,
        key_detector_configured=False, history_csv=None)
    runner.run_file(str(path),args,calibration,object(),(object(),))
    result = json.loads((tmp_path/'status.json').read_text())
    assert result['mode'] == 'recorded_video'
    assert result['production_resolution_validated'] is False
    assert result['stations']['rack']['state'] == 'free'
    assert result['recording_time_seconds'] >= 40


def test_evaluator_flags_false_free_and_unknown():
    from evaluate import evaluate
    truth = [dict(station_id='a', start_s='0', end_s='20', state='occupied')]
    preds = [dict(station_id='a',time_s='1',predicted_state='occupied'),
             dict(station_id='a',time_s='2',predicted_state='free'),
             dict(station_id='a',time_s='3',predicted_state='unknown')]
    outcome = evaluate(preds,truth)['a']
    assert outcome['false_free_rate_among_occupied'] == pytest.approx(1/3, rel=.01)
    assert outcome['occupied_recall_including_unknown'] == pytest.approx(1/3, rel=.01)
    assert outcome['unknown_fraction'] == pytest.approx(1/3, rel=.01)

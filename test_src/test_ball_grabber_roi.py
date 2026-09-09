from __future__ import annotations

import threading

import numpy as np
import pytest

from src.ball_grabber import (
    ActionTriggerLoop,
    Frame,
    PixelType_Gvsp_BayerRG8,
    frame_bayer_roi_to_numpy,
    frame_to_numpy,
)


def _trigger(fps: float, acquisition_frame_rate: float) -> ActionTriggerLoop:
    return ActionTriggerLoop(
        stop_event=threading.Event(),
        fps=fps,
        acquisition_frame_rate=acquisition_frame_rate,
        device_key=1,
        group_key=1,
        group_mask=1,
        broadcast_address="255.255.255.255",
        timeout_ms=0,
    )


def test_action_trigger_holds_the_grid_after_a_late_wake_up():
    """A late trigger must cost that slot only, never shift the whole schedule.

    The old rule re-based on `now` whenever the fire was late by more than
    period - minimum_period (3.9 ms at 40/48 Hz), which stretched a 25 ms
    request into a 30.9 ms mean and cost 18% of the frames (0905 213942).
    """
    period = 1.0 / 40.0
    roomy = _trigger(40.0, 60.0)      # minimum 16.9 ms, i.e. 8.1 ms of margin
    # a hair late -> still the very next slot on the original grid
    assert roomy._next_slot(10, 10 * period + 0.004) == 11
    # 16 ms late: slot 11 is now closer than the sensor's minimum -> skip it
    assert roomy._next_slot(10, 10 * period + 0.016) == 12
    # a long stall skips whole slots instead of re-basing on `now`
    assert roomy._next_slot(10, 10 * period + 0.100) == 15

    # Same lateness against the shipped 40/48 pairing: the margin is only 3.9 ms,
    # so even a 4 ms late wake-up already costs a slot. That thin margin is why
    # 40 Hz delivered 32 Hz -- raising acquisition_frame_rate is part of the fix,
    # not just the scheduler.
    tight = _trigger(40.0, 48.0)
    assert tight._next_slot(10, 10 * period + 0.004) == 12


def test_action_trigger_slots_are_never_closer_than_the_sensor_minimum():
    trigger = _trigger(60.0, 80.0)
    period = 1.0 / 60.0
    previous, elapsed = 5, 5 * period
    for _ in range(200):
        elapsed += 0.9 * period          # persistent lateness
        slot = trigger._next_slot(previous, elapsed)
        assert slot > previous
        assert (slot - previous) * period >= trigger._minimum_period - 1e-12
        previous = slot
        elapsed = max(elapsed, slot * period)


def test_action_trigger_refuses_a_period_the_sensor_cannot_accept():
    # 60 Hz against today's acquisition_frame_rate=48 is 16.7 ms against a 21.1 ms
    # floor: every trigger would be silently rejected. Fail loudly instead.
    with pytest.raises(ValueError, match="acquisition_frame_rate"):
        _trigger(60.0, 48.0)


def test_bayer_roi_matches_full_decode_away_from_crop_edges():
    width = 24
    height = 24
    raw = np.random.default_rng(7).integers(
        0, 256, size=(height, width), dtype=np.uint8
    )
    frame = Frame(
        data=raw.tobytes(),
        width=width,
        height=height,
        frame_num=1,
        pixel_type=PixelType_Gvsp_BayerRG8,
    )

    x, y, size = 3, 5, 12
    roi = frame_bayer_roi_to_numpy(
        frame,
        x=x,
        y=y,
        width=size,
        height=size,
        resize_to=size,
    )
    expected = frame_to_numpy(frame)[y:y + size, x:x + size]

    np.testing.assert_array_equal(roi[2:-2, 2:-2], expected[2:-2, 2:-2])

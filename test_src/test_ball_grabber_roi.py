from __future__ import annotations

import math
import threading

import numpy as np

from src.ball_grabber import (
    ActionTriggerLoop,
    Frame,
    PixelType_Gvsp_BayerRG8,
    frame_bayer_roi_to_numpy,
    frame_to_numpy,
)


def test_action_trigger_keeps_cadence_without_unsafe_catch_up():
    trigger = ActionTriggerLoop(
        stop_event=threading.Event(),
        fps=29.0,
        acquisition_frame_rate=30.0,
        device_key=1,
        group_key=1,
        group_mask=1,
        broadcast_address="255.255.255.255",
        timeout_ms=0,
    )
    previous_deadline = 100.0

    ordinary_jitter = trigger._next_trigger_time(previous_deadline, 100.0005)
    unsafe_catch_up = trigger._next_trigger_time(previous_deadline, 100.002)

    assert math.isclose(ordinary_jitter, previous_deadline + 1.0 / 29.0)
    assert math.isclose(unsafe_catch_up, 100.002 + 1.0 / 29.0)


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

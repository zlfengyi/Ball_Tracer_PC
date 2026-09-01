from __future__ import annotations

from src import ball_grabber


def test_rotation_defaults_prefer_camera_hardware():
    assert ball_grabber._ENV_CAMERA_REVERSE_180 is True
    assert ball_grabber._ENV_SOFTWARE_ROTATE_180 is False


def test_open_camera_enables_digital_shift_before_setting_value(monkeypatch):
    calls = []

    class FakeCamera:
        def MV_CC_CreateHandle(self, _device_info):
            return 0

        def MV_CC_OpenDevice(self, _access_mode, _switchover_key):
            return 0

        def MV_CC_SetEnumValueByString(self, node, value):
            calls.append(("enum", node, value))
            return 0

        def MV_CC_SetBoolValue(self, node, value):
            calls.append(("bool", node, value))
            return 0

        def MV_CC_SetFloatValue(self, node, value):
            calls.append(("float", node, value))
            return 0

        def MV_CC_StartGrabbing(self):
            return 0

    device_info = ball_grabber.MV_CC_DEVICE_INFO()
    device_info.nTLayerType = ball_grabber.MV_USB_DEVICE
    device_list = ball_grabber.MV_CC_DEVICE_INFO_LIST()
    device_list.nDeviceNum = 1
    device_list.pDeviceInfo[0] = ball_grabber.ctypes.pointer(device_info)
    monkeypatch.setattr(ball_grabber, "MvCamera", FakeCamera)
    monkeypatch.setattr(ball_grabber, "_serial_of", lambda *_args: "camera")

    ball_grabber.open_camera(
        "camera",
        trigger_source=None,
        digital_shift=-2.5,
        reverse_x=False,
        reverse_y=False,
        _st_dev_list=device_list,
    )

    enable_call = ("bool", "DigitalShiftEnable", True)
    value_call = ("float", "DigitalShift", -2.5)
    assert calls.index(enable_call) < calls.index(value_call)


def test_camera_settings_reads_digital_shift_state_and_value():
    class FakeCamera:
        _values = {
            "ExposureTime": 9000.0,
            "Gain": 12.7,
            "DigitalShift": 2.0,
        }

        def MV_CC_GetBoolValue(self, node, value):
            assert node == "DigitalShiftEnable"
            value.value = True
            return 0

        def MV_CC_GetFloatValue(self, node, value):
            value.fCurValue = self._values[node]
            return 0

    capture = object.__new__(ball_grabber.SyncCapture)
    capture._cameras = {"camera": FakeCamera()}

    assert capture.camera_settings() == {
        "camera": {
            "digital_shift_enabled": True,
            "exposure_us": 9000.0,
            "gain_db": 12.7,
            "digital_shift": 2.0,
        }
    }

# Ball Grabber - 海康威视相机精简采集

精简实现：
1. **软触发 30fps**：向主相机发送 TriggerSoftware
2. **ImageGrabber**：连接指定相机，取图入队（最大 10 张，超出丢弃最旧）
3. **get_frame()**：从队列取出一张图并出队

## 依赖

- 海康 MVS SDK（MvCameraControl.dll 需在 PATH 或 MVCAM_COMMON_RUNENV 指定）
- Python 3.8+

## 用法

```python
from ball_grabber import (
    open_camera,
    close_camera,
    SoftTriggerLoop,
    ImageGrabber,
    list_devices,
)

# 1. 打开主相机（Software 触发模式）
cam = open_camera("DA8199285")  # 替换为实际序列号

# 2. 启动软触发 30fps
stop = threading.Event()
trigger = SoftTriggerLoop(cam, fps=30.0, stop_event=stop)
trigger.start()

# 3. 启动 Grabber
grabber = ImageGrabber(cam, timeout_ms=1000)
grabber.start()

# 4. 取图（非阻塞）
frame = grabber.get_frame(timeout_s=0)  # None 表示队列空

# 5. 或阻塞等待
frame = grabber.get_frame(timeout_s=1.0)

# 6. 退出
stop.set()
grabber.stop()
grabber.join(timeout=2)
close_camera(cam)
```

## 运行演示

```bash
cd ball_tracer_pc
python -m ball_grabber.demo [相机序列号]
```

未指定序列号时使用第一个发现的设备。

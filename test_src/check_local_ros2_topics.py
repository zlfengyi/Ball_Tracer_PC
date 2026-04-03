from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import ros2_support

ros2_support.ensure_ros2_environment()

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String


@dataclass(frozen=True)
class TopicCase:
    topic: str
    qos: str


DEFAULT_CASES = (
    TopicCase("/arm_logger/control", "reliable"),
    TopicCase("/predict_hit_pos", "reliable"),
    TopicCase("/pc_car_loc", "best_effort"),
)


def _make_qos(name: str):
    normalized = str(name).strip().lower()
    if normalized == "reliable":
        return ros2_support.make_reliable_qos(depth=20)
    if normalized == "best_effort":
        return ros2_support.make_best_effort_qos(depth=20)
    raise ValueError(f"unsupported qos: {name}")


def _publisher_role(
    *,
    topic: str,
    qos: str,
    token: str,
    warmup_s: float,
    publish_count: int,
    publish_interval_s: float,
) -> int:
    rclpy.init(args=None)
    node_name = f"local_ros2_pub_{uuid.uuid4().hex[:8]}"
    node = Node(node_name)
    publisher = node.create_publisher(String, topic, _make_qos(qos))
    try:
        if warmup_s > 0.0:
            time.sleep(warmup_s)
        for index in range(max(int(publish_count), 1)):
            payload = {
                "token": token,
                "topic": topic,
                "qos": qos,
                "publish_index": index,
                "stamp_pc_ns": time.perf_counter_ns(),
            }
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            publisher.publish(msg)
            print(
                f"[pub] topic={topic} qos={qos} index={index + 1}/{publish_count}",
                flush=True,
            )
            if index + 1 < publish_count:
                time.sleep(max(float(publish_interval_s), 0.0))
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _subscriber_role(
    *,
    topic: str,
    qos: str,
    token: str,
    ready_file: Path,
    recv_file: Path,
    timeout_s: float,
) -> int:
    rclpy.init(args=None)
    node_name = f"local_ros2_sub_{uuid.uuid4().hex[:8]}"
    node = Node(node_name)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    received: dict[str, object] = {"payload": None}

    def _on_msg(message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        if str(payload.get("token")) != token:
            return
        payload["receipt_stamp_pc_ns"] = time.perf_counter_ns()
        recv_file.write_text(
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
            encoding="utf-8",
        )
        received["payload"] = payload
        print(
            f"[sub] received topic={topic} qos={qos} token={token}",
            flush=True,
        )

    node.create_subscription(String, topic, _on_msg, _make_qos(qos))
    ready_file.write_text(
        json.dumps(
            {
                "topic": topic,
                "qos": qos,
                "token": token,
                "stamp_pc_ns": time.perf_counter_ns(),
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    deadline = time.perf_counter() + max(float(timeout_s), 0.1)
    try:
        while time.perf_counter() < deadline:
            if received["payload"] is not None:
                return 0
            executor.spin_once(timeout_sec=0.1)
        print(f"[sub] timeout topic={topic} qos={qos}", flush=True)
        return 2
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _wait_for_file(path: Path, timeout_s: float) -> bool:
    deadline = time.perf_counter() + max(float(timeout_s), 0.1)
    while time.perf_counter() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return path.exists()


def _run_case(case: TopicCase, *, timeout_s: float) -> bool:
    token = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="local_ros2_topics_") as temp_dir:
        temp_path = Path(temp_dir)
        ready_file = temp_path / "subscriber_ready.json"
        recv_file = temp_path / "subscriber_recv.json"

        sub_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--role",
            "sub",
            "--topic",
            case.topic,
            "--qos",
            case.qos,
            "--token",
            token,
            "--ready-file",
            str(ready_file),
            "--recv-file",
            str(recv_file),
            "--timeout-s",
            str(float(timeout_s)),
        ]
        pub_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--role",
            "pub",
            "--topic",
            case.topic,
            "--qos",
            case.qos,
            "--token",
            token,
            "--warmup-s",
            "1.0",
            "--publish-count",
            "25",
            "--publish-interval-s",
            "0.1",
        ]

        sub_proc = subprocess.Popen(sub_cmd, cwd=str(_ROOT))
        try:
            if not _wait_for_file(ready_file, timeout_s=5.0):
                print(f"[check] subscriber not ready: topic={case.topic}", flush=True)
                sub_proc.terminate()
                sub_proc.wait(timeout=5.0)
                return False

            pub_proc = subprocess.run(pub_cmd, cwd=str(_ROOT), check=False)
            sub_exit = sub_proc.wait(timeout=max(float(timeout_s), 1.0) + 2.0)
            ok = pub_proc.returncode == 0 and sub_exit == 0 and recv_file.exists()
            print(
                f"[check] topic={case.topic} qos={case.qos} "
                f"pub_exit={pub_proc.returncode} sub_exit={sub_exit} recv={recv_file.exists()}",
                flush=True,
            )
            if recv_file.exists():
                print(f"[check] payload={recv_file.read_text(encoding='utf-8')}", flush=True)
            return ok
        finally:
            if sub_proc.poll() is None:
                sub_proc.terminate()
                try:
                    sub_proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    sub_proc.kill()


def _check_role(*, timeout_s: float) -> int:
    all_ok = True
    for case in DEFAULT_CASES:
        ok = _run_case(case, timeout_s=timeout_s)
        all_ok = all_ok and ok
    if all_ok:
        print("[check] all local ROS2 topic loopback checks passed", flush=True)
        return 0
    print("[check] local ROS2 topic loopback check failed", flush=True)
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-process local ROS2 topic loopback check for this PC."
    )
    parser.add_argument("--role", choices=("check", "pub", "sub"), default="check")
    parser.add_argument("--topic", type=str, default="")
    parser.add_argument("--qos", choices=("reliable", "best_effort"), default="reliable")
    parser.add_argument("--token", type=str, default="")
    parser.add_argument("--ready-file", type=Path, default=None)
    parser.add_argument("--recv-file", type=Path, default=None)
    parser.add_argument("--timeout-s", type=float, default=8.0)
    parser.add_argument("--warmup-s", type=float, default=1.0)
    parser.add_argument("--publish-count", type=int, default=25)
    parser.add_argument("--publish-interval-s", type=float, default=0.1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.role == "check":
        return _check_role(timeout_s=float(args.timeout_s))

    if not args.topic:
        raise SystemExit("--topic is required for pub/sub roles")
    if not args.token:
        raise SystemExit("--token is required for pub/sub roles")

    if args.role == "pub":
        return _publisher_role(
            topic=args.topic,
            qos=args.qos,
            token=args.token,
            warmup_s=float(args.warmup_s),
            publish_count=int(args.publish_count),
            publish_interval_s=float(args.publish_interval_s),
        )

    if args.ready_file is None or args.recv_file is None:
        raise SystemExit("--ready-file and --recv-file are required for sub role")
    return _subscriber_role(
        topic=args.topic,
        qos=args.qos,
        token=args.token,
        ready_file=args.ready_file,
        recv_file=args.recv_file,
        timeout_s=float(args.timeout_s),
    )


if __name__ == "__main__":
    raise SystemExit(main())

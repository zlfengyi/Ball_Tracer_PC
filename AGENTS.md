# AGENTS.md

## Tracker Launching

- Use `.\run_tracker.ps1` for tracker runs. It is the only supported tracker entry point, and it auto-selects the best available environment while preferring `.venv_ros2` when CUDA/TensorRT is available.
- Use `.\run_tracker.ps1 -PreferredEnv ros2` to force the GPU/TensorRT environment.
- Use `.\run_tracker.ps1 -PreferredEnv clean` only when you explicitly want the CPU-only fallback.
- For quick checks without starting the tracker, run `.\run_tracker.ps1 -ProbeOnly` to see which environment would be selected.

## Performance Notes

- `.venv_ros2` is the high-performance tracker environment. On this machine it is the one that exposes CUDA and TensorRT.
- `.venv_clean` is CPU-oriented and should not be used for performance-sensitive tracker validation unless GPU is unavailable.


## Coding Style ##
原则1）不要过度写兼容代码，比如新增一个接口的时候，为了兼容老的不用的接口，写一堆fall_back。如果不兼容直接让程序报错 log下来就好了！

原则2）精简代码！！尽量按照核心目的去实现路径。 对于除了核心目的以外要增加的可有可无辅助代码请提前问我需不需要！

原则3）如果用户要的是一个非常明确的核心功能，就只实现这条最短可执行路径。
例如：用户要“根据(x, z)查表返回Q/路径”，那就只保留“加载表 -> 查表 -> 返回结果 -> 接到现有执行链路”。
不要顺手增加 CLI、show/debug 子命令、reference capture、离线建表工具、额外包装层、兼容旧接口、花哨的数据结构，除非用户明确要求。

原则4）不要把同一个核心逻辑拆成多个语义重复的方法。
如果本质上只是“给定输入A，算出结果B”，就应该优先收敛成一个主入口。
不要出现多个 wrapper 只是参数名字略有不同，但实际都在做同一件事。

原则5）不要保留已经失真的历史命名和废参数。
如果实现已经不是 IK，就不要继续保留 `ik` 风格的多层包装、旧的 reference 概念、无效 tolerance 参数、已经不用的配置项。
代码里的名字必须反映当前真实行为，而不是历史来源。

原则6）默认删掉非核心辅助路径，而不是默认保留。
对于不影响主执行链路的打印、调试输出、兼容分支、额外 metadata、辅助脚本入口，默认视为可删项。
只有当这些东西直接服务于当前任务目标时，才保留。

原则7）测试也要围绕核心 contract，而不是围绕辅助路径展开。
像查表问题，测试重点应该是：给定一批由正解生成的输入，返回结果是否正确。
不要把大量测试预算浪费在 CLI、打印格式、过渡接口、辅助对象上，除非这些正是当前任务目标。

## Time Rule ##

原则8）所有业务时间统一使用 `perf_counter()` / `perf_counter_ns()`。

- 只要是 tracker 内部状态、ROS2 topic、pc logger、JSON 日志、预测时间、调度/超时判断、测试时间轴等“业务时间”，一律使用 `time.perf_counter()` 或 `time.perf_counter_ns()`。
- 不要再使用 `time.time()`、`datetime.now()`、`time.monotonic()` 承载这些业务时间语义。
- 不要写任何 epoch / wall time / monotonic 到 `perf_counter` 的兼容、fallback、自动识别、互转逻辑。
- 如果旧字段或旧命名已经带有 epoch / wall / monotonic 的历史含义，直接改名或删除，不要保留失真的命名。

请注意由于语法原因，单词间隔可能使用了-， 但在代码里统一使用下划线


1）确认双目相机获取图片、yolo检测使用GPU，延迟ok
2）使用AprilTag板，完成相机内外参校准
单个AprilTag 55.5mm
AprilTag+中间间隔： 16mm
因此相邻两个AprilTag，同一侧边缘相距71.5mm
板子底部到第一个AprilTag下边缘，49mm

3）方格棋盘是9*12， 单个正方形45mm
使用T点的手动标注，完成大地坐标系校准
T点宽2m中间在1m处，长2.54m

3）跑通相机捕捉视频流 -> yolo检测网球：已经完成，并且统计了总延迟在50ms内
	

4）基于网球轨迹和以前curve2逻辑，完成对击球点的预测。 假设击球高度ideal_hit_z。 重新写一个curve3， 大体原理与curve2相当，但是有以下区别：
        - 之前的curve2中的坐标系，y代表高度，z代表深度。 而现在我们用z代表高度、y代表深度方向，这是一个区别
	- 尽量精简代码，仅用于固定相机、捕捉一次网球从远处飞过来的情况
	
	
	- 当满足条件进入曲线追踪后， 每次更新曲线都会生成一个预期击球位置 predict-hit-pos（x,y, z, stage, ct，ht), 其中ct表示给出预测时的当前时刻，ht表示最终击球时刻， x,y,z表示击球位置
	- 关于 predict-hit-pos， 当stage=0是，表示在曲线0阶段（也就是反弹前），按照当前的逻辑：我们去分别拟合曲线0和根据假定的反弹率拟合曲线1，最终得到球在反弹后的下降期，位于ideal_hit_z高度时的击球位置。 当stage=1时，球已经落地反弹了，我们假设车会到达最后一个stage0的predict-hit-pos， 曲线1在这个predict-hit-pos时（其实就是确定了y深度的情况下），对应的击球高度和左右。
	- 注意curve3返回结果的在一个扔球周期内应该是
		- predict-hit-pos （此时大概率stage=0）
		- in_landing (球正在落地和刚反弹几帧里，不拟合，只告知）
		- predict-hit-pos （经过landing后stage=1）
		- done （球已经2次落地、或者当前时刻超过了2次落地的时间），结束一次扔球追踪

	- 记录所有 ball (x,y,z,t), 表示在曝光时刻t球在(x,y,z)位置 和predict-hit-pos
	- 生成横轴是时间的可视化图片。对所有的ball（x，y，z，t）对每个时刻t，分别展示x、y、z 3个点。predict-hit-pos， 按 ct时刻，分别展示x,y,z, ht-ct 4个点, 其中stage有0和1，分别表示在曲线0阶段和曲线1阶段预测生成的击球位置。 记录数据以及可视化数据可以使用单独文件



4.5） 网球定位、轨迹拟合与运行的视频保存：
	-  双目相机在拍摄-》yolo检测完成-》 如果左右图片各有一个网球，则用之前求得的相机内外参，使用三角测距得到网球3d位置
	- 将网球位置送入步骤4中完成的curve3中，获得curve3相应的结果
	- 把左右两张图片拼接起来，并在图片中分别标注出左右图片中检测到的网球以及图片开始曝光的时间（注意是基于PC轴的）， 如果有3d网球，再在图片中打印网球3d位置，和curve3给出的状态，比如是predict-hit-pos 或是 in_landing 或是done（总之基于curve3输出）
	- 在程序运行结束或者手动打断后，将curve3结果数据保存在json中。 可以用一个viewer打开，viewer查看的json的效果和 4中的html一样
	- 请注意json和视频的保存要有个单独的线程，不要拖慢相机视频获取、yolo检测和曲线拟合
	
	主过程文件可以命名为trace_ball_run_

5） 手动分析误差，判断是否可以给底盘进行使用

	合格的结果：
		- predict-hit-pos 变化误差 ，尤其是stage0和1的差别
		- 理想的 z在stage0、1的误差应该在30cm以内，以保证机械臂高度可调整，x在5cm之内
		- 反应时间：理想的反应时间应当有0.2s+， 也就是在击球时刻0.2s之前，predict-hit-pos在x、z上已经收敛


6）车位置定位， car-loc (x,y,z,t) 表示t时刻车的位置
	- 6.1） 整理所有的当前代码，所有源码全部放到src中（不再使用ball_grabber， 但不包括calibration，因为这是一个独立的子项目），config也在src中。 如果是一些临时测试文件，则放到test_src中（与src平级）
	- 6.2） ✅ 已完成（2026-02-28）
		- 新建 src/car_localizer.py：CarLocalizer 类（AprilTag tag36h11 检测 + 双目三角测量）
		  - CarDetection: tag_id, cx, cy, corners
		  - CarLoc: x, y, z, t, tag_id, pixel_1, pixel_2, reprojection_error
		  - detect() / triangulate() / locate() 三个核心方法
		- 新建 test_src/test_car_localizer.py：双目采集 → 检测 → 三角测量 → 标注图片
		- src/__init__.py 已导出 CarDetection, CarLoc, CarLocalizer
		- 测试结果：10/10 帧双目均检测到 tag id=0，3D 定位 std < 0.2mm
		- 注意：当前重投影误差较大(46px)，需要重新校准相机后改善
		- 输出：test_car_output/ 下有标注图片和 tag36h11 打印模板
	-6.3）最后在run_tracker中增加小车位置检测，并在结果视频中再增加打印小车位置


7）使用ROS2 topic，发送给rk3588以下信息：
	predict-hit-pos
	car-loc

8）双目相机升级为3目 @3月1日

	8.1）相机设置：
		使用mvs 客户端检查相机情况是否ok：已人工完成
		使用的3个相机序列尾号，从左到右分别为：243、285、402， 我们分别称之为1、2、3号相机
		ROI设置： 主相机不使用，依然最低ROI
			3个从相机，图片高度都是减掉上面的700像素
			2、3号相机，最右边再截去200像素
		曝光时间、帧率、增益等维持不变
	8.2） ✅ 已完成 — 更新代码从双目到三目，延迟测试结果如下：
		三目延迟测试 (test_trinocular.py, 30帧, batch=3):
		  - GigE 传输（整组）: avg=46ms（243最慢 ~46ms，285/402 ~29ms）
		  - Bayer 解码（3张）: avg=9ms
		  - YOLO 推理（3张，TensorRT batch=3）: avg=35ms
		  - 端到端延迟（曝光→YOLO完成）: avg=91ms (min=83ms, max=107ms)
		  - 分辨率: 243=2448x2048, 285/402=2248x1348
		  - 样本图片保存在 test_trinocular_output/
	8.3）更新相机校准代码，从2目变成3目。 流程和之前都是一样的，只是从2目变成3目而已。 等你全部改完代码后，告诉我 我分别手动启动拍摄AprilTag和地面。 然后我手动标注地面坐标。最后运行校准！
	8.4）✅ 已完成 — ball_localizer.py 升级为多视图 DLT 三角测量（multi_calib.json），支持 2+ 台相机定位。run_tracker.py 同步升级为三目管线。
	8.5） ✅ 已完成 — test_src/test_ball_localizer.py：三目网球 3D 定位测试脚本。


9) 更新3目相机的校准方式
	使用单格4.5cm的方格板， 拍摄2分钟，500张图片，保证每个相机都有远近距离拍到方格板子。
	先独立给每一个相机，基于他拍摄的图片做内参校准。
	然后再使用BA，基于共同拍摄到的图片，优化3台相机的相对外参，此时内参已经不变。

	9.1） 告诉拍摄中用的拍摄系数，曝光时间和增益是多少。然后每个相机拍一张，看看亮度是否合适


10）✅ 使用3目相机对车坐定位，
	实现代码，然后实时输出车的位置。


11）基于ROS2 topic，实现windows 和RK上的通信
	11.1）✅ 先在windows上安装， 发布一个测试的ros2 topic，确保RK收到

	11.2） ✅win和rk做时间同步，具体的RK定期发布一个系统时间，win收到后马上回复。 RK在本地统计得到和 win系统时间的offset，未来rk收到win时间后，自动校准offset，得到rk时间轴上的信息


	11.3）run_tracker 实时发布小车位置
		在运行run_tracker时，我们已经可以获得小车的位置， 通过ros2 topic /pc_car_loc  win->rk， 消息格式为(car_x, car_y, t), 用m和s做单位，其中t表示windows上的时间， rk获取后需要处理时间校准的offset后变成 rk上的时间



	11.4） ✅ 已完成 — run_tracker 发布预测击球信息 /predict_hit_pos  Win->RK
		- 通过 UDP→ROS2 桥接发布到 /predict_hit_pos topic（String JSON）
		- 格式：{"x": m, "y": m, "z": m, "stage": 0|1, "ct": s, "ht": s, "duration": s}
		  其中 duration = ht - ct，表示 duration 秒后到达目标点 (x, y, z)
		- 性能指标（2026-03-09，全画幅 2448x2048，23fps）：
		  - 帧率：稳定 22.9-23.0 fps（小车定位异步化后从 12fps 提升）
		  - 延迟：avg=74ms
		  - 小车定位成功率：99.9%+
		  - COR 参数：cor_z=0.70, cor_xy=0.42（S0/S1 预测 z 差异 <100mm）


12）✅ 加速网球的检测
	- 未修改前做法： 收到相机图片后，统一压缩到1280分辨率，然后同时对3张1280分辨率yolo检测
	- 更新的做法：对每个相机， 实现800*800分片， 比如划分6个区域 
		- 记录上一次检测到网球的图片位置和时间（条件是图片中只有一个球，且检测到了网球3d位置）
		- 如果检测时间距离当前小于0.3s， 则用上一次检测到网球的中心位置，以800切片 ， 
		- 如果上次检测到网球时间超过了0.3s， 启动搜索（就是说这一帧用预先设定的切片1区域， 下一帧用切片2区域） 
		- 保存视频的时候，同时把每个相机当前切片区域给框出来


13） 球拍检测
	13.1）tracker运行中如果要保存视频，改为保存原始3个视频（分辨率/2） + json
	13.2) 专门拼接+label的代码
	13.3）实现单视频里的球拍检测


14）✅ 四相机定位、校准、基础性能测试已完成
	14.1）当前结构
		- 当前四相机硬件为：主相机 `DA7403103`，从相机 `DA8571029 / DA7403087 / DA8474746`。
		- 当前默认采集配置为：全画幅 `2048x1536`、主相机 `35fps`、`3ms` 曝光、`23.5dB` 增益、主代码默认 `180°` 翻转。
		- 触发链路采用主机输出 `ExposureStartActive`，从机使用 `FallingEdge`；该配置对应“曝光开始对齐”。
		- 当前关键配置/结果文件：
			- `src/config/camera.json`：四相机采集配置
			- `src/config/four_camera_intrinsics.json`：固定内参与来源
			- `src/config/four_camera_calib.json`：完整四相机内外参与世界坐标系结果
		- 当前关键脚本：
			- `calibration/four_camera_intrinsics_capture.py`：四相机内参图片采集
			- `calibration/four_camera_intrinsics_calibrate.py`：四相机内参标定
			- `calibration/run_calibration.py`：四相机相对外参标定
			- `calibration/ground_capture.py`：地面注册图片采集
			- `calibration/ground_annotator.py`：人工地面点标注 GUI
			- `calibration/register_ground.py`：注册到大地坐标系
		- 当前主要数据源：
			- `data/four_camera_calibration/retake_mono8_g10_e5000_retry_03181523/`：正式内参数据
			- `data/four_camera_calibration/merge_old_plus_bridge_all_reordered_20260319/`：相对外参合并数据与结果
			- `data/four_camera_ground_registration/ground_reg_20260319_121635/`：大地坐标系注册图片与标注

	14.2）当前流程
		- 内参：使用 `45mm` 方格棋盘（`8x11` 内角点），采集同步图片后运行单目标定，输出固定内参到 `src/config/four_camera_intrinsics.json`。
		- 角点检测与对齐：当前 `8x11` 角点并不是只做“左到右、上到下”的图像排序；还会做一次棋盘原点奇偶检查，专门消除同一块棋盘在不同相机/不同会话里可能出现的 `180°` 角点翻转歧义。
			- 当前实现会在返回网格的第一个角点附近，沿当前网格的行/列方向采样四个象限的亮度，计算一个对角亮度差符号（parity sign）。
			- 系统固定要求棋盘原点满足统一的 canonical parity sign；如果某张图检测出的 parity sign 为反号，就将整组角点按 `corners[::-1]` 反转，使其回到统一的物理原点定义。
			- 完成奇偶对齐后，当前角点编号规则才是稳定的：`8` 列、`11` 行内角点，按“从左到右、从上到下”编号；也就是先列方向递增，再行方向递增，但前提是左上角必须已经被统一到同一个物理棋盘原点。
			- 这套 canonical 角点顺序会同时用于角点缓存生成和相对外参实时检测路径，因此单相机内参、跨相机 pairwise 初始化和最终 BA 使用的是同一套角点原点规则。
		- 相对外参：基于同一棋盘数据，使用“相机重叠图 + pairwise 约束 + BA”的方案恢复 4 台相机相对位姿；不再要求每一帧都必须看到参考相机，只要求整体观测图连通。
		- 世界坐标注册：采集 4 台相机各自的地面图片，在 GUI 中标注已知世界点，再结合固定相对外参把整套相机注册到大地坐标系。
		- 地面注册代码已支持“并非每台相机都能看到地面点”的情况：只要有部分相机完成地面标注，就可以将整套 rig 推到世界坐标；未标到地面的相机会通过固定相对外参一起获得 `R_world / t_world / pos_world`。
		- 当前统一结果文件为 `src/config/four_camera_calib.json`，其中已记录内参、相对外参、世界坐标外参，以及原始数据来源和人工复核信息。

	14.3）当前表现
		- 曝光同步：PC 时间轴原始 exposure spread 约 `0.44ms`，主要来自 PC offset bias；去掉固定 bias 后，在 `FallingEdge` 配置下真实曝光同步误差 `mean=0.007ms`、`p95=0.017ms`，小于 `50us`，可接受。
		- 采集链路：四相机默认支持按相机分别保存视频，输出到 `four_camera_data/`；默认流程已稳定工作。
		- 四相机内参 RMS：
			- `DA7403103`：`0.073px`
			- `DA8571029`：`0.066px`
			- `DA7403087`：`0.094px`
			- `DA8474746`：`0.066px`
		- 四相机相对外参 BA：
			- 总 RMS：`0.115px`
			- 4 台相机已经联通进入 BA
		- 大地坐标系注册：
			- 使用 3 台相机的地面标注完成整套 4 台相机注册
			- 总 ground reprojection RMS：`1.94px`
			- 缺失地面标注的 `DA8571029` 仍已成功获得世界坐标位姿
		- 人工复核：
			- `DA8571029` 高度与现场粗测误差约 `1cm`
			- 其余相机 `x/y` 与现场粗测差异在 `10cm` 内；该数值主要受人工测量误差限制，不代表真实标定误差有 `10cm`

15）✅ 已完成（2026-03-22）机械臂校准与验证
	注意：在根目录下 单独建立一个项目目录， ArmCalibration, 所有代码和数据都在这个项目目录里进行。
	尽量使用ros2已有的节点和topic进行定义，
	- 15.1）RK中已经确定发布了机械臂状态， 用的是  /joint_states sensor_msgs/msg/JointState
			已完成（2026-03-19）：PC 已验证可接收。
			- `/joint_states` 已发现，publisher 数量为 `1`
			- `sensor_msgs/msg/JointState` 接收通过
			- 当前收到 `5` 个关节：`joint_1 ~ joint_5`

	- 15.2）启动捕捉数据， 以指定的时间拍摄4相机分别的图片（刚开始测试10s、10张图片， 正式开始120s，400张图片）， 同时记录拍摄对应图片时的Joint信息。 由于我们没有做 PC <--> RK的时间校准，但是没关系，就P以C收到图片时最新的关节信息就好了（几十ms没对齐，在这个case完全可以接受）。 跑完测试后，把结果保存在对应目录。
			已完成：正式会话为 `ArmCalibration/data/004_15_2_formal_capture_domain2_03192121`
			- 正式采集：`120s / 400` 组
			- 每组保存 4 相机彩色图像 + 最新 `JointState`
			- 运行域：`ROS_DOMAIN_ID=2`
	
	- 15.3）对刚才拍摄的图片， 做网球拍关节点识别。 模型是用yolo_model/racket_pose.onnx, 使用也可以参考一下racket_pose.py, 我们对识的关键点 求出几何中心，作为球拍拍面的中心。 同时把标注结果图片，保存在相应位置。 
		人工检查识别的结果，判断一下阈值，如果可用进入下一阶段。
			已完成：球拍拍面中心只使用关键点 `0-3`
			- 阈值：`0-3` 号点都必须 `>= 40`
			- `4` 号点只作调试，不参与几何中心
			- 合格图片 `1153` 张，已导出人工检查目录 `racket_pose_accepted_flat/`

	- 15.4) 启动POE校准， 自动优化T_BASE_IN_WORLD 和整个机械臂的POE模型， 使得基于T_BASE_IN_WORLD 和POE算出的末端的球拍几何中心位置，在各自相机里的重投影最小。

		首先查看T_BASE_IN_WORLD是否正确（理想情况，高度误差应该在2cm以内）：T_base 可以有很多个
		其次探索能否通过POE模型，得出joint_i  <--> joint_i+1 的距离： 正确
			已完成：按 2D 重投影误差最小做 position-only POE
			- 有效样本：`399`
			- 全局重投影：mean `6.189 px`，median `4.145 px`，p95 `11.764 px`
			- 关节轴距离：`joint_1 -> joint_2 = 8.33 mm`，`joint_2 -> joint_3 = 388.58 mm`，`joint_3 -> joint_4 = 4.28 mm`
		
	
	- 15.5） base重新定义为Joint1轴与大地坐标系地面的交点， 基于此算出新的POE模型。 
			已完成：当前 `base` 原点 = `joint_1` 轴与世界地面 `z=0` 的交点

	- 15.6） 继续接受来自RK的JointState信息， 这里要做的是以1HZ的速度打印基于最新JointState信息和刚计算出来的POE模型， 正解得到末端位置 在大地坐标系下， 相对于base的位移（x，y，z）。  同时也打印出4个相机 识别出的AprilTag 码下的，然后多目计算得到的小车位置p_car。 以及4个相机识别到的球拍几何中点，以及根据此多目三角测距算出的球拍中点在大地坐标系下的位置（p_racket), 以及p_racket - p_car。请注意利用视觉计算车位置和球拍位置（都是基于，几个相机识别到了就用几个相机的原则）。 整个流程相当于启动相机， 然后做ball_location, car_location, 和racket_location。 其中ball_location, car_location 已经有相关代码了，唯一的区别是用3个和4个相机要改一下。 racket_location只要参照类似逻辑来做就能输出了！
		补充约定：视觉先多目三角化得到 AprilTag 中心坐标 `p_apriltag = (x, y, z)`，再换算车底盘中心坐标 `p_car = (x + 0.06, y + 0.10, z - 0.34)`（单位 m）。这里车底盘中心 = 机械臂 base；也就是 `p_car = p_apriltag + (60, 100, -340) mm`。
			已完成：1Hz 实时输出已接通，重点比较
			- `p_racket_rel_base_in_world(by poe)`
			- `p_racket_rel_base_in_world(by vision)`


		人工检查10组，手动测量、视觉测量和POE测距，如果精度OK就过关了！！
		测了4组，总体误差是在几cm的水平，特别是z精度在cm级别！！
	

16) 介绍当前已有的Tracker逻辑和功能
	- 当前 tracker 默认使用四相机配置：
		- 采集配置来自 `src/config/camera.json`
		- 球 / 小车 / 球拍相关多目定位默认使用 `src/config/four_camera_calib.json`
	- 当前主流程：
		- `SyncCapture` 读取 4 台硬触发同步相机图像
		- 每台相机做 `1000x1000` 分片，并压缩到 `640x640` 做 YOLO
		- 分片调度支持 `search / hold / track` 三种模式
		- YOLO 检测后会做重复框过滤、长宽比过滤、静态误检过滤
		- 至少 `2` 台相机各检测到 `1` 个网球时，做多视图 DLT 三角测量
		- 若球 3D 重投影误差 `<= 15px`，则送入 `Curve3Tracker`
	- 当前 `Curve3Tracker` 功能：
		- 状态机：`IDLE -> TRACKING_S0 -> IN_LANDING -> TRACKING_S1 -> DONE`
		- Stage 0：落地前拟合轨迹，预测反弹后到达 `ideal_hit_z` 的击球点
		- Stage 1：落地后拟合轨迹，在小车最终停留的 `y` 位置求球到达时的 `x / z`
		- 支持运动过滤、落地窗口跳过、跳变帧忽略、超时自动重置
	- 当前小车相关功能：
		- 后台线程异步做 AprilTag 多目定位，不阻塞主循环
		- 实时发布 `/pc_car_loc`
		- 实时发布 `/predict_hit_pos`
	- 当前输出结果：
		- 保存半分辨率原始拼接视频
		- 保存完整 JSON（每帧检测、切片、球 3D、预测、小车位置、状态变化）
		- 结束后可自动生成 HTML 和离线标注视频
	- 2026-03-23 最新人工验收结论：
		- 视频记录、球识别、轨迹追踪、车定位，当前所需功能已经具备并跑通
		- 人工检查多个网球位置后，整体 3D 定位误差为 cm 级
		- 固定相机安装后经过约 1 周放置和现场扰动，标定结果仍保持稳定
	- 2026-03-23 当前性能 debug 结论：
		- 四相机同步采集链本身正常，`src/benchmark.py --duration 10` 实测 `35.1 fps`
		- 优化前，tracker 在 `--no-video` 条件下实测约 `22.3 fps`
		- 优化前，tracker 在保存原始拼接视频时实测约 `13.8 fps`
		- 当时的主要瓶颈有两个：
			- YOLO TensorRT 模型是 `tennis_yolo26_v2_20260203_b3_640.engine`，固定 `batch=3`，因此 4 相机每帧需要 `2` 次推理
			- 原始视频保存虽然放在后台线程，但会明显占用 CPU / 内存带宽，拖慢 Bayer 解码和 YOLO 前处理
	- 2026-03-24 已完成的性能优化：
		- 新增并接入默认检测模型 `yolo_model/tennis_yolo26_v2_20260203_b4_640.engine`
		- `BallDetector` 已支持固定 batch 的 TensorRT engine 自动补齐/分批，因此默认 `detect()` / `detect_batch()` 也能安全使用 `b4` engine
		- `frame_to_numpy()` 的 Bayer 快路径改为“先旋转 raw Bayer，再做 demosaic”，输出与旧路径逐像素一致
		- 实测该 Bayer 快路径在 4 相机并行解码 benchmark 中约从 `11.9ms` 降到 `8.6ms`
	- 2026-03-24 优化后实测结果：
		- tracker `--no-video` 短跑实测约 `24.9 fps`
		- tracker 保存原始拼接视频时，`10s` 短跑实测约 `23.1 fps`
		- 当前短跑里视频线程 `qmax=1`，说明录像已不再明显拖住主循环
		- 当前主要剩余开销大致为：`decode ~11.6ms`，`yolo ~26.7ms`，后台写视频 `~35ms`
	- 现阶段判断：
		- 当前 tracker 功能和帧率都已经明显进入可用状态
		- 若后续还要继续逼近 `35fps`，下一个最值得做的是继续压 YOLO 耗时，以及想办法从系统层面去掉 `180°` 旋转需求（例如相机侧翻转或后续做坐标变换替代整帧旋转）


--------------------------------------Windows部分结束-------------------------------------------

21）✅ RK runner上，收到来自windows的(car_loc, t)信息， 使用car-loc和IMU信息，实现卡尔曼滤波，得到一个融合后的速度和位置。 请注意IMU更新频率较快，但是car-loc更新频率是慢的 且有时候不会给，最终要的是car-loc是滞后的。基于这个信息能否完成融合？？重点项目！需要calude给出详细的做法，确认后再进行实施
      

22) ✅ RK runner上 收到(tx, ty, target_t)信息， 开始启动移动！
     - 跑了第一组，试着超目标位置运行了4次，看起来总体是ok的， 移动误差是在5cm水平。

	 
23） 控制机械调整到stage1z-in-stage0-car-loc 高度和位置
	- Newarm 测试第一版本，可行，但是非常的卡。 Next： 希望换成低速但更丝滑一点的版本
	- 
	- ✅ 进行加入挥拍的测试！但如何看球会成为一个严重的问题啊！
		- 优先在机械臂启动的时候扔球查看，期望在挥臂时仍然能看到球！		
		- 从目前摇拍看球的效果评估，基本可以接受！ 目前就是要过滤掉静态的误识别。周末上了更好的ground truth即可
		- 优化球的检测后，阻挡问题完全可以接受了！
	
	- 结论： 之前的Arm参数完全不行， 交由POE建模 + 自己全套控制机械臂



24）室内搭建固定相机棚子，全局追踪击球效果！尽可能准， 0.3s收敛结果。  室内通了以后也可以上楼再搞


25）基于固定相机的击球误差分析，本周目标实现稳定可击打！
	


并行推进的：
	- 新机械臂 by mingshan， 需要验证达秒情况
	- 新底盘、需要采购行走轮与转向轮电机, by minshan?
	- 新的电路控制系统  by lixiang
	- 300w像素 海康相机, 部署固定相机，尽可能的保持准确，0.3s获得反馈不再改变  by  Yi， yuhao： 推进中
	- 机器端感知， by jianbo：failed， 离职，后续by Yi



--------------------------------------测试记录-----------------------------
	- tracker_20260312_172126：
		- 看球都没问题了，遗漏项： 第一次抛球y还是跑过（没有杀住车）
		- 


# Create 0911， 优化curve的代码结构，按照第一曲线、第二曲线、第三曲线分别进行拟合
# 支持同时输出第二曲线上升期、下降期，第三曲线上升期、下降期，在不同高度的接球点


from queue import Queue
import numpy as np
import math
import time
import logging
import os
from .bot_motion_config import BotMotionConfig

np.set_printoptions(suppress=True, precision=3)

class Curve:
    G = 9.8

    # 定义空气阻力系数
    AIR_DRAG_K = 0.02

    # 球拍出球无旋转，瓷砖条件下，落点误差约为在0.1m之内
    BOUNCE_G_K = 0.670 + 0.025 # + 0.072
    BOUNCE_G_B = 0.388  + 0.012  + 0.025# 0.08

    # update 0812, 稍微调高反弹，使得z轴更好跑到位
    BOUNCE_VX_K = 0.55 + 0.09 
    BOUNCE_VX_B = -0.26 # + 0.1 + 0.15 #+0.1 #+ 0.25
    
    # 反弹速度预测为 vx' = vx / (1 + a*vx)
    BOUNCE_VX_A = BotMotionConfig.BOUNCE_VX_A
    
    # 反弹速度预测为 vx' = a * e^(-b*x) * x , a是最大反弹系数，b是衰减率
    # BOUNCE_VX_MAX_RATIO = 0.88
    # BOUNCE_VX_DEDUCTION_RATIO = -0.075

    FIT_Y_WEIGHT = 1.0
    FIT_X_WEIGHT = 1.10

    MAX_LEN_TO_FIT = 10 # 最多只拟合个数，避免积分误差

    # --------------用于判断第一曲线的是否成立（确认是不是网球轨迹）--------------
    Z_SPEED_RANGE = [1.0, 25] # z方向的速度范围

    def __init__(self) -> None:
        self.is_debug = False
        self.reset()
        # 为此类定义记录器
        self.logger = logging.getLogger('curve2')


    def reset(self, is_bot_serve = True):
        self.xs = []
        self.ys = []
        self.zs = []
        self.ts = []
        self.x_ws = []
        self.y_ws = []

        # update 0614: 不再在curve中寻找起点，交由ball_tracer判断球的状态
        
        self.time_base = time.perf_counter()

        # 采用数组方式进行各个曲线的存储
        
        self.move_frames_threshold = [3, 6, 6]
        # self.curve_sufficient = [False, False, False]
        self.curve_samples_cnt = [0, 0, 0] # 每条曲线对应有几帧
        # self.measure_take_off_speed = [None,None,None] # 曲线自足后，反推的曲线起跳速度
        self.ball_start_cnt = []

        self.land_point = [None, None, None]
        self.land_speed = [None, None, None]
        self.x_coeff = [None, None, None]
        self.y_coeff = [None, None, None]
        self.z_coeff = [None, None, None]

        self.x_error_rate = [0, 0, 0]
        self.y_error_rate = [0, 0, 0]
        self.z_error_rate = [0, 0, 0]
        
        self.is_curve_valid = True # 用来帮助外层业务判断当前曲线是否有效，
        
    # 直接取g的加速度，不考虑空气阻力和旋转。返回系数和误差
    # TODO：未来可能需要优化
    def constrained_polyfit(self, x, y, w, fixed_a=-4.9):
        """
        拟合一个二次多项式，并确保二次项系数为固定值 -4.9，
        通过线性拟合求解剩余的b和c系数。
        
        参数：
        x (array-like): 自变量数据
        y (array-like): 因变量数据
        w (array-like): 权重
        
        返回：
        coefficients (np.ndarray): 多项式系数 [a, b, c]
        """
        x = np.array(x)
        y = np.array(y)
        w = np.array(w)
        
        # 调整因变量
        y_adjusted = y - fixed_a * x**2
        
        # 使用np.polyfit进行加权线性拟合，拟合一次多项式 b*x + c
        coeffs = np.polyfit(x, y_adjusted, 1, w=w)
        fit_values = np.polyval(coeffs, x)
        mse = np.mean((fit_values - y_adjusted) ** 2)  # 均方误差

        b, c = coeffs
        
        # 返回最终的多项式系数
        final_coefficients = np.array([fixed_a, b, c])
        
        return final_coefficients, mse
    
    # 返回一次曲线的拟合系数，以及误差 
    def linear_polyfit(self, x, y, w):
        y = np.array(y)
        coeffs = np.polyfit(x, y, 1, w=w)
        fit_values = np.polyval(coeffs, x)
        mse = np.mean((fit_values - y) ** 2)  # 均方误差   

        return coeffs, mse 
    
    # 给定曲线， 曲线开始计算的时间，目标到达时间，加入空气阻力模拟后的落点和速度
    def calc_land_point_and_speed(self, x_coeff, y_coeff, z_coeff, start_t, final_t):
        # 曲线初始的x、z速度和位置
        vx = np.polyval(np.polyder(x_coeff), start_t)
        vz = np.polyval(np.polyder(z_coeff), start_t)
        cx = np.polyval(x_coeff, start_t)
        cz = np.polyval(z_coeff, start_t)
        ct = start_t
        
        # 以50ms forward 模拟, 最后一段时间则有多少算多少
        dt = 0.05
        is_final_dt_calculated = False
        while not is_final_dt_calculated:
            # 更新时间
            if ct + dt > final_t:
                dt = final_t - ct
                is_final_dt_calculated = True
            ct += dt

            # 计算位置
            cx += vx * dt
            cz += vz * dt
            
            # 更新速度
            a = (vx ** 2 + vz ** 2) * Curve.AIR_DRAG_K
            x_ratio = vx / math.sqrt(vx ** 2 + vz ** 2)
            z_ratio = vz / math.sqrt(vx ** 2 + vz ** 2)
            vx -= a * x_ratio * dt
            vz -= a * z_ratio * dt
    
        vy = np.polyval(np.polyder(y_coeff), final_t)
        final_speed = [vx, vy, vz, math.sqrt(vx ** 2 + vz ** 2)]
        final_point = [cx, 0, cz, final_t]

        return final_point, final_speed

    # 保存当前帧的数据和权重   
    def append_loc(self, ball_loc):
        self.xs.append(ball_loc[0])
        self.ys.append(ball_loc[1])
        self.zs.append(ball_loc[2])
        self.ts.append(ball_loc[3] - self.time_base)
        if len(self.y_ws) == 0:
            self.x_ws.append(1.0)
            self.y_ws.append(1.0)
        else:
            self.x_ws.append(self.x_ws[-1] * Curve.FIT_X_WEIGHT)
            self.y_ws.append(self.y_ws[-1] * Curve.FIT_Y_WEIGHT)
    

    # 根据给定曲线的落地点和速度，预测该曲线的反弹后的曲线
    # deprecated
    def predict_next_curve(self, land_point, land_speed):
        t1 = land_point[-1] # 当前曲线的落地时刻

        # speed 格式为 [vx, vy, vz, sqrt(vx**2+vz**2)]
        # new_v = land_speed[-1] * Curve.BOUNCE_VX_K + Curve.BOUNCE_VX_B
        self.logger.info(f"bounce_vx_a:{Curve.BOUNCE_VX_A}")
        new_v = land_speed[-1]/(1+ Curve.BOUNCE_VX_A * land_speed[-1])
        # new_v = Curve.BOUNCE_VX_MAX_RATIO * (math.e ** (Curve.BOUNCE_VX_DEDUCTION_RATIO * land_speed[-1])) * land_speed[-1]
        vx = land_speed[0] / land_speed[-1] * new_v
        vz = land_speed[2] / land_speed[-1] * new_v
        x2_coeff = np.array([vx, -t1 * vx + land_point[0]])
        z2_coeff = np.array([vz, -t1 * vz + land_point[2]])

        vy = -land_speed[1] * Curve.BOUNCE_G_K + Curve.BOUNCE_G_B
        y2_coeff = np.array([-0.5 * Curve.G,  Curve.G * t1 + vy, -0.5 * Curve.G * t1 * t1 - t1 * vy])
        # y2_coeff = np.array([-0.5 * Curve.G, Curve.G * t1 + vy, -0.5 * Curve.G * t1 * t1 - t1 * vy])

        return x2_coeff, y2_coeff, z2_coeff
    
    # 根据BotMotionConfig中指定的反弹系数，进行预测
    def predict_next_curve_v2(self, land_point, land_speed):
        t1 = land_point[-1] # 当前曲线的落地时刻

        # speed 格式为 [vx, vy, vz, sqrt(vx**2+vz**2)]
        # new_v = land_speed[-1] * Curve.BOUNCE_VX_K + Curve.BOUNCE_VX_B
        # self.logger.info(f"bounce_vx_a:{Curve.BOUNCE_VX_A}")
        # new_v = land_speed[-1]/(1+ Curve.BOUNCE_VX_A * land_speed[-1])
        # new_v = Curve.BOUNCE_VX_MAX_RATIO * (math.e ** (Curve.BOUNCE_VX_DEDUCTION_RATIO * land_speed[-1])) * land_speed[-1]
        
        new_v = land_speed[-1] * BotMotionConfig.BOUNCE_XZ_COEFF[0] + BotMotionConfig.BOUNCE_XZ_COEFF[1]
        vx = land_speed[0] / land_speed[-1] * new_v
        vz = land_speed[2] / land_speed[-1] * new_v
        x2_coeff = np.array([vx, -t1 * vx + land_point[0]])
        z2_coeff = np.array([vz, -t1 * vz + land_point[2]])

        # vy = -land_speed[1] * Curve.BOUNCE_G_K + Curve.BOUNCE_G_B
        vy = -(land_speed[1] * BotMotionConfig.BOUNCE_Y_COEFF[0] + BotMotionConfig.BOUNCE_Y_COEFF[1])
        y2_coeff = np.array([-0.5 * Curve.G,  Curve.G * t1 + vy, -0.5 * Curve.G * t1 * t1 - t1 * vy])
        # y2_coeff = np.array([-0.5 * Curve.G, Curve.G * t1 + vy, -0.5 * Curve.G * t1 * t1 - t1 * vy])

        return x2_coeff, y2_coeff, z2_coeff
        

    # calculate the time when the ball is at the target height
    def calc_t_at_height(self, y_coeff, target_y):
        y_coeff = y_coeff.copy()
        y_coeff[2] -= target_y
        if y_coeff[1] ** 2 - 4*y_coeff[0]*y_coeff[2] <= 0:
            return None
        else:
            ts = np.roots(y_coeff)    
        
        return [min(ts), max(ts)]

    # 计算给定的id曲线，在target_y高度接球点，以及对应上升期和下降期的得分
    def add_receive_locs_at_height(self, id, score0, score1):             
           
        low_net_receive_ts = self.calc_t_at_height(self.y_coeff[id], BotMotionConfig.NET_HEIGHT_1)
        if low_net_receive_ts is None:   # 最低接球高度也无法碰到球，返回None
            return None
        
        high_net_receive_ts = self.calc_t_at_height(self.y_coeff[id], BotMotionConfig.NET_HEIGHT_2)
        if high_net_receive_ts is None:
            mid_t = (low_net_receive_ts[0] + low_net_receive_ts[1]) / 2
            high_net_receive_ts = [mid_t, mid_t]


        # 分别计算接球点的上升期和下降期的中点接球时间
        # t_up = (low_net_receive_ts[0] + high_net_receive_ts[0]) / 2
        # t_down = (low_net_receive_ts[1] + high_net_receive_ts[1]) / 2
        # update 0221, 往最高点跑，忽略高球过网的情况来减少打中机器
        t_up = high_net_receive_ts[0]
        t_down = high_net_receive_ts[1]

        # 使得接球点从小车中心移到小车摄像头位置，因此车辆总体需要后退，也就是加多时间）
        t_move_fix = BotMotionConfig.CURVE_STEP_BACK_METER / abs(self.land_speed[id][2])

        # 计算最佳接球点, 轨迹对应角度, z方向落点容忍误差
        # 上升期，score1，该时间点，球到达车子中心距离地面80cm的点，轨迹会打到摄像头，为了不打到摄像头，向后退25cm/ 使用t_move_fix 配置
        p_up = [np.polyval(self.x_coeff[id], t_up+t_move_fix), np.polyval(self.y_coeff[id], t_up), np.polyval(self.z_coeff[id] , t_up+t_move_fix), t_up+self.time_base]
        # 下降期， score2
        p_down = [np.polyval(self.x_coeff[id], t_down), np.polyval(self.y_coeff[id], t_down), np.polyval(self.z_coeff[id], t_down), t_down+self.time_base]

        # 球飞行方向的角度
        line_an = math.atan2(p_up[2] - p_down[2], p_up[0] - p_down[0])
        z_speed = np.polyval(np.polyder(self.z_coeff[id]), t_up)
        x_speed = np.polyval(np.polyder(self.x_coeff[id]), t_up) 

        # 1220室内测试，想接曲线2时，存在球反弹先打到机器的情况。因此在还未曲线1拟合完，优先对曲线2落点靠后处理，避免球打机器。
        if id == 2 and self.curve_samples_cnt[1] < 4:
            p_down[2] -= 0.5
            p_up[2] -= 0.25

        # z方向落点容忍误差，等于时间误差乘以z速度
        z_tolerance = abs((low_net_receive_ts[0] - t_up) * self.land_speed[id][2])
        
        self.loc_results.append({"point":p_up, "score":score0, "curve_id": str(id)+"_up", "angle":line_an, "z_tolerance":z_tolerance, "x_speed":x_speed, "z_speed":z_speed, "fit_samples":self.curve_samples_cnt[0] + self.curve_samples_cnt[1]})
        self.loc_results.append({"point":p_down, "score":score1, "curve_id": str(id)+"_down", "angle":line_an, "z_tolerance":z_tolerance, "x_speed":x_speed, "z_speed":z_speed, "fit_samples":self.curve_samples_cnt[0] + self.curve_samples_cnt[1]})
    
        
    # 如果是用户回球， 默认情况下是用户回球，也即is_bot_fire=-1， 当机器发球时，vz为正
    def add_frame(self, ball_loc, is_bot_fire=-1):
        # 加入第一个点，设置第一曲线起点和时间基准
        if len(self.ts) == 0:
            self.time_base = ball_loc[-1] 
            self.ball_start_cnt.append(0)
        self.append_loc(ball_loc)

        # id表示 当前第几条曲线
        id = len(self.ball_start_cnt) - 1
        self.curve_samples_cnt[id] += 1

        # 如果接近当前曲线落地点，为了防止落地反弹阶段不对的误差，不予计算而退出
        if (self.land_point[id] is not None) and abs(self.land_point[id][-1] - self.ts[-1]) < 0.03:
            # self.logger.info(f"Too close to land point, return None. Time: {self.ts[-1]} id:{id} Predict landing ball {self.land_point[id]}")
            return None

        # 如果超过了当前曲线的落点，则进入下一条曲线阶段
        if (self.land_point[id] is not None) and self.ts[-1] > self.land_point[id][-1]:
            if id == 2:
                return None
            self.ball_start_cnt.append(len(self.ts)-1)
            id += 1
            self.logger.info(f"start to fit curve {id}")   
            

        # 开始对当前曲线进行拟合， 如果采样点不足，直接返回。 
        start_cnt = self.ball_start_cnt[id]
        n_sample = len(self.ts) - self.ball_start_cnt[id]
        self.logger.info(f"start_cnt: {start_cnt}, n_sample: {n_sample}")
        if n_sample < 3 or (id == 0 and n_sample < self.move_frames_threshold[0]):
            # self.logger.info(f"  Not enough sample points, return None. Time: {self.ts[-1]} Predict landing ball {self.land_point[id]}, speed: {self.land_speed[id]}")
            return None
        if n_sample >= 10:          # 拟合最多只需10帧数据，保证曲线0的质量
            self.ball_start_cnt[id] += 1
            # if id == 0:
            #     self.curve_sufficient[0] = True

        # 采样点足够，进行当前数据拟合
        self.x_coeff[id], self.x_error_rate[id] = self.linear_polyfit(self.ts[start_cnt:], self.xs[start_cnt:], w=self.x_ws[start_cnt:])
        self.z_coeff[id], self.z_error_rate[id] = self.linear_polyfit(self.ts[start_cnt:], self.zs[start_cnt:], w=self.x_ws[start_cnt:])
        self.y_coeff[id], self.y_error_rate[id] = self.constrained_polyfit(self.ts[start_cnt:], self.ys[start_cnt:], self.y_ws[start_cnt:])   

        # 如果是第二或第三曲线，前期采样点不够多，需要与之前曲线的反弹预测进行融合
        if id >= 1 and n_sample < self.move_frames_threshold[id]:
            x2_coeff, y2_coeff, z2_coeff = self.predict_next_curve_v2(self.land_point[id-1], self.land_speed[id-1])
            self.y_coeff[id] = y2_coeff # 依然沿用上一曲线的y轴拟合，因为y对采样数更敏感
            self.x_coeff[id] = (self.x_coeff[id] * n_sample + x2_coeff * (self.move_frames_threshold[id] - n_sample)) / self.move_frames_threshold[id]
            self.z_coeff[id] = (self.z_coeff[id] * n_sample + z2_coeff * (self.move_frames_threshold[id] - n_sample)) / self.move_frames_threshold[id]
        # else: # 反之曲线已经自足了，不需要加入接球点的抖动项
        #     self.curve_sufficient[id] = True

        # 对当前曲线进行质检。 根据y轴是否能落地，以及z轴速度是否合理判断。如果质量有误，机器人退出当前接球行动，并重置追踪。
        vz = np.polyval(np.polyder(self.z_coeff[id]), self.ts[-1])
        if n_sample >=6 and (vz * is_bot_fire < Curve.Z_SPEED_RANGE[0] or vz * is_bot_fire > Curve.Z_SPEED_RANGE[1]):
            print(f"  Z speed is not quqlified! return None. Time: {self.ts[-1]} Predict landing ball {self.land_point[id]}, speed: {self.land_speed[id]}, z_coeff: {self.z_coeff[id]} is not good, zs is {self.zs[start_cnt:]}")
            self.is_curve_valid = False
            return -1
        
        ts_at_ground = self.calc_t_at_height(self.y_coeff[id], 0)
        if ts_at_ground is None:
            # self.logger.info(f"  Predict curve has no root, Return None!  y_coeff: {self.y_coeff[id]}")
            return -1
        
        # 计算当前曲线落地点与落地速度，并预测接下来曲线的落点和速度
        self.land_point[id], self.land_speed[id] = self.calc_land_point_and_speed(self.x_coeff[id], self.y_coeff[id], self.z_coeff[id], self.ts[-1], max(np.roots(self.y_coeff[id])))

        for i in range(id, 2):
            self.x_coeff[i+1], self.y_coeff[i+1], self.z_coeff[i+1] = self.predict_next_curve_v2(self.land_point[i], self.land_speed[i])
            self.land_point[i+1], self.land_speed[i+1] = self.calc_land_point_and_speed(self.x_coeff[i+1], self.y_coeff[i+1], self.z_coeff[i+1], self.land_point[i][-1], max(np.roots(self.y_coeff[i+1])))

        # 曲线全部预测完毕，开始计算接球点. 依次计算第二曲线上升、下降， 第三曲线上升、下降的接球点。 
 
        self.loc_results = [] 
        # 这里的打分得结合预测准确性考虑，第一曲线下降期还有第二曲线上升期预测的结果较为准确，因为这期间看的球比较多
        self.add_receive_locs_at_height(1, 1.0, 1.5) 
        # 室内场景下，第二曲线下降期看到的球较少
        self.add_receive_locs_at_height(2, 1.2, 1.45) 
        # self.add_receive_locs_at_height(2, 2, 5)

        return self.loc_results
    
    # 返回当前曲线id，曲线0和1的落点速度
    def get_current_curve_land_speed(self):
        id = len(self.ball_start_cnt) - 1
        return id, self.land_speed[0], self.land_speed[1]
    
    # update: 曲线1修改为起跳速度, 返回格式是id0的v
    def get_bounce_speed(self):
        id = len(self.ball_start_cnt) - 1
        # return id, self.land_speed[0], self.land_speed[1]
        land_t = self.land_point[0][-1]
        # 计算从落地点 到 高点的速度，再补偿空气阻力系数
        bounce_vxz = math.sqrt((self.xs[-1] - self.land_point[0][0])**2 + (self.zs[-1] - self.land_point[0][2])**2) / (self.ts[-1] - land_t) #+  (self.AIR_DRAG_K * 15 * (self.ts[-1] - land_t))
        bounce_vy = np.polyval(np.polyder(self.y_coeff[1]), land_t) #+ (self.AIR_DRAG_K * 15 * (self.ts[-1] - land_t))
        
        print(f"t gap: {self.ts[-1] - land_t}, bounce vxz: {bounce_vxz} ")
        return id, self.land_speed[0], [0, -bounce_vy, 0, bounce_vxz]
    
    # 用来获得机器发球时的初始速度
    # 其中vy使用y系数在发球高度时的值， 而vxz，采用落点距离除以落点时间
    def get_fire_speed(self):
        fire_t = self.calc_t_at_height(self.y_coeff[0], 0.2)[0]
        fire_vy = np.polyval(np.polyder(self.y_coeff[0]), fire_t)
        dis = math.sqrt(self.land_point[0][0]**2 + self.land_point[0][2]**2)
        fire_vxz = dis / (self.land_point[0][-1]-fire_t)
        
        return fire_vy, fire_vxz
    # 获得t时刻，抛物线的点
    def get_point_at_time(self, t):
        for i in range(3):
            if self.land_point[i] is None:
                return None
            if t < self.land_point[i][-1]: # 表示在第i曲线内
                return [np.polyval(self.x_coeff[i], t), np.polyval(self.y_coeff[i], t), np.polyval(self.z_coeff[i], t)]
        return None
    

    # 计算在某一高度时的速度
    #  deprecated    
    # def calculate_velocity_at_height(self, target_height=0.2):
        
    #     """
    #     Calculate the velocity of the ball at a specific height (e.g., y = 20 cm).

    #     Args:
    #         target_height (float): The target height in meters (default is 0.2 m).

    #     Returns:
    #         np.array: Velocity vector [v_x, v_y, v_z] at the specified height.
    #     """
    #     # Use the first segment of the trajectory (id=0)
    #     id = 0

    #     # Extract y(t) polynomial and solve for t when y = target_height
    #     y_coeff = self.y_coeff[id]
    #     print(f"y_coeff:{y_coeff}, target_height:{target_height}")
    #     y_coeff_at_target = y_coeff.copy()
    #     y_coeff_at_target[-1] -= target_height  # Subtract target height from constant term

    #     # Find roots to get t values where y = target_height
    #     t_candidates = np.roots(y_coeff_at_target)
    #     print(f"target_hight ts:{t_candidates}")
    #     t_candidates = [t for t in t_candidates if np.isreal(t)]
    #     if not t_candidates:
    #         print(f"No valid time found for height {target_height}.")
    #         return None
    #     print(f"t_candidates:{t_candidates}")
    #     # Select the earliest valid time
    #     t_at_target = min(t_candidates)

    #     # Calculate velocity at t_at_target
    #     vx_coeff = np.polyder(self.x_coeff[id])
    #     vy_coeff = np.polyder(self.y_coeff[id])
    #     vz_coeff = np.polyder(self.z_coeff[id])

    #     vx = np.polyval(vx_coeff, t_at_target)
    #     vy = np.polyval(vy_coeff, t_at_target)
    #     vz = np.polyval(vz_coeff, t_at_target)

    #     velocity = np.array([vx, vy, vz])
    #     print(f"Velocity at height {target_height}: {velocity} (time: {t_at_target}) ,vx:{vx},vy:{vy},vz:{vz}")
    #     return velocity,t_at_target
    
    
    # 计算在某一高度时的速度
    # deprecated    
    # def calculate_shot_boll_time(self, target_height=0.2):
        
    #     """
    #     Calculate the velocity of the ball at a specific height (e.g., y = 20 cm).

    #     Args:
    #         target_height (float): The target height in meters (default is 0.2 m).

    #     Returns:
    #         np.array: Velocity vector [v_x, v_y, v_z] at the specified height.
    #     """
    #     # Use the first segment of the trajectory (id=0)
    #     id = 0

    #     # Extract y(t) polynomial and solve for t when y = target_height
    #     y_coeff = self.y_coeff[id]
    #     print(f"y_coeff:{y_coeff}, target_height:{target_height}")
    #     y_coeff_at_target = y_coeff.copy()
    #     y_coeff_at_target[-1] -= target_height  # Subtract target height from constant term

    #     # Find roots to get t values where y = target_height
    #     t_candidates = np.roots(y_coeff_at_target)
    #     t_candidates = [t for t in t_candidates if np.isreal(t)]
    #     if not t_candidates:
    #         print(f"No valid time found for height {target_height}.")
    #         return None
    #     print(f"t_candidates:{t_candidates}")
    #     # Select the earliest valid time
    #     t_at_target = min(t_candidates)

    #     # Calculate velocity at t_at_target
    #     vx_coeff = np.polyder(self.x_coeff[id])
    #     vy_coeff = np.polyder(self.y_coeff[id])
    #     vz_coeff = np.polyder(self.z_coeff[id])

    #     vx = np.polyval(vx_coeff, t_at_target)
    #     vy = np.polyval(vy_coeff, t_at_target)
    #     vz = np.polyval(vz_coeff, t_at_target)

    #     velocity = np.array([vx, vy, vz])
    #     print(f"Velocity at height {target_height}: {velocity} (time: {t_at_target}) ,vx:{vx},vy:{vy},vz:{vz}")
    #     return velocity
    
    # 计算网球在过网时的高度
    def calc_net_clearance(self):
        # 根据z系数，计算网球位于z=12时的时间（即过网时刻）. 然后根据y系数，得到过网高度
        t_at_net = (12-self.z_coeff[0][1]) / self.z_coeff[0][0]

        return np.polyval(self.y_coeff[0], t_at_net)
# -*- coding: utf-8 -*-
"""
压电泵微流控可视化操作系统 (HMI)
功能模块：
1. 主控仪表盘 (Dashboard)
2. 实时曲线监控 (Real-time Curves)
3. 参数配置面板 (Control Panel)
4. 系统控制按钮 (Control Buttons)
5. 报警与事件管理 (Alarm & Event Log)
6. 数据导出与分析 (Data Export)
7. 系统自检与诊断 (Self-Diagnostics)
"""

import time
import math
import json
import random
import csv
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
from enum import Enum
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import numpy as np

# Matplotlib 嵌入Tkinter
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D

# 尝试导入scipy进行FFT分析
try:
    from scipy.fft import fft, fftfreq

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("警告: 未安装scipy，频谱分析功能将不可用")


# -----------------------------
# 配置与常量
# -----------------------------
class PumpState(Enum):
    STOPPED = "停止"
    STANDBY = "待机"
    RUNNING = "运行"
    FAULT = "故障"
    CRITICAL = "紧急"


class AlarmLevel(Enum):
    WARNING = "警告"
    FAULT = "故障"
    CRITICAL = "紧急"


class OperationMode(Enum):
    MANUAL = "手动模式"
    AUTO = "自动模式"
    LEARNING = "学习模式"


class ScheduleStrategy(Enum):
    FIXED = "固定台数"
    ADAPTIVE = "热负载自适应"
    BALANCED = "寿命均衡"


@dataclass
class Config:
    # 驱动参数
    target_flow_ml_min: float = 13.5
    drive_freq_hz: float = 1097.0
    drive_voltage_v: float = 1200.0
    duty_cycle: float = 50.0  # %

    # PID参数
    kp: float = 18.0
    ki: float = 2.5
    kd: float = 1.5

    # 保护阈值
    temp_limit_c: float = 45.0
    pressure_limit_kpa: float = 220.0
    flow_deviation_limit: float = 2.0  # ±mL/min

    # 系统配置
    sample_dt: float = 0.2
    history_len: int = 2000  # 扩展历史长度
    pump_count: int = 3
    max_history_hours: float = 24.0

    # 调度策略
    schedule_strategy: ScheduleStrategy = ScheduleStrategy.ADAPTIVE

    # 显示配置
    time_window_seconds: float = 300.0  # 默认5分钟


# -----------------------------
# PID控制器
# -----------------------------
class PID:
    def __init__(self, kp, ki, kd, out_min=-250, out_max=250):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.integral = 0.0
        self.prev_error = 0.0
        self.output = 0.0

    def update(self, setpoint, measured, dt):
        error = setpoint - measured
        self.integral += error * dt
        self.integral = max(-100, min(100, self.integral))  # 积分限幅
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        self.output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(self.out_min, min(self.out_max, self.output))

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.output = 0.0


# -----------------------------
# 压电驱动器
# -----------------------------
class PiezoDriver:
    def __init__(self, pump_id, base_freq, base_voltage):
        self.pump_id = pump_id
        self.freq_hz = base_freq
        self.voltage_v = base_voltage
        self.duty_cycle = 50.0
        self.enabled = False
        self.state = PumpState.STANDBY
        self.run_time_total = 0.0  # 累计运行时间(秒)
        self.start_time = None

    def set_params(self, freq_hz=None, voltage_v=None, duty_cycle=None):
        if freq_hz is not None:
            self.freq_hz = max(100, min(5000, freq_hz))
        if voltage_v is not None:
            self.voltage_v = max(100, min(1600, voltage_v))
        if duty_cycle is not None:
            self.duty_cycle = max(0, min(100, duty_cycle))

    def start(self):
        if self.state != PumpState.FAULT and self.state != PumpState.CRITICAL:
            self.enabled = True
            self.state = PumpState.RUNNING
            self.start_time = time.time()

    def stop(self):
        self.enabled = False
        if self.start_time:
            self.run_time_total += time.time() - self.start_time
            self.start_time = None
        if self.state == PumpState.RUNNING:
            self.state = PumpState.STANDBY

    def set_fault(self, level: AlarmLevel):
        self.stop()
        if level == AlarmLevel.CRITICAL:
            self.state = PumpState.CRITICAL
        else:
            self.state = PumpState.FAULT

    def reset_fault(self):
        if self.state in [PumpState.FAULT, PumpState.CRITICAL]:
            self.state = PumpState.STANDBY

    def get_run_time(self):
        rt = self.run_time_total
        if self.enabled and self.start_time:
            rt += time.time() - self.start_time
        return rt


# -----------------------------
# 泵物理模型
# -----------------------------
class PumpPlant:
    def __init__(self):
        self.flow = 0.0
        self.pressure = 0.0
        self.temp = 28.0
        self.leak = False
        self.flow_noise = 0.0

    def step(self, driver: PiezoDriver, thermal_load, dt):
        if not driver.enabled:
            target_flow = 0.0
        else:
            # 更精确的流量模型
            voltage_factor = 0.0065 * driver.voltage_v
            freq_factor = 0.0028 * (driver.freq_hz / 10.0)
            load_penalty = 0.03 * thermal_load
            duty_factor = driver.duty_cycle / 100.0

            target_flow = (voltage_factor + freq_factor - load_penalty) * duty_factor
            target_flow = max(0.0, min(25.0, target_flow))

            # 添加谐振特性
            if abs(driver.freq_hz - 1097) < 50:
                target_flow *= 1.15  # 谐振增益

        # 一阶惯性 + 噪声
        self.flow += (target_flow - self.flow) * min(1.0, dt * 2.5)
        self.flow += random.uniform(-0.1, 0.1)
        self.flow = max(0, self.flow)

        # 压力模型
        self.pressure = 45 + 7.5 * self.flow + random.uniform(-2, 2)

        # 温度模型（考虑功耗和散热）
        power = (driver.voltage_v / 1200.0) ** 2 * (driver.freq_hz / 1097.0) * (driver.duty_cycle / 100.0)
        temp_rise = 0.06 * power + 0.015 * thermal_load
        cooling = 0.08 * (self.temp - 24.0)  # 环境温度24°C
        self.temp += (temp_rise - cooling) * dt + random.uniform(-0.03, 0.03)

        # 泄漏检测
        if random.random() < 0.0005:
            self.leak = True

        return self.flow, self.pressure, self.temp, self.leak


# -----------------------------
# 报警管理
# -----------------------------
class AlarmManager:
    def __init__(self):
        self.active_alarms = []
        self.alarm_history = deque(maxlen=1000)
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def raise_alarm(self, source, message, level: AlarmLevel):
        alarm = {
            'id': len(self.alarm_history),
            'timestamp': datetime.now(),
            'source': source,
            'message': message,
            'level': level,
            'active': True,
            'acknowledged': False
        }
        self.active_alarms.append(alarm)
        self.alarm_history.append(alarm)

        for cb in self.callbacks:
            cb(alarm)

    def clear_alarm(self, alarm_id):
        for alarm in self.active_alarms:
            if alarm['id'] == alarm_id:
                alarm['active'] = False
                self.active_alarms.remove(alarm)
                break

    def acknowledge_all(self):
        for alarm in self.active_alarms:
            alarm['acknowledged'] = True

    def get_active_alarms(self):
        return sorted(self.active_alarms, key=lambda x: x['timestamp'], reverse=True)

    def get_alarm_history(self, limit=100):
        return list(self.alarm_history)[-limit:]


# -----------------------------
# 数据记录器
# -----------------------------
class DataLogger:
    def __init__(self, system):
        self.system = system
        self.data_buffer = deque(maxlen=10000)  # 扩展缓冲区
        self.log_file = f"pump_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.csv_file = f"pump_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'elapsed_time_s', 'status', 'thermal_load',
                'avg_flow', 'avg_pressure', 'avg_temp', 'power_consumption',
                'pump1_flow', 'pump1_temp', 'pump1_pressure', 'pump1_state',
                'pump2_flow', 'pump2_temp', 'pump2_pressure', 'pump2_state',
                'pump3_flow', 'pump3_temp', 'pump3_pressure', 'pump3_state'
            ])

    def log(self, payload: dict):
        # JSON格式记录
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

        # CSV格式记录
        self.data_buffer.append(payload)
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                payload.get('timestamp', ''),
                payload.get('time_s', 0),
                payload.get('status', ''),
                payload.get('thermal_load', 0),
                payload.get('avg_flow_ml_min', 0),
                payload.get('avg_pressure_kpa', 0),
                payload.get('avg_temp_c', 0),
                payload.get('power_w', 0),
                # Pump 1
                payload['pumps'][0]['flow'] if 'pumps' in payload else 0,
                payload['pumps'][0]['temp'] if 'pumps' in payload else 0,
                payload['pumps'][0]['pressure'] if 'pumps' in payload else 0,
                payload['pumps'][0]['state'] if 'pumps' in payload else '',
                # Pump 2
                payload['pumps'][1]['flow'] if 'pumps' in payload else 0,
                payload['pumps'][1]['temp'] if 'pumps' in payload else 0,
                payload['pumps'][1]['pressure'] if 'pumps' in payload else 0,
                payload['pumps'][1]['state'] if 'pumps' in payload else '',
                # Pump 3
                payload['pumps'][2]['flow'] if 'pumps' in payload else 0,
                payload['pumps'][2]['temp'] if 'pumps' in payload else 0,
                payload['pumps'][2]['pressure'] if 'pumps' in payload else 0,
                payload['pumps'][2]['state'] if 'pumps' in payload else '',
            ])

    def export_csv(self, filepath, start_time=None, end_time=None):
        """导出指定时间段的CSV数据"""
        filtered_data = []
        for record in self.data_buffer:
            ts = record.get('timestamp', '')
            if start_time and ts < start_time:
                continue
            if end_time and ts > end_time:
                continue
            filtered_data.append(record)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'time_s', 'avg_flow', 'avg_temp', 'avg_pressure'])
            for r in filtered_data:
                writer.writerow([
                    r.get('timestamp', ''),
                    r.get('time_s', 0),
                    r.get('avg_flow_ml_min', 0),
                    r.get('avg_temp_c', 0),
                    r.get('avg_pressure_kpa', 0)
                ])
        return len(filtered_data)


# -----------------------------
# 主系统类
# -----------------------------
class PiezoMicrofluidicSystem:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.drivers = [PiezoDriver(i + 1, cfg.drive_freq_hz, cfg.drive_voltage_v)
                        for i in range(cfg.pump_count)]
        self.plants = [PumpPlant() for _ in range(cfg.pump_count)]
        self.pid = PID(cfg.kp, cfg.ki, cfg.kd)
        self.alarm_manager = AlarmManager()
        self.logger = DataLogger(self)
        self.start_time = time.time()
        self.system_state = PumpState.STANDBY
        self.operation_mode = OperationMode.AUTO
        self.thermal_load = 40.0
        self.total_run_time = 0.0
        self.efficiency = 0.0  # mL/min·W

        # 数据历史
        self.ts = deque(maxlen=cfg.history_len)
        self.flow_hist = deque(maxlen=cfg.history_len)
        self.press_hist = deque(maxlen=cfg.history_len)
        self.temp_hist = deque(maxlen=cfg.history_len)
        self.power_hist = deque(maxlen=cfg.history_len)
        self.load_hist = deque(maxlen=cfg.history_len)

        self.running = False
        self.paused = False
        self.frame_count = 0

        # 报警状态
        self.leak_detected = False
        self.over_temp = False
        self.over_pressure = False

    def simulate_external_load(self, t):
        """模拟外部热负载波动"""
        base = 45
        variation = 22 * math.sin(2 * math.pi * t / 40.0)
        noise = random.uniform(-3, 3)
        return base + variation + noise

    def calculate_power(self):
        """计算系统总功耗"""
        total_power = 0
        for d in self.drivers:
            if d.enabled:
                # 简化功耗模型: P ~ V^2 * f * duty / 1e6
                p = (d.voltage_v ** 2) * d.freq_hz * (d.duty_cycle / 100.0) / 1e6
                total_power += p
        return total_power

    def calculate_efficiency(self, avg_flow, power):
        """计算系统效率 mL/min·W"""
        if power > 0:
            return avg_flow / power
        return 0.0

    def check_safety(self, avg_temp, avg_pressure, leak_any):
        """安全检查"""
        status = "OK"

        if leak_any and not self.leak_detected:
            self.leak_detected = True
            self.alarm_manager.raise_alarm("系统", "检测到泄漏", AlarmLevel.CRITICAL)
            return False, "LEAK_DETECTED"

        if avg_temp >= self.cfg.temp_limit_c and not self.over_temp:
            self.over_temp = True
            self.alarm_manager.raise_alarm("系统", f"过温: {avg_temp:.1f}°C", AlarmLevel.CRITICAL)
            return False, "OVER_TEMP"
        elif avg_temp < self.cfg.temp_limit_c - 2:
            self.over_temp = False

        if avg_pressure >= self.cfg.pressure_limit_kpa and not self.over_pressure:
            self.over_pressure = True
            self.alarm_manager.raise_alarm("系统", f"过压: {avg_pressure:.1f} kPa", AlarmLevel.CRITICAL)
            return False, "OVER_PRESSURE"
        elif avg_pressure < self.cfg.pressure_limit_kpa - 10:
            self.over_pressure = False

        # 流量偏差检查
        flow_error = abs(avg_flow - self.cfg.target_flow_ml_min)
        if flow_error > self.cfg.flow_deviation_limit:
            if not any(a['message'].startswith('流量偏差') for a in self.alarm_manager.active_alarms):
                self.alarm_manager.raise_alarm("系统", f"流量偏差: ±{flow_error:.2f} mL/min", AlarmLevel.WARNING)

        return True, status

    def schedule_pumps(self, thermal_load):
        """多泵调度策略"""
        strategy = self.cfg.schedule_strategy

        if strategy == ScheduleStrategy.FIXED:
            # 固定启用所有泵
            need = self.cfg.pump_count
        elif strategy == ScheduleStrategy.ADAPTIVE:
            # 热负载自适应
            if thermal_load < 30:
                need = 1
            elif thermal_load < 60:
                need = 2
            else:
                need = 3
        elif strategy == ScheduleStrategy.BALANCED:
            # 寿命均衡模式：选择运行时间最短的泵
            run_times = [(i, d.get_run_time()) for i, d in enumerate(self.drivers)]
            run_times.sort(key=lambda x: x[1])
            need = 2 if thermal_load < 50 else 3
            # 启用运行时间最短的need个泵
            for i, _ in run_times[:need]:
                self.drivers[i].start()
            for i, _ in run_times[need:]:
                self.drivers[i].stop()
            return

        for i, d in enumerate(self.drivers):
            if i < need:
                if d.state not in [PumpState.FAULT, PumpState.CRITICAL]:
                    d.start()
            else:
                d.stop()

    def step(self):
        """系统单步仿真"""
        now = time.time()
        t = now - self.start_time

        if self.running and not self.paused:
            # 1. 更新热负载
            self.thermal_load = self.simulate_external_load(t)

            # 2. 多泵调度
            self.schedule_pumps(self.thermal_load)

            # 3. PID控制（仅在自动模式下）
            if self.operation_mode == OperationMode.AUTO:
                current_flows = [p.flow for p in self.plants if self.drivers[i].enabled]
                if current_flows:
                    avg_flow = sum(current_flows) / len(current_flows)
                    delta_v = self.pid.update(self.cfg.target_flow_ml_min, avg_flow, self.cfg.sample_dt)

                    for d in self.drivers:
                        if d.enabled:
                            d.set_params(
                                freq_hz=d.freq_hz + 0.02 * delta_v,
                                voltage_v=d.voltage_v + delta_v
                            )

            # 4. 更新物理模型
            flows, pressures, temps = [], [], []
            leak_any = False

            for i, (d, p) in enumerate(zip(self.drivers, self.plants)):
                f, pr, tp, leak = p.step(d, self.thermal_load, self.cfg.sample_dt)
                flows.append(f)
                pressures.append(pr)
                temps.append(tp)
                leak_any = leak_any or leak

                # 单泵状态检查
                if tp > self.cfg.temp_limit_c:
                    d.set_fault(AlarmLevel.CRITICAL)
                    self.alarm_manager.raise_alarm(f"泵#{i + 1}", f"过温: {tp:.1f}°C", AlarmLevel.CRITICAL)

            avg_flow = sum(flows) / len(flows)
            avg_pressure = sum(pressures) / len(pressures)
            avg_temp = max(temps)  # 使用最高温度

            # 5. 安全检查
            ok, status = self.check_safety(avg_temp, avg_pressure, leak_any)
            if not ok:
                self.emergency_stop()
                self.system_state = PumpState.CRITICAL

            # 6. 计算功耗和效率
            power = self.calculate_power()
            self.efficiency = self.calculate_efficiency(avg_flow, power)

            # 7. 记录数据
            payload = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "time_s": round(t, 2),
                "status": status,
                "thermal_load": round(self.thermal_load, 2),
                "avg_flow_ml_min": round(avg_flow, 3),
                "avg_pressure_kpa": round(avg_pressure, 3),
                "avg_temp_c": round(avg_temp, 3),
                "power_w": round(power, 3),
                "efficiency": round(self.efficiency, 4),
                "pumps": [
                    {
                        "id": i + 1,
                        "flow": round(flows[i], 3),
                        "pressure": round(pressures[i], 3),
                        "temp": round(temps[i], 3),
                        "state": self.drivers[i].state.value,
                        "enabled": self.drivers[i].enabled,
                        "freq": round(self.drivers[i].freq_hz, 2),
                        "voltage": round(self.drivers[i].voltage_v, 2)
                    }
                    for i in range(self.cfg.pump_count)
                ]
            }
            self.logger.log(payload)

            # 8. 更新历史数据
            self.ts.append(t)
            self.flow_hist.append(avg_flow)
            self.press_hist.append(avg_pressure)
            self.temp_hist.append(avg_temp)
            self.power_hist.append(power)
            self.load_hist.append(self.thermal_load)

            self.frame_count += 1

            return payload

        return None

    def start_system(self):
        """启动系统"""
        self.running = True
        self.system_state = PumpState.RUNNING
        self.start_time = time.time()
        self.pid.reset()

    def stop_system(self):
        """停止系统"""
        self.running = False
        for d in self.drivers:
            d.stop()
        self.system_state = PumpState.STANDBY

    def emergency_stop(self):
        """紧急停止"""
        self.running = False
        for d in self.drivers:
            d.stop()
            d.state = PumpState.CRITICAL
        self.system_state = PumpState.CRITICAL

    def reset_alarms(self):
        """复位报警"""
        self.leak_detected = False
        self.over_temp = False
        self.over_pressure = False
        for d in self.drivers:
            d.reset_fault()
        self.alarm_manager.active_alarms.clear()
        self.system_state = PumpState.STANDBY

    def set_manual_params(self, pump_id, freq, voltage, duty):
        """手动模式设置参数"""
        if 0 <= pump_id < len(self.drivers):
            self.drivers[pump_id].set_params(freq, voltage, duty)

    def get_pump_status(self):
        """获取所有泵状态"""
        return [
            {
                'id': i + 1,
                'state': d.state,
                'enabled': d.enabled,
                'run_time': d.get_run_time(),
                'freq': d.freq_hz,
                'voltage': d.voltage_v,
                'duty': d.duty_cycle
            }
            for i, d in enumerate(self.drivers)
        ]


# -----------------------------
# GUI界面类
# -----------------------------
class HMIApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("压电泵微流控可视化操作系统 v2.0")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        # 初始化系统
        self.cfg = Config()
        self.system = PiezoMicrofluidicSystem(self.cfg)
        self.update_interval = int(self.cfg.sample_dt * 1000)  # ms

        # 创建界面
        self._create_menu()
        self._create_main_layout()
        self._create_dashboard()
        self._create_curve_panel()
        self._create_control_panel()
        self._create_alarm_panel()
        self._create_status_bar()

        # 绑定报警回调
        self.system.alarm_manager.add_callback(self._on_alarm)

        # 启动更新循环
        self._schedule_update()

    def _create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="导出CSV", command=self._export_csv)
        file_menu.add_command(label="导出报表", command=self._generate_report)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)

        # 视图菜单
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图", menu=view_menu)
        view_menu.add_command(label="时间窗口: 30秒", command=lambda: self._set_time_window(30))
        view_menu.add_command(label="时间窗口: 5分钟", command=lambda: self._set_time_window(300))
        view_menu.add_command(label="时间窗口: 30分钟", command=lambda: self._set_time_window(1800))

        # 分析菜单
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="分析", menu=analysis_menu)
        analysis_menu.add_command(label="频谱分析", command=self._show_spectrum)

    def _create_main_layout(self):
        """创建主布局"""
        # 主框架
        self.main_frame = ttk.Frame(self.root, padding="5")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=3)  # 左侧面板
        self.main_frame.columnconfigure(1, weight=2)  # 右侧面板
        self.main_frame.rowconfigure(0, weight=1)

    def _create_dashboard(self):
        """创建主控仪表盘"""
        dashboard = ttk.LabelFrame(self.main_frame, text="主控仪表盘", padding="10")
        dashboard.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        dashboard.columnconfigure(0, weight=1)

        # 系统状态
        status_frame = ttk.Frame(dashboard)
        status_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(status_frame, text="系统状态:", font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(status_frame, text="待机", font=('Arial', 12), foreground='gray')
        self.status_label.pack(side=tk.LEFT, padx=5)

        self.mode_label = ttk.Label(status_frame, text="[自动模式]", font=('Arial', 10))
        self.mode_label.pack(side=tk.LEFT, padx=20)

        # 运行时间
        self.runtime_label = ttk.Label(status_frame, text="运行时间: 00:00:00")
        self.runtime_label.pack(side=tk.RIGHT, padx=5)

        # 关键指标面板
        metrics_frame = ttk.Frame(dashboard)
        metrics_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)

        # 创建指标显示
        self.metrics = {}
        metrics = [
            ('avg_flow', '平均流量', 'mL/min', '#2196F3'),
            ('max_temp', '最高温度', '°C', '#FF9800'),
            ('pressure', '系统压力', 'kPa', '#4CAF50'),
            ('thermal_load', '热负载', 'W', '#9C27B0'),
            ('power', '功耗', 'W', '#F44336'),
            ('efficiency', '效率', 'mL/min·W', '#00BCD4')
        ]

        for i, (key, name, unit, color) in enumerate(metrics):
            frame = ttk.Frame(metrics_frame, relief='ridge', padding=5)
            frame.grid(row=i // 3, column=i % 3, padx=5, pady=5, sticky=(tk.W, tk.E))
            metrics_frame.columnconfigure(i % 3, weight=1)

            ttk.Label(frame, text=name, font=('Arial', 10)).pack()
            self.metrics[key] = ttk.Label(frame, text="0.00", font=('Arial', 16, 'bold'), foreground=color)
            self.metrics[key].pack()
            ttk.Label(frame, text=unit, font=('Arial', 9)).pack()

        # 泵组状态可视化
        pump_frame = ttk.LabelFrame(dashboard, text="泵组状态", padding="10")
        pump_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)

        self.pump_indicators = []
        for i in range(self.cfg.pump_count):
            pframe = ttk.Frame(pump_frame)
            pframe.grid(row=0, column=i, padx=20, pady=5)

            # 状态指示灯（用Canvas绘制圆形）
            canvas = tk.Canvas(pframe, width=60, height=60, bg='white', highlightthickness=0)
            canvas.pack()
            circle = canvas.create_oval(10, 10, 50, 50, fill='gray', outline='black')
            self.pump_indicators.append({
                'canvas': canvas,
                'circle': circle,
                'label': ttk.Label(pframe, text=f"泵#{i + 1}\n待机", font=('Arial', 9)),
                'info': ttk.Label(pframe, text="F:0Hz\nV:0V", font=('Arial', 8))
            })
            self.pump_indicators[i]['label'].pack()
            self.pump_indicators[i]['info'].pack()

    def _create_curve_panel(self):
        """创建实时曲线监控面板"""
        curve_frame = ttk.LabelFrame(self.main_frame, text="实时曲线监控", padding="5")
        curve_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        curve_frame.rowconfigure(0, weight=1)
        curve_frame.columnconfigure(0, weight=1)

        # 创建matplotlib图表
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # 初始化曲线
        self.lines = {}
        self.lines['flow'], = self.ax.plot([], [], 'b-', label='流量(mL/min)', linewidth=1.5)
        self.lines['temp'], = self.ax.plot([], [], 'r-', label='温度(°C)', linewidth=1.5)
        self.lines['pressure'], = self.ax.plot([], [], 'g-', label='压力(kPa)', linewidth=1.5)
        self.lines['power'], = self.ax.plot([], [], 'm-', label='功耗(W)', linewidth=1.5)

        self.ax.set_xlabel('时间 (s)', fontsize=10)
        self.ax.set_ylabel('数值', fontsize=10)
        self.ax.set_title('实时运行曲线', fontsize=12)
        self.ax.legend(loc='upper left', fontsize=8)
        self.ax.grid(True, alpha=0.3)

        # 目标流量线
        self.target_line = self.ax.axhline(self.cfg.target_flow_ml_min, color='b',
                                           linestyle='--', linewidth=1, alpha=0.5, label='目标流量')

        # 嵌入Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=curve_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 添加工具栏
        toolbar = NavigationToolbar2Tk(self.canvas, curve_frame, pack_toolbar=False)
        toolbar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # 游标数据标签
        self.cursor_label = ttk.Label(curve_frame, text="游标: --", font=('Arial', 9))
        self.cursor_label.grid(row=2, column=0, sticky=tk.W, pady=2)

        # 绑定鼠标事件
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    def _create_control_panel(self):
        """创建控制面板"""
        control_frame = ttk.LabelFrame(self.main_frame, text="系统控制", padding="10")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)

        # 控制按钮
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.LEFT, padx=10)

        self.start_btn = ttk.Button(btn_frame, text="▶ 启动系统", command=self._start_system)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="⏹ 停止系统", command=self._stop_system)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.emergency_btn = ttk.Button(btn_frame, text="⛔ 紧急停止", command=self._emergency_stop)
        self.emergency_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = ttk.Button(btn_frame, text="🔄 复位报警", command=self._reset_alarms)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        # 模式选择
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(side=tk.LEFT, padx=20)

        ttk.Label(mode_frame, text="运行模式:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="AUTO")
        ttk.Radiobutton(mode_frame, text="手动", variable=self.mode_var,
                        value="MANUAL", command=self._change_mode).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="自动", variable=self.mode_var,
                        value="AUTO", command=self._change_mode).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="学习", variable=self.mode_var,
                        value="LEARNING", command=self._change_mode).pack(side=tk.LEFT, padx=5)

        # 参数设置
        param_frame = ttk.Frame(control_frame)
        param_frame.pack(side=tk.RIGHT, padx=10)

        ttk.Label(param_frame, text="目标流量:").pack(side=tk.LEFT)
        self.target_flow_var = tk.DoubleVar(value=self.cfg.target_flow_ml_min)
        ttk.Spinbox(param_frame, from_=0, to=25, increment=0.1,
                    textvariable=self.target_flow_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(param_frame, text="mL/min").pack(side=tk.LEFT)

        ttk.Button(param_frame, text="应用", command=self._apply_params).pack(side=tk.LEFT, padx=10)

    def _create_alarm_panel(self):
        """创建报警面板"""
        alarm_frame = ttk.LabelFrame(self.main_frame, text="报警与事件", padding="5")
        alarm_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)

        # 报警列表
        self.alarm_tree = ttk.Treeview(alarm_frame, columns=('time', 'level', 'source', 'message'),
                                       show='headings', height=5)
        self.alarm_tree.heading('time', text='时间')
        self.alarm_tree.heading('level', text='级别')
        self.alarm_tree.heading('source', text='来源')
        self.alarm_tree.heading('message', text='消息')

        self.alarm_tree.column('time', width=150)
        self.alarm_tree.column('level', width=80)
        self.alarm_tree.column('source', width=100)
        self.alarm_tree.column('message', width=400)

        self.alarm_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 滚动条
        scrollbar = ttk.Scrollbar(alarm_frame, orient=tk.VERTICAL, command=self.alarm_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.alarm_tree.configure(yscrollcommand=scrollbar.set)

        # 报警统计
        self.alarm_stats = ttk.Label(alarm_frame, text="活跃报警: 0 | 今日总计: 0")
        self.alarm_stats.pack(side=tk.BOTTOM, anchor=tk.W, pady=2)

    def _create_status_bar(self):
        """创建状态栏"""
        self.status_bar = ttk.Label(self.root, text="就绪 | 系统未运行 | 数据帧: 0",
                                    relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=2)

    def _update_display(self):
        """更新所有显示"""
        if not self.system.running:
            return

        payload = self.system.step()
        if not payload:
            return

        # 更新仪表盘
        self.status_label.config(
            text=payload['status'],
            foreground='green' if payload['status'] == 'OK' else 'red'
        )

        # 更新指标
        self.metrics['avg_flow'].config(text=f"{payload['avg_flow_ml_min']:.2f}")
        self.metrics['max_temp'].config(text=f"{payload['avg_temp_c']:.2f}")
        self.metrics['pressure'].config(text=f"{payload['avg_pressure_kpa']:.2f}")
        self.metrics['thermal_load'].config(text=f"{payload['thermal_load']:.2f}")
        self.metrics['power'].config(text=f"{payload['power_w']:.2f}")
        self.metrics['efficiency'].config(text=f"{payload['efficiency']:.4f}")

        # 更新泵状态指示器
        for i, pump in enumerate(payload['pumps']):
            state = pump['state']
            color = {
                '运行': '#4CAF50',
                '待机': '#FF9800',
                '停止': '#9E9E9E',
                '故障': '#F44336',
                '紧急': '#B71C1C'
            }.get(state, 'gray')

            self.pump_indicators[i]['canvas'].itemconfig(
                self.pump_indicators[i]['circle'], fill=color
            )
            self.pump_indicators[i]['label'].config(text=f"泵#{i + 1}\n{state}")
            self.pump_indicators[i]['info'].config(
                text=f"F:{pump['freq']:.0f}Hz\nV:{pump['voltage']:.0f}V"
            )

        # 更新曲线
        self._update_curves()

        # 更新运行时间
        elapsed = int(time.time() - self.system.start_time)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        self.runtime_label.config(text=f"运行时间: {hours:02d}:{minutes:02d}:{seconds:02d}")

        # 更新状态栏
        self.status_bar.config(
            text=f"运行中 | 模式: {self.system.operation_mode.value} | "
                 f"帧: {self.system.frame_count} | "
                 f"报警: {len(self.system.alarm_manager.active_alarms)}"
        )

    def _update_curves(self):
        """更新曲线图"""
        if len(self.system.ts) < 2:
            return

        # 获取时间窗口数据
        xs = list(self.system.ts)
        window = self.cfg.time_window_seconds

        # 找到窗口起始索引
        if len(xs) > 0:
            latest = xs[-1]
            start_idx = 0
            for i, x in enumerate(xs):
                if latest - x <= window:
                    start_idx = i
                    break

            xs = xs[start_idx:]
            flow = list(self.system.flow_hist)[start_idx:]
            temp = list(self.system.temp_hist)[start_idx:]
            press = list(self.system.press_hist)[start_idx:]
            power = list(self.system.power_hist)[start_idx:]

            # 更新曲线数据
            self.lines['flow'].set_data(xs, flow)
            self.lines['temp'].set_data(xs, temp)
            self.lines['pressure'].set_data(xs, press)
            self.lines['power'].set_data(xs, power)

            # 调整坐标轴
            if xs:
                self.ax.set_xlim(max(0, xs[-1] - window), xs[-1] + 1)

            # 自动调整Y轴
            all_data = flow + temp + press + power
            if all_data:
                ymin, ymax = min(all_data), max(all_data)
                margin = (ymax - ymin) * 0.1 if ymax != ymin else 1
                self.ax.set_ylim(ymin - margin, ymax + margin)

            self.canvas.draw_idle()

    def _on_mouse_move(self, event):
        """鼠标悬停显示数据"""
        if event.inaxes != self.ax:
            return

        x = event.xdata
        if x is None or len(self.system.ts) == 0:
            return

        # 找到最近的数据点
        xs = list(self.system.ts)
        idx = min(range(len(xs)), key=lambda i: abs(xs[i] - x))

        if idx < len(self.system.flow_hist):
            t = xs[idx]
            f = list(self.system.flow_hist)[idx]
            tp = list(self.system.temp_hist)[idx]
            p = list(self.system.press_hist)[idx]
            pw = list(self.system.power_hist)[idx]

            self.cursor_label.config(
                text=f"游标: t={t:.1f}s | 流量:{f:.2f} | 温度:{tp:.2f} | "
                     f"压力:{p:.2f} | 功耗:{pw:.2f}"
            )

    def _on_alarm(self, alarm):
        """报警回调"""
        # 更新报警列表
        self.alarm_tree.insert('', 0, values=(
            alarm['timestamp'].strftime('%H:%M:%S'),
            alarm['level'].value,
            alarm['source'],
            alarm['message']
        ))

        # 限制显示数量
        if len(self.alarm_tree.get_children()) > 50:
            self.alarm_tree.delete(self.alarm_tree.get_children()[-1])

        # 更新统计
        total = len(self.system.alarm_manager.alarm_history)
        self.alarm_stats.config(
            text=f"活跃报警: {len(self.system.alarm_manager.active_alarms)} | 今日总计: {total}"
        )

        # 严重报警弹窗
        if alarm['level'] == AlarmLevel.CRITICAL:
            messagebox.showerror("严重报警",
                                 f"[{alarm['source']}] {alarm['message']}\n"
                                 f"时间: {alarm['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

    def _start_system(self):
        """启动系统"""
        self.system.start_system()
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.emergency_btn.config(state='normal')

    def _stop_system(self):
        """停止系统"""
        self.system.stop_system()
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')

    def _emergency_stop(self):
        """紧急停止"""
        self.system.emergency_stop()
        messagebox.showwarning("紧急停止", "系统已紧急停止！请检查设备状态。")
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')

    def _reset_alarms(self):
        """复位报警"""
        self.system.reset_alarms()
        # 清空报警显示
        for item in self.alarm_tree.get_children():
            self.alarm_tree.delete(item)

    def _change_mode(self):
        """切换运行模式"""
        mode = self.mode_var.get()
        if mode == "MANUAL":
            self.system.operation_mode = OperationMode.MANUAL
        elif mode == "AUTO":
            self.system.operation_mode = OperationMode.AUTO
        elif mode == "LEARNING":
            self.system.operation_mode = OperationMode.LEARNING

        self.mode_label.config(text=f"[{self.system.operation_mode.value}]")

    def _apply_params(self):
        """应用参数"""
        try:
            target = self.target_flow_var.get()
            self.cfg.target_flow_ml_min = target
            self.system.cfg.target_flow_ml_min = target
            # 更新目标线
            self.target_line.set_ydata([target, target])
            self.canvas.draw_idle()
            messagebox.showinfo("成功", f"目标流量已设置为 {target} mL/min")
        except Exception as e:
            messagebox.showerror("错误", f"参数设置失败: {str(e)}")

    def _set_time_window(self, seconds):
        """设置时间窗口"""
        self.cfg.time_window_seconds = seconds
        self._update_curves()

    def _export_csv(self):
        """导出CSV"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        if filepath:
            count = self.system.logger.export_csv(filepath)
            messagebox.showinfo("导出完成", f"已导出 {count} 条记录到:\n{filepath}")

    def _generate_report(self):
        """生成运行报表"""
        # 简化实现
        if not self.system.flow_hist:
            messagebox.showwarning("警告", "无数据可生成报表")
            return

        report = f"""=== 压电泵系统运行报表 ===
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
运行时长: {len(self.system.ts) * self.cfg.sample_dt:.1f} 秒
平均流量: {sum(self.system.flow_hist) / len(self.system.flow_hist):.3f} mL/min
最高温度: {max(self.system.temp_hist):.2f} °C
平均压力: {sum(self.system.press_hist) / len(self.system.press_hist):.3f} kPa
总能耗: {sum(self.system.power_hist) * self.cfg.sample_dt:.3f} J
平均效率: {sum(self.system.power_hist) / len(self.system.power_hist) if self.system.power_hist else 0:.4f} mL/min·W
报警次数: {len(self.system.alarm_manager.alarm_history)}
"""

        # 显示报表
        report_window = tk.Toplevel(self.root)
        report_window.title("运行报表")
        report_window.geometry("500x400")

        text = scrolledtext.ScrolledText(report_window, wrap=tk.WORD, padx=10, pady=10)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, report)
        text.config(state=tk.DISABLED)

    def _show_spectrum(self):
        """显示频谱分析"""
        if not HAS_SCIPY:
            messagebox.showerror("错误", "未安装scipy库，无法使用频谱分析功能")
            return

        if len(self.system.flow_hist) < 100:
            messagebox.showwarning("警告", "数据不足，需要至少100个数据点")
            return

        # 创建频谱窗口
        spec_window = tk.Toplevel(self.root)
        spec_window.title("频谱分析")
        spec_window.geometry("600x400")

        # 计算FFT
        flow_data = np.array(list(self.system.flow_hist))
        n = len(flow_data)
        yf = fft(flow_data)
        xf = fftfreq(n, self.cfg.sample_dt)[:n // 2]
        yf_abs = 2.0 / n * np.abs(yf[0:n // 2])

        # 绘制频谱
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(xf, yf_abs, 'b-', linewidth=1)
        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('幅值')
        ax.set_title('流量信号频谱分析')
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=spec_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def _schedule_update(self):
        """定时更新"""
        self._update_display()
        self.root.after(self.update_interval, self._schedule_update)

    def run(self):
        """运行应用"""
        self.root.mainloop()


# -----------------------------
# 主程序入口
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = HMIApplication(root)
    app.run()

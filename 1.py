# -*- coding: utf-8 -*-
"""
压电泵微流控系统仿真 + 实时折线图（可执行）
功能覆盖：
1) 高压频率驱动（1097Hz / 1200V，软件层仿真）
2) 流量/压力闭环控制（PID）
3) 温度与状态监测（阈值保护）
4) 多泵协同调度（按热负载动态启停）
5) 通信与数据记录（JSON日志）
6) 实时显示微泵流量变化折线图
7) 无外部输入时全系统自动仿真运行

运行方式：
python piezo_pump_system.py
依赖：
pip install matplotlib
"""

import time
import math
import json
import random
import sys
from dataclasses import dataclass, asdict
from collections import deque
import matplotlib

matplotlib.use('TkAgg')  # 设置后端，确保兼容性
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# -----------------------------
# 配置
# -----------------------------
@dataclass
class Config:
    target_flow_ml_min: float = 13.5
    flow_tolerance: float = 0.01  # ±1%
    drive_freq_hz: float = 1097.0
    drive_voltage_v: float = 1200.0
    temp_target_c: float = 36.4
    temp_limit_c: float = 45.0
    pressure_limit_kpa: float = 220.0
    sample_dt: float = 0.2  # 控制周期(s)
    history_len: int = 200  # 折线图窗口点数
    pump_count: int = 3
    enable_simulation_without_input: bool = True  # 无输入自动模拟


# -----------------------------
# PID
# -----------------------------
class PID:
    def __init__(self, kp, ki, kd, out_min=-200, out_max=200):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, setpoint, measured, dt):
        error = setpoint - measured
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(self.out_min, min(self.out_max, out))


# -----------------------------
# 驱动模块（仿真层）
# -----------------------------
class PiezoDriver:
    def __init__(self, base_freq, base_voltage):
        self.freq_hz = base_freq
        self.voltage_v = base_voltage
        self.enabled = True

    def set_params(self, freq_hz=None, voltage_v=None):
        if freq_hz is not None:
            self.freq_hz = max(100, min(5000, freq_hz))
        if voltage_v is not None:
            self.voltage_v = max(100, min(1600, voltage_v))

    def stop(self):
        self.enabled = False

    def start(self):
        self.enabled = True


# -----------------------------
# 单泵物理近似模型（仿真）
# -----------------------------
class PumpPlant:
    """
    简化动态模型：
    flow ~ a*V + b*f + c*负载 + 扰动
    pressure ~ 与流量相关
    temp ~ 与功耗和负载相关，带散热项
    """

    def __init__(self):
        self.flow = 8.0
        self.pressure = 80.0
        self.temp = 32.0
        self.leak = False

    def step(self, driver: PiezoDriver, thermal_load, dt):
        if not driver.enabled:
            target_flow = 0.0
        else:
            # 经验近似：电压/频率提升促进流量，负载抑制流量
            target_flow = (
                    0.0065 * driver.voltage_v
                    + 0.0028 * (driver.freq_hz / 10.0)
                    - 0.03 * thermal_load
            )
            target_flow = max(0.0, min(25.0, target_flow))

        # 一阶惯性
        self.flow += (target_flow - self.flow) * min(1.0, dt * 2.5)

        # 压力与流量正相关 + 噪声
        self.pressure = 45 + 7.5 * self.flow + random.uniform(-2, 2)

        # 温度：功耗相关（~V^2）+ 负载，散热回落
        power_term = (driver.voltage_v / 1200.0) ** 2 * (driver.freq_hz / 1097.0)
        temp_rise = 0.06 * power_term + 0.015 * thermal_load
        cooling = 0.08 * (self.temp - 28.0)
        self.temp += (temp_rise - cooling) * dt + random.uniform(-0.03, 0.03)

        # 低概率泄漏事件（仿真）
        if random.random() < 0.0005:
            self.leak = True

        return self.flow, self.pressure, self.temp, self.leak


# -----------------------------
# 多泵调度
# -----------------------------
class PumpScheduler:
    def __init__(self, drivers):
        self.drivers = drivers

    def dispatch(self, thermal_load):
        # 简单策略：按热负载启停泵
        # <30: 1台; 30~60: 2台; >=60: 3台
        need = 1 if thermal_load < 30 else (2 if thermal_load < 60 else 3)
        for i, d in enumerate(self.drivers):
            if i < need:
                d.start()
            else:
                d.stop()


# -----------------------------
# 数据记录与"通信"
# -----------------------------
class DataLogger:
    def __init__(self, filename="pump_runtime_log.jsonl"):
        self.filename = filename

    def log(self, payload: dict):
        try:
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"日志写入错误: {e}")


# -----------------------------
# 主系统
# -----------------------------
class PiezoMicrofluidicSystem:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.drivers = [PiezoDriver(cfg.drive_freq_hz, cfg.drive_voltage_v) for _ in range(cfg.pump_count)]
        self.plants = [PumpPlant() for _ in range(cfg.pump_count)]
        self.scheduler = PumpScheduler(self.drivers)
        self.pid = PID(kp=18, ki=2.5, kd=1.5, out_min=-250, out_max=250)
        self.logger = DataLogger()
        self.start_time = time.time()

        # 实时曲线缓存
        self.ts = deque(maxlen=cfg.history_len)
        self.flow_hist = deque(maxlen=cfg.history_len)
        self.press_hist = deque(maxlen=cfg.history_len)
        self.temp_hist = deque(maxlen=cfg.history_len)

        self.running = True
        self.frame_count = 0  # 用于控制缓存

    def simulate_external_load(self, t):
        # 无输入情况下自动生成"热负载"波动
        return 45 + 22 * math.sin(2 * math.pi * t / 40.0) + random.uniform(-3, 3)

    def safety_check(self, avg_temp, avg_pressure, leak_any):
        if leak_any:
            return False, "LEAK_DETECTED"
        if avg_temp >= self.cfg.temp_limit_c:
            return False, "OVER_TEMP"
        if avg_pressure >= self.cfg.pressure_limit_kpa:
            return False, "OVER_PRESSURE"
        return True, "OK"

    def step(self):
        now = time.time()
        t = now - self.start_time

        # 1) 无输入自动仿真负载
        thermal_load = self.simulate_external_load(t) if self.cfg.enable_simulation_without_input else 40.0

        # 2) 多泵调度
        self.scheduler.dispatch(thermal_load)

        # 3) 闭环控制：以"平均流量"追踪目标
        current_flows = [p.flow for p in self.plants]
        avg_flow = sum(current_flows) / len(current_flows)
        delta_v = self.pid.update(self.cfg.target_flow_ml_min, avg_flow, self.cfg.sample_dt)

        # 按同样增量调所有已启用泵的电压，频率微调
        for d in self.drivers:
            if d.enabled:
                d.set_params(
                    freq_hz=d.freq_hz + 0.02 * delta_v,
                    voltage_v=d.voltage_v + delta_v
                )

        # 4) 植物更新
        flows, pressures, temps = [], [], []
        leak_any = False
        for d, p in zip(self.drivers, self.plants):
            f, pr, tp, leak = p.step(d, thermal_load, self.cfg.sample_dt)
            flows.append(f);
            pressures.append(pr);
            temps.append(tp)
            leak_any = leak_any or leak

        avg_flow = sum(flows) / len(flows)
        avg_pressure = sum(pressures) / len(pressures)
        avg_temp = sum(temps) / len(temps)

        # 5) 安全检查
        ok, status = self.safety_check(avg_temp, avg_pressure, leak_any)
        if not ok:
            for d in self.drivers:
                d.stop()
            self.running = False

        # 6) 记录 + 上报（JSON）
        payload = {
            "time_s": round(t, 2),
            "status": status,
            "thermal_load": round(thermal_load, 2),
            "avg_flow_ml_min": round(avg_flow, 3),
            "avg_pressure_kpa": round(avg_pressure, 3),
            "avg_temp_c": round(avg_temp, 3),
            "drivers": [
                {"enabled": d.enabled, "freq_hz": round(d.freq_hz, 2), "voltage_v": round(d.voltage_v, 2)}
                for d in self.drivers
            ]
        }
        self.logger.log(payload)

        # 7) 更新曲线数据
        self.ts.append(t)
        self.flow_hist.append(avg_flow)
        self.press_hist.append(avg_pressure)
        self.temp_hist.append(avg_temp)

        self.frame_count += 1
        return payload


def main():
    cfg = Config()
    system = PiezoMicrofluidicSystem(cfg)

    # 图形初始化
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(10, 5))
    line_flow, = ax.plot([], [], lw=2, label="Flow (mL/min)")
    ax.axhline(cfg.target_flow_ml_min, linestyle="--", linewidth=1.5, label="Target 13.5")
    text_box = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title("Piezo Pump Real-time Flow Curve")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flow (mL/min)")
    ax.legend(loc="upper right")
    ax.grid(True)

    def init():
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 25)
        line_flow.set_data([], [])
        return line_flow, text_box

    def update(_frame):
        if system.running:
            payload = system.step()
        else:
            payload = {
                "status": "STOPPED",
                "avg_flow_ml_min": system.flow_hist[-1] if system.flow_hist else 0.0,
                "avg_temp_c": system.temp_hist[-1] if system.temp_hist else 0.0,
                "avg_pressure_kpa": system.press_hist[-1] if system.press_hist else 0.0
            }

        xs = list(system.ts)
        ys = list(system.flow_hist)
        if xs:
            xmin = max(0, xs[-1] - 40)
            xmax = xmin + 40
            ax.set_xlim(xmin, xmax)

        line_flow.set_data(xs, ys)
        text_box.set_text(
            f"Status: {payload['status']}\n"
            f"Flow: {payload['avg_flow_ml_min']:.2f} mL/min\n"
            f"Temp: {payload['avg_temp_c']:.2f} °C\n"
            f"Pressure: {payload['avg_pressure_kpa']:.2f} kPa"
        )
        return line_flow, text_box

    # 关键修改：添加 cache_frame_data=False 消除警告
    ani = FuncAnimation(
        fig,
        update,
        init_func=init,
        interval=int(cfg.sample_dt * 1000),
        blit=False,
        cache_frame_data=False,  # 禁用帧缓存，避免警告
        save_count=cfg.history_len  # 设置最大保存帧数
    )

    plt.tight_layout()

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n用户中断，系统停止")
        system.running = False
    finally:
        print(f"系统运行结束，共采集 {system.frame_count} 帧数据")
        plt.close('all')


if __name__ == "__main__":
    main()

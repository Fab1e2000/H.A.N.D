# PC-Arduino 仿生手舵机控制系统

这是一个用于 PC + Arduino + PCA9685 的舵机控制项目，目标是让你可以：

1. 在电脑界面上选择多路舵机并统一控制弯曲幅度（0-100%）。
2. 给每一路舵机设置独立标定范围（最小角度/最大角度，可反向）。
3. 通过串口把命令发送到 Arduino，再由 Arduino 驱动 PCA9685 控制舵机。
4. 在界面中实时看到每个关节的角度信息和三维手部模型变化。

适合人群：第一次接触本项目、第一次做仿生手联调、希望快速跑通并二次开发的人。

## 功能总览

- 图形界面控制（Tkinter）
- 12 路常用舵机通道（0,1,2,4,5,6,8,9,10,12,13,14）
- 幅度滑条控制（0-100）
- 串口发送节流（拖动滑条时不会疯狂刷串口）
- 每路舵机独立标定并保存到 JSON
- 三维模型 + 三视图联动显示

## 目录结构

- `servocontrol.py`: PC 端主程序（GUI + 串口通信 + 模型显示）
- `handtracking.py`: 手部视觉追踪程序（MediaPipe + 逆解模型 + 标定）
- `requirements.txt`: Python 依赖
- `servo_calibration.json`: 舵机标定数据（运行中可更新）
- `distance_calibration.json`: 追踪侧摆/距离标定数据
- `hand_model_config.json`: 手部几何模型配置
- `arduino/pc_servo_controller/pc_servo_controller.ino`: Arduino 固件

## 运行前准备

### 1. 硬件准备

- Arduino 开发板（Uno/Nano 等）
- PCA9685 舵机驱动板（默认地址 `0x40`）
- 舵机电源（建议独立供电，不要直接吃 Arduino 5V）
- 多个舵机
- USB 数据线

### 2. 软件准备

- Windows（当前项目已在 Windows 环境使用）
- Python 3.10+
- Arduino IDE
- Arduino 库：`Adafruit PWM Servo Driver Library`

## 快速开始（建议按顺序）

### 步骤 A：安装 Python 依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

### 步骤 B：烧录 Arduino 固件

1. 打开 `arduino/pc_servo_controller/pc_servo_controller.ino`。
2. 在 Arduino IDE 中安装库 `Adafruit PWM Servo Driver Library`。
3. 连接 Arduino 与 PCA9685（I2C：SDA/SCL，电源正确连接）。
4. 选择正确板型和端口，点击上传。
5. 串口波特率请与 PC 端一致，默认 `115200`。

### 步骤 C：启动 PC 控制程序

```bash
python servocontrol.py
```

打开界面后：

1. 选择串口（例如 `COM3`）。
2. 点击“连接”。
3. 勾选要控制的舵机通道。
4. 拖动幅度滑条，观察舵机动作和模型变化。

## 界面说明

### 选择区块

- 按手指分组显示通道。
- 每个通道旁会显示“真实角度”。
- 支持“全选/清空”。

### 标定区块

- 选择舵机编号。
- 设置最小角度和最大角度（0-180）。
- 点击“保存标定”写入 `servo_calibration.json`。
- 支持反向：当 `min > max` 时会反向映射。

### 控制区块

- 滑条 0-100% 表示弯曲幅度。
- 表格显示每路舵机当前映射角度、关节角度和标定范围。
- “发送当前幅度”：立刻下发当前命令。
- “回中位 (50%)”：快速回中。

## 串口协议

PC 每行发送一条命令，格式如下：

```text
S,<channel>,<angle>
```

示例：

```text
S,0,90
S,1,120
```

Arduino 会返回：

- `OK,<channel>,<angle>`：执行成功
- `ERR,...`：格式错误、通道错误或其他异常

## 配置文件说明

### servo_calibration.json

- 用于保存每个通道的三类映射数据：
  - `servo`：旋转程度到舵机角度的端点范围（用于标定）
  - `joint`：旋转程度到关节角度的端点范围（用于物理模型）
	- `mapping`：显式映射点（`degree` -> `servo_angle`/`joint_angle`），默认只使用 `0` 和 `100` 两点做线性映射
- 典型结构：

```json
{
	"0": {
		"servo": { "min": 23, "max": 120 },
		"joint": { "min": 10, "max": 90 },
		"mapping": [
			{ "degree": 0, "servo_angle": 23, "joint_angle": 10 },
			{ "degree": 100, "servo_angle": 120, "joint_angle": 90 }
		]
	}
}
```

- 兼容性说明：旧格式 `{ "min": ..., "max": ... }` 仍可读取，程序会自动补全为新结构。

## 两个映射逻辑详解

系统里使用同一个控制输入 `degree`（0-100）驱动两条并行映射链路，这两条链路服务于不同目标：

1. `degree -> servo_angle`：用于硬件执行和舵机标定。
2. `degree -> joint_angle`：用于关节物理意义表达和三维模型驱动。

其中 `degree` 是归一化弯曲程度，不是直接物理角度。它的作用是把“控制层输入”统一成 0-100，再分别投影到舵机空间与关节空间。

### 映射逻辑 1：degree -> servo_angle（标定链路）

- 目标：把统一的弯曲程度映射成具体舵机角度，最终通过串口命令下发到 Arduino/PCA9685。
- 数据来源：`servo_calibration.json` 中每个通道的 `servo.min` 与 `servo.max`（或 `mapping` 的 0/100 端点）。
- 线性公式：

$$
servo\_angle = servo\_min + \frac{degree}{100} \cdot (servo\_max - servo\_min)
$$

- 说明：
	- 当 `servo_min < servo_max` 时是正向映射。
	- 当 `servo_min > servo_max` 时自动形成反向映射（同样满足线性关系）。
	- `degree=0` 对应 `servo_min`，`degree=100` 对应 `servo_max`。

### 映射逻辑 2：degree -> joint_angle（物理模型链路）

- 目标：把同一控制输入映射为关节角度，用于界面显示、运动学解释和三维手部模型更新。
- 数据来源：`servo_calibration.json` 中每个通道的 `joint.min` 与 `joint.max`（或 `mapping` 的 0/100 端点）。
- 线性公式：

$$
joint\_angle = joint\_min + \frac{degree}{100} \cdot (joint\_max - joint\_min)
$$

- 说明：
	- 这条映射不直接下发给舵机，而是用于“物理层语义”。
	- 允许出现负角度区间（例如 MCP 外展/内收类关节），如 `-15° ~ 15°`。
	- 与舵机链路共用同一个 `degree`，因此硬件动作和模型显示会保持同步。

### 两条映射为何必须同时存在

- 舵机角度是执行器空间，受安装偏置、连杆方向和舵机零位影响。
- 关节角度是生物/机械关节空间，描述的是模型中的真实屈伸或展收含义。
- 同一 `degree` 同时映射到两者后，系统即可满足：
	- 控制上能正确驱动硬件；
	- 表达上能正确展示物理关节状态。

### 端点两点与中间点说明

- 当前项目采用线性映射，因此 `mapping` 使用 `degree=0` 与 `degree=100` 两个端点即可唯一确定整条映射。
- 若将来发现某些通道存在明显非线性，可扩展更多中间点用于分段拟合或插值。

### 示例（以通道 0 为例）

配置：

- `servo: min=23, max=120`
- `joint: min=10, max=90`

当 `degree=50` 时：

- $servo\_angle = 23 + 0.5 \cdot (120-23) = 71.5$
- $joint\_angle = 10 + 0.5 \cdot (90-10) = 50$

含义：硬件会执行约 `71.5°` 的舵机角度，同时模型显示关节约 `50°`。

### hand_model_config.json

- 用于描述四根手指在三维模型中的根部坐标、初始方向和段长。
- 修改后可影响模型显示效果，不直接改变串口协议。

## 常见问题排查

### 1. 打开程序提示缺少模块

执行：

```bash
pip install -r requirements.txt
```

### 2. 看不到串口

- 确认 USB 数据线是“数据线”而非仅充电线。
- 在设备管理器检查端口号。
- 关闭占用串口的软件（Arduino 串口监视器等）。
- 点击界面“刷新”按钮。

### 3. 舵机不动或抖动

- 优先检查舵机供电是否足够。
- 检查地线是否共地（Arduino GND 与电源 GND）。
- 检查 PCA9685 地址是否为 `0x40`。
- 必要时调整固件中的 `SERVOMIN` / `SERVOMAX`。

### 4. 角度方向反了

进入标定区块，把该通道设置为反向区间（例如 `min=180, max=0`），保存后再测试。

## 安全建议

- 首次测试先断开机械连杆，确认角度方向正确后再上机构。
- 不要让舵机长时间堵转。
- 先小幅度测试，再逐步扩大动作范围。

## 开发者提示

- 主逻辑在 `servocontrol.py`，包含：
	- 通道与关节映射
	- 标定映射
	- 串口通信
	- 三维模型计算与渲染
- 若你要扩展新手指或新关节，建议同步更新：
	- 通道元数据
	- `hand_model_config.json`
	- UI 分组展示

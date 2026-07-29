#!/usr/bin/env python3
"""Build the frozen AeroComm-Small8 public/private task package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


SOURCES = [
    {
        "source_id": "NASA-BSF-VISVIVA",
        "title": "Basics of Space Flight, Chapter 4",
        "organization": "NASA",
        "url": "https://science.nasa.gov/learn/basics-of-space-flight/chapter4-1/",
        "supports": ["AC-AERO-01", "AC-AERO-02", "AC-AERO-03"],
        "use": "orbital-energy, transfer-orbit, and Kepler-equation context",
    },
    {
        "source_id": "NASA-JPL-HOHMANN",
        "title": "Let's Go to Mars! Calculating Launch Windows",
        "organization": "NASA/JPL",
        "url": "https://www.jpl.nasa.gov/edu/resources/lesson-plan/lets-go-to-mars-calculating-launch-windows/",
        "supports": ["AC-AERO-02"],
        "use": "real mission-planning context for Hohmann transfers",
    },
    {
        "source_id": "NASA-NTRS-QUATERNION",
        "title": "Spacecraft Attitude Determination Methods",
        "organization": "NASA Goddard Space Flight Center",
        "url": "https://ntrs.nasa.gov/citations/20000115482",
        "supports": ["AC-AERO-04"],
        "use": "spacecraft-attitude and quaternion context",
    },
    {
        "source_id": "NASA-NAIF-ROTATION",
        "title": "SPICE Rotation Required Reading",
        "organization": "NASA/JPL NAIF",
        "url": "https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/rotation.html",
        "supports": ["AC-AERO-04"],
        "use": "unit-quaternion vector-rotation convention",
    },
    {
        "source_id": "ITU-R-P525-5",
        "title": "Calculation of free-space attenuation",
        "organization": "International Telecommunication Union",
        "url": "https://www.itu.int/rec/R-REC-P.525-5-202411-I/en",
        "supports": ["AC-COMM-02"],
        "use": "free-space basic transmission-loss formula",
    },
    {
        "source_id": "CCSDS-TC-CRC",
        "title": "CCSDS Telecommand CRC implementation note",
        "organization": "Consultative Committee for Space Data Systems",
        "url": "https://mailman.ccsds.org/pipermail/sls-slp/attachments/20060505/ea6d712e/attachment.pdf",
        "supports": ["AC-COMM-03"],
        "use": "CRC polynomial x^16+x^12+x^5+1 and initial seed 0xFFFF",
    },
    {
        "source_id": "COMMPY",
        "title": "CommPy channel coding documentation",
        "organization": "CommPy contributors",
        "url": "https://commpy.readthedocs.io/en/latest/channelcoding.html",
        "supports": ["AC-COMM-01", "AC-COMM-04"],
        "use": "digital modulation and convolutional-code software context",
    },
    {
        "source_id": "KOMM-CONV",
        "title": "Komm ConvolutionalCode reference",
        "organization": "Komm contributors",
        "url": "https://komm.dev/ref/ConvolutionalCode/",
        "supports": ["AC-COMM-04"],
        "use": "independent published example for generators (7,5) in octal",
    },
]


TASKS = [
    {
        "task_id": "AC-AERO-01",
        "domain": "aerospace",
        "scenario": "orbital speed from vis-viva",
        "entry_point": "vis_viva_speed",
        "expected_loc": "6-12",
        "source_ids": ["NASA-BSF-VISVIVA"],
        "problem": r"""# AC-AERO-01 — 轨道状态快速校核：Vis-viva 速度

航天任务分析软件需要一个小型数值原语，用轨道半长轴和当前位置半径校核
二体轨道速度。

实现：

```python
def vis_viva_speed(mu, r, semi_major_axis):
    ...
```

其中 `mu` 是中心天体标准引力参数，`r` 是当前到中心天体的距离，
`semi_major_axis` 是椭圆轨道半长轴；三者使用相容单位。返回同一单位制下的
标量速度：

\[
v=\sqrt{\mu\left(\frac{2}{r}-\frac{1}{a}\right)}.
\]

契约：

- 三个输入必须是有限的实数并且严格大于零，否则抛出 `ValueError`。
- 若根号内表达式为负（输入不对应有效椭圆轨道位置），抛出 `ValueError`。
- 返回 Python `float`，不得读写文件、网络或全局状态。
- 只提交一个完整、可导入的 Python 模块，不要添加命令行输入输出。
""",
        "oracle": """import math

def vis_viva_speed(mu, r, semi_major_axis):
    values = (mu, r, semi_major_axis)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in values):
        raise ValueError("inputs must be finite positive real numbers")
    mu, r, semi_major_axis = map(float, values)
    if not all(math.isfinite(x) and x > 0.0 for x in (mu, r, semi_major_axis)):
        raise ValueError("inputs must be finite positive real numbers")
    radicand = mu * (2.0 / r - 1.0 / semi_major_axis)
    if radicand < 0.0:
        raise ValueError("invalid orbital position")
    return float(math.sqrt(radicand))
""",
        "tests": r"""import math
import pytest
from candidate import vis_viva_speed

def expected(mu, r, a):
    return math.sqrt(mu * (2.0 / r - 1.0 / a))

def test_circular_leo():
    assert vis_viva_speed(398600.4418, 7000.0, 7000.0) == pytest.approx(expected(398600.4418, 7000.0, 7000.0), rel=1e-12)

def test_elliptic_perigee():
    assert vis_viva_speed(398600.4418, 7000.0, 10000.0) == pytest.approx(expected(398600.4418, 7000.0, 10000.0), rel=1e-12)

def test_elliptic_apogee():
    assert vis_viva_speed(398600.4418, 13000.0, 10000.0) == pytest.approx(expected(398600.4418, 13000.0, 10000.0), rel=1e-12)

def test_mars_orbit():
    assert vis_viva_speed(42828.375214, 4000.0, 6000.0) == pytest.approx(expected(42828.375214, 4000.0, 6000.0), rel=1e-12)

def test_returns_python_float():
    assert type(vis_viva_speed(1, 1, 1)) is float

def test_nonpositive_input_rejected():
    for args in [(0, 1, 1), (1, -1, 1), (1, 1, 0)]:
        with pytest.raises(ValueError):
            vis_viva_speed(*args)

def test_nonfinite_input_rejected():
    for value in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError):
            vis_viva_speed(1.0, value, 1.0)

def test_negative_radicand_rejected():
    with pytest.raises(ValueError):
        vis_viva_speed(1.0, 3.0, 1.0)
""",
    },
    {
        "task_id": "AC-AERO-02",
        "domain": "aerospace",
        "scenario": "coplanar Hohmann transfer budget",
        "entry_point": "hohmann_transfer",
        "expected_loc": "12-20",
        "source_ids": ["NASA-BSF-VISVIVA", "NASA-JPL-HOHMANN"],
        "problem": r"""# AC-AERO-02 — 共面圆轨道 Hohmann 转移预算

任务设计人员需要在不运行完整轨道传播器的情况下，快速估算两个同中心、
共面圆轨道之间的理想 Hohmann 转移。

实现：

```python
def hohmann_transfer(mu, r1, r2):
    ...
```

返回三元组 `(delta_v1, delta_v2, time_of_flight)`。两个 `delta_v` 都是燃烧
速度增量的非负**幅值**；`time_of_flight` 是转移椭圆半周期。使用：

\[
a_t=(r_1+r_2)/2,\quad
v_c(r)=\sqrt{\mu/r},\quad
v_t(r)=\sqrt{\mu(2/r-1/a_t)}
\]

\[
\Delta v_1=|v_t(r_1)-v_c(r_1)|,\quad
\Delta v_2=|v_c(r_2)-v_t(r_2)|,\quad
t=\pi\sqrt{a_t^3/\mu}.
\]

契约：

- `mu`、`r1`、`r2` 必须是有限正实数，且 `r1 != r2`，否则抛出
  `ValueError`。
- 既要支持升轨，也要支持降轨。
- 返回恰好三个 Python `float`。
- 只提交完整可导入模块，不要添加文件、网络或命令行 I/O。
""",
        "oracle": """import math

def hohmann_transfer(mu, r1, r2):
    values = (mu, r1, r2)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in values):
        raise ValueError("inputs must be finite positive real numbers")
    mu, r1, r2 = map(float, values)
    if not all(math.isfinite(x) and x > 0.0 for x in values) or r1 == r2:
        raise ValueError("invalid transfer radii")
    at = (r1 + r2) / 2.0
    vc1, vc2 = math.sqrt(mu / r1), math.sqrt(mu / r2)
    vt1 = math.sqrt(mu * (2.0 / r1 - 1.0 / at))
    vt2 = math.sqrt(mu * (2.0 / r2 - 1.0 / at))
    return float(abs(vt1 - vc1)), float(abs(vc2 - vt2)), float(math.pi * math.sqrt(at ** 3 / mu))
""",
        "tests": r"""import math
import pytest
from candidate import hohmann_transfer

def reference(mu, r1, r2):
    at = (r1 + r2) / 2.0
    vc1, vc2 = math.sqrt(mu / r1), math.sqrt(mu / r2)
    vt1 = math.sqrt(mu * (2.0 / r1 - 1.0 / at))
    vt2 = math.sqrt(mu * (2.0 / r2 - 1.0 / at))
    return abs(vt1 - vc1), abs(vc2 - vt2), math.pi * math.sqrt(at ** 3 / mu)

def assert_transfer(args):
    got = hohmann_transfer(*args)
    assert type(got) is tuple and len(got) == 3
    assert all(type(x) is float for x in got)
    assert got == pytest.approx(reference(*args), rel=1e-12)

def test_leo_to_geo():
    assert_transfer((398600.4418, 7000.0, 42164.0))

def test_geo_to_leo():
    assert_transfer((398600.4418, 42164.0, 7000.0))

def test_two_to_one_radius_ratio():
    assert_transfer((398600.4418, 7000.0, 14000.0))

def test_mars_transfer():
    assert_transfer((42828.375214, 4000.0, 17000.0))

def test_burns_are_magnitudes():
    dv1, dv2, _ = hohmann_transfer(398600.4418, 42164.0, 7000.0)
    assert dv1 > 0.0 and dv2 > 0.0

def test_nonpositive_rejected():
    for args in [(0, 1, 2), (1, 0, 2), (1, 2, -3)]:
        with pytest.raises(ValueError):
            hohmann_transfer(*args)

def test_equal_radii_rejected():
    with pytest.raises(ValueError):
        hohmann_transfer(1.0, 2.0, 2.0)

def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        hohmann_transfer(math.inf, 1.0, 2.0)
""",
    },
    {
        "task_id": "AC-AERO-03",
        "domain": "aerospace",
        "scenario": "elliptic Kepler-equation solve",
        "entry_point": "kepler_eccentric_anomaly",
        "expected_loc": "12-22",
        "source_ids": ["NASA-BSF-VISVIVA"],
        "problem": r"""# AC-AERO-03 — 椭圆轨道 Kepler 方程求解

轨道传播器需要把平均近点角转换为偏近点角。实现标量求解器：

```python
def kepler_eccentric_anomaly(mean_anomaly, eccentricity, tol=1e-12, max_iter=50):
    ...
```

求解

\[
M=E-e\sin(E)
\]

并返回 `E`。先把 `M` 归一化到 `[0, 2π)`；结果也必须在该区间。使用确定性的
Newton 迭代，停止条件为相邻两次 `E` 的绝对变化不大于 `tol`。

契约：

- `mean_anomaly` 必须是有限实数；`0 <= eccentricity < 1`。
- `tol` 必须是有限正实数；`max_iter` 必须是正整数。违反时抛出
  `ValueError`。
- 若达到迭代上限仍未收敛，抛出 `RuntimeError`。
- 返回 Python `float`；不得调用 SciPy 或读写外部状态。
- 只提交完整可导入模块，不要添加命令行 I/O。
""",
        "oracle": """import math

def kepler_eccentric_anomaly(mean_anomaly, eccentricity, tol=1e-12, max_iter=50):
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in (mean_anomaly, eccentricity, tol)):
        raise ValueError("invalid numeric input")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int):
        raise ValueError("max_iter must be a positive integer")
    m, e, tol = float(mean_anomaly), float(eccentricity), float(tol)
    if not math.isfinite(m) or not math.isfinite(e) or not 0.0 <= e < 1.0 or not math.isfinite(tol) or tol <= 0.0 or max_iter <= 0:
        raise ValueError("invalid solver parameters")
    m %= 2.0 * math.pi
    estimate = m if e < 0.8 else math.pi
    for _ in range(max_iter):
        updated = estimate - (estimate - e * math.sin(estimate) - m) / (1.0 - e * math.cos(estimate))
        if abs(updated - estimate) <= tol:
            return float(updated % (2.0 * math.pi))
        estimate = updated
    raise RuntimeError("Newton iteration did not converge")
""",
        "tests": r"""import math
import pytest
from candidate import kepler_eccentric_anomaly

def check_solution(m, e, **kwargs):
    result = kepler_eccentric_anomaly(m, e, **kwargs)
    normalized = m % (2.0 * math.pi)
    assert type(result) is float
    assert 0.0 <= result < 2.0 * math.pi
    residual = (result - e * math.sin(result)) - normalized
    assert abs(residual) <= max(2e-11, kwargs.get("tol", 1e-12) * 2)

def test_circular_normalization():
    assert kepler_eccentric_anomaly(8.0 * math.pi + 0.3, 0.0) == pytest.approx(0.3, abs=1e-12)

def test_zero_mean_high_eccentricity():
    assert kepler_eccentric_anomaly(0.0, 0.9) == pytest.approx(0.0, abs=1e-12)

def test_low_eccentricity():
    check_solution(1.0, 0.1)

def test_moderate_eccentricity():
    check_solution(2.4, 0.75)

def test_negative_mean_anomaly():
    check_solution(-0.8, 0.4)

def test_large_mean_anomaly():
    check_solution(123.456, 0.6)

def test_near_parabolic_elliptic_case():
    check_solution(0.05, 0.999, tol=1e-13, max_iter=100)

def test_invalid_eccentricity():
    for e in (-0.01, 1.0, 1.2):
        with pytest.raises(ValueError):
            kepler_eccentric_anomaly(1.0, e)

def test_invalid_controls():
    for kwargs in ({"tol": 0}, {"max_iter": 0}, {"max_iter": 1.5}):
        with pytest.raises(ValueError):
            kepler_eccentric_anomaly(1.0, 0.1, **kwargs)

def test_iteration_limit_is_enforced():
    with pytest.raises(RuntimeError):
        kepler_eccentric_anomaly(0.1, 0.999, max_iter=1)
""",
    },
    {
        "task_id": "AC-AERO-04",
        "domain": "aerospace",
        "scenario": "spacecraft attitude vector rotation",
        "entry_point": "quaternion_rotate",
        "expected_loc": "10-18",
        "source_ids": ["NASA-NTRS-QUATERNION", "NASA-NAIF-ROTATION"],
        "problem": r"""# AC-AERO-04 — 航天器姿态四元数向量旋转

姿态处理软件需要把机体系测量向量旋转到目标坐标系。实现：

```python
def quaternion_rotate(q, vectors):
    ...
```

`q` 采用 scalar-first 顺序 `[w, x, y, z]`。函数必须先归一化非零四元数，
再执行主动旋转 `v' = q * (0,v) * conjugate(q)`。`vectors` 可以是形状
`(3,)` 的单个向量，也可以是形状 `(N,3)` 的向量批次。

契约：

- 使用 NumPy；接受 array-like 输入。
- `q` 必须恰有 4 个有限元素且范数非零；`vectors` 必须是上述形状并且所有
  元素有限。违反时抛出 `ValueError`。
- 返回浮点 `numpy.ndarray`，形状与 `vectors` 完全相同。
- 不得修改任何输入。
- 只提交一个完整可导入模块，不要添加文件、网络或命令行 I/O。
""",
        "oracle": """import numpy as np

def quaternion_rotate(q, vectors):
    quat = np.asarray(q, dtype=float)
    vec = np.asarray(vectors, dtype=float)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError("q must have four finite elements")
    if vec.ndim not in (1, 2) or vec.shape[-1:] != (3,) or not np.all(np.isfinite(vec)):
        raise ValueError("vectors must have shape (3,) or (N,3)")
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("zero quaternion")
    w = quat[0] / norm
    xyz = quat[1:] / norm
    first = np.cross(xyz, vec)
    return np.asarray(vec + 2.0 * w * first + 2.0 * np.cross(xyz, first), dtype=float)
""",
        "tests": r"""import numpy as np
import pytest
from candidate import quaternion_rotate

def test_identity_single_vector():
    got = quaternion_rotate([1, 0, 0, 0], [1, -2, 3])
    assert isinstance(got, np.ndarray) and got.dtype.kind == "f" and got.shape == (3,)
    np.testing.assert_allclose(got, [1, -2, 3], atol=1e-12)

def test_z_axis_quarter_turn():
    q = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    np.testing.assert_allclose(quaternion_rotate(q, [1, 0, 0]), [0, 1, 0], atol=1e-12)

def test_x_axis_half_turn():
    np.testing.assert_allclose(quaternion_rotate([0, 1, 0, 0], [0, 2, -3]), [0, -2, 3], atol=1e-12)

def test_nonunit_quaternion_is_normalized():
    np.testing.assert_allclose(quaternion_rotate([2, 0, 0, 2], [1, 0, 0]), [0, 1, 0], atol=1e-12)

def test_batch_rotation_and_shape():
    q = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    vectors = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 2.]])
    got = quaternion_rotate(q, vectors)
    assert got.shape == vectors.shape
    np.testing.assert_allclose(got, [[0, 1, 0], [-1, 0, 0], [0, 0, 2]], atol=1e-12)

def test_inputs_are_not_mutated():
    q = np.array([1., 2., 3., 4.])
    v = np.array([[1., 2., 3.]])
    q0, v0 = q.copy(), v.copy()
    quaternion_rotate(q, v)
    np.testing.assert_array_equal(q, q0)
    np.testing.assert_array_equal(v, v0)

def test_zero_quaternion_rejected():
    with pytest.raises(ValueError):
        quaternion_rotate([0, 0, 0, 0], [1, 2, 3])

def test_bad_shapes_rejected():
    for q, v in [([1, 0, 0], [1, 2, 3]), ([1, 0, 0, 0], [1, 2]), ([1, 0, 0, 0], np.zeros((2, 2, 3)))]:
        with pytest.raises(ValueError):
            quaternion_rotate(q, v)

def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        quaternion_rotate([1, 0, 0, 0], [1, np.nan, 3])
""",
    },
    {
        "task_id": "AC-COMM-01",
        "domain": "communications",
        "scenario": "Gray-coded QPSK baseband mapping",
        "entry_point": "qpsk_gray_modulate",
        "expected_loc": "8-15",
        "source_ids": ["COMMPY"],
        "problem": r"""# AC-COMM-01 — Gray 编码 QPSK 基带映射

数字通信链路仿真需要把二进制流映射为单位平均能量的复数 QPSK 符号。
实现：

```python
def qpsk_gray_modulate(bits):
    ...
```

每两个连续比特组成一符号，使用以下冻结映射：

```text
00 -> ( 1 + 1j) / sqrt(2)
01 -> (-1 + 1j) / sqrt(2)
11 -> (-1 - 1j) / sqrt(2)
10 -> ( 1 - 1j) / sqrt(2)
```

契约：

- `bits` 接受一维 array-like；元素只能是整数 0 或 1，长度必须为偶数。
- 空输入返回形状 `(0,)` 的复数 NumPy 数组。
- 非一维、非二进制或奇数长度输入抛出 `ValueError`。
- 返回形状 `(len(bits)//2,)`、复数 dtype 的 `numpy.ndarray`，不修改输入。
- 只提交一个完整可导入模块，不要添加文件、网络或命令行 I/O。
""",
        "oracle": """import numpy as np

def qpsk_gray_modulate(bits):
    raw = np.asarray(bits)
    if raw.ndim != 1 or raw.size % 2 or not np.all((raw == 0) | (raw == 1)):
        raise ValueError("bits must be an even one-dimensional binary sequence")
    if raw.size and raw.dtype.kind not in "biu":
        raise ValueError("bits must contain integers")
    pairs = raw.astype(np.uint8, copy=False).reshape(-1, 2)
    real = 1.0 - 2.0 * pairs[:, 1]
    imag = 1.0 - 2.0 * pairs[:, 0]
    return (real + 1j * imag) / np.sqrt(2.0)
""",
        "tests": r"""import numpy as np
import pytest
from candidate import qpsk_gray_modulate

S = np.sqrt(2.0)

def test_complete_gray_constellation():
    got = qpsk_gray_modulate([0,0, 0,1, 1,1, 1,0])
    np.testing.assert_allclose(got, [(1+1j)/S, (-1+1j)/S, (-1-1j)/S, (1-1j)/S], atol=1e-12)

def test_single_00_symbol():
    np.testing.assert_allclose(qpsk_gray_modulate([0,0]), [(1+1j)/S], atol=1e-12)

def test_single_01_symbol():
    np.testing.assert_allclose(qpsk_gray_modulate([0,1]), [(-1+1j)/S], atol=1e-12)

def test_single_11_symbol():
    np.testing.assert_allclose(qpsk_gray_modulate([1,1]), [(-1-1j)/S], atol=1e-12)

def test_single_10_symbol():
    np.testing.assert_allclose(qpsk_gray_modulate([1,0]), [(1-1j)/S], atol=1e-12)

def test_empty_input_contract():
    got = qpsk_gray_modulate([])
    assert isinstance(got, np.ndarray) and got.shape == (0,) and got.dtype.kind == "c"

def test_unit_symbol_energy_and_shape():
    got = qpsk_gray_modulate(np.array([1,0,0,1,1,1], dtype=np.uint8))
    assert got.shape == (3,) and got.dtype.kind == "c"
    np.testing.assert_allclose(np.abs(got) ** 2, np.ones(3), atol=1e-12)

def test_odd_length_rejected():
    with pytest.raises(ValueError):
        qpsk_gray_modulate([0,1,0])

def test_nonbinary_and_noninteger_rejected():
    for bits in ([0,2], [0.0,1.0]):
        with pytest.raises(ValueError):
            qpsk_gray_modulate(bits)

def test_input_not_mutated():
    bits = np.array([0,1,1,0], dtype=np.int64)
    original = bits.copy()
    qpsk_gray_modulate(bits)
    np.testing.assert_array_equal(bits, original)
""",
    },
    {
        "task_id": "AC-COMM-02",
        "domain": "communications",
        "scenario": "radio free-space path loss",
        "entry_point": "free_space_path_loss_db",
        "expected_loc": "6-12",
        "source_ids": ["ITU-R-P525-5"],
        "problem": r"""# AC-COMM-02 — 无线链路自由空间路径损耗

卫星/地面通信链路预算需要按 SI 单位计算自由空间基本传输损耗。实现：

```python
def free_space_path_loss_db(distance_m, frequency_hz):
    ...
```

使用精确光速常数 `c = 299792458.0 m/s` 和

\[
L_{fs}=20\log_{10}\left(\frac{4\pi d f}{c}\right)\;\mathrm{dB}.
\]

契约：

- 使用 NumPy，两个参数都接受标量或 array-like，并按 NumPy broadcasting
  规则组合。
- 所有输入必须有限且严格大于零；否则抛出 `ValueError`。
- 广播不兼容时抛出 `ValueError`（不要泄漏 NumPy 的其他异常类型）。
- 标量广播结果返回 Python `float`；其他结果返回浮点 `numpy.ndarray`。
- 不修改输入；只提交完整可导入模块，不要添加外部 I/O。
""",
        "oracle": """import numpy as np

def free_space_path_loss_db(distance_m, frequency_hz):
    d = np.asarray(distance_m, dtype=float)
    f = np.asarray(frequency_hz, dtype=float)
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(f)) or np.any(d <= 0.0) or np.any(f <= 0.0):
        raise ValueError("distance and frequency must be finite and positive")
    try:
        value = 20.0 * np.log10(4.0 * np.pi * d * f / 299792458.0)
    except ValueError as exc:
        raise ValueError("inputs cannot be broadcast") from exc
    if value.ndim == 0:
        return float(value)
    return np.asarray(value, dtype=float)
""",
        "tests": r"""import numpy as np
import pytest
from candidate import free_space_path_loss_db

C = 299792458.0
def reference(d, f):
    return 20.0 * np.log10(4.0 * np.pi * np.asarray(d) * np.asarray(f) / C)

def test_one_kilometre_one_ghz():
    got = free_space_path_loss_db(1000.0, 1.0e9)
    assert type(got) is float
    assert got == pytest.approx(float(reference(1000.0, 1.0e9)), abs=1e-12)

def test_geostationary_s_band():
    assert free_space_path_loss_db(35_786_000.0, 2.2e9) == pytest.approx(float(reference(35_786_000.0, 2.2e9)), abs=1e-12)

def test_zero_db_reference_point():
    frequency = C / (4.0 * np.pi)
    assert free_space_path_loss_db(1.0, frequency) == pytest.approx(0.0, abs=1e-12)

def test_distance_vector():
    d = np.array([1.0, 10.0, 1000.0])
    got = free_space_path_loss_db(d, 437e6)
    assert isinstance(got, np.ndarray) and got.shape == (3,) and got.dtype.kind == "f"
    np.testing.assert_allclose(got, reference(d, 437e6), atol=1e-12)

def test_two_dimensional_broadcast():
    d = np.array([[1.0], [1000.0]])
    f = np.array([[100e6, 1e9, 10e9]])
    got = free_space_path_loss_db(d, f)
    assert got.shape == (2, 3)
    np.testing.assert_allclose(got, reference(d, f), atol=1e-12)

def test_scaling_laws():
    base = free_space_path_loss_db(1000.0, 1e9)
    assert free_space_path_loss_db(2000.0, 1e9) - base == pytest.approx(20*np.log10(2), abs=1e-12)
    assert free_space_path_loss_db(1000.0, 10e9) - base == pytest.approx(20.0, abs=1e-12)

def test_nonpositive_or_nonfinite_rejected():
    for d, f in [(0, 1), (-1, 1), (1, 0), (np.inf, 1), (1, np.nan)]:
        with pytest.raises(ValueError):
            free_space_path_loss_db(d, f)

def test_broadcast_error_normalized():
    with pytest.raises(ValueError):
        free_space_path_loss_db(np.ones(2), np.ones(3))

def test_inputs_not_mutated():
    d, f = np.array([1.,2.]), np.array([3.,4.])
    d0, f0 = d.copy(), f.copy()
    free_space_path_loss_db(d, f)
    np.testing.assert_array_equal(d, d0)
    np.testing.assert_array_equal(f, f0)
""",
    },
    {
        "task_id": "AC-COMM-03",
        "domain": "space_communications",
        "scenario": "CCSDS telecommand frame CRC",
        "entry_point": "ccsds_crc16",
        "expected_loc": "10-18",
        "source_ids": ["CCSDS-TC-CRC"],
        "problem": r"""# AC-COMM-03 — CCSDS 遥控链路 CRC-16

空间链路帧处理器需要按 CCSDS 遥控 CRC 约定生成 16 位校验值。实现：

```python
def ccsds_crc16(data, initial=0xFFFF):
    ...
```

采用非反射的多项式
`x^16 + x^12 + x^5 + 1`（去掉最高项后的十六进制表示为 `0x1021`），
逐字节从最高位到最低位处理；默认初值 `0xFFFF`，无最终异或。返回
`0..65535` 的 Python `int`。

契约：

- `data` 只接受 `bytes`、`bytearray` 或 `memoryview`。
- `initial` 必须是 Python `int`（不接受 `bool`）且位于 `0..0xFFFF`；
  否则抛出 `ValueError`。
- 非法 `data` 类型抛出 `TypeError`。
- 支持分段计算：把上一段返回值作为下一段 `initial`，结果应等于一次处理完整
  字节串。
- 不使用第三方 CRC 库；只提交完整可导入模块，不要添加外部 I/O。
""",
        "oracle": """def ccsds_crc16(data, initial=0xFFFF):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    if isinstance(initial, bool) or not isinstance(initial, int) or not 0 <= initial <= 0xFFFF:
        raise ValueError("initial must be a 16-bit integer")
    crc = initial
    for byte in bytes(data):
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return int(crc)
""",
        "tests": r"""import pytest
from candidate import ccsds_crc16

def reference(data, initial=0xFFFF):
    crc = initial
    for byte in bytes(data):
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc

def test_empty_keeps_seed():
    assert ccsds_crc16(b"") == 0xFFFF

def test_standard_check_vector():
    assert ccsds_crc16(b"123456789") == 0x29B1

def test_zero_bytes():
    assert ccsds_crc16(b"\x00" * 32) == reference(b"\x00" * 32)

def test_all_byte_values():
    data = bytes(range(256))
    assert ccsds_crc16(data) == reference(data)

def test_bytearray_and_memoryview():
    data = bytearray(b"CCSDS telemetry frame")
    assert ccsds_crc16(data) == reference(data)
    assert ccsds_crc16(memoryview(data)) == reference(data)

def test_incremental_processing():
    first, second = b"space-", b"link"
    assert ccsds_crc16(second, ccsds_crc16(first)) == ccsds_crc16(first + second)

def test_custom_initial_value_and_return_type():
    got = ccsds_crc16(b"\x01\x02\x03", initial=0x1D0F)
    assert type(got) is int and 0 <= got <= 0xFFFF and got == reference(b"\x01\x02\x03", 0x1D0F)

def test_invalid_data_type():
    with pytest.raises(TypeError):
        ccsds_crc16([1, 2, 3])

def test_invalid_initial_value():
    for initial in (-1, 65536, True, 1.5):
        with pytest.raises(ValueError):
            ccsds_crc16(b"x", initial)
""",
    },
    {
        "task_id": "AC-COMM-04",
        "domain": "communications",
        "scenario": "rate-one-half convolutional channel coding",
        "entry_point": "convolutional_encode_7_5",
        "expected_loc": "10-18",
        "source_ids": ["COMMPY", "KOMM-CONV"],
        "problem": r"""# AC-COMM-04 — (7,5) 八进制卷积编码

低码率通信链路仿真需要一个可复核的前向纠错编码原语。实现约束长度 3、
码率 1/2、生成多项式 `(7, 5)`（八进制）的非递归卷积编码器：

```python
def convolutional_encode_7_5(bits, terminate=True):
    ...
```

编码器从全零状态开始。对当前输入 `u` 和两个从新到旧的存储位 `(s1, s2)`，
按顺序输出：

```text
y0 = u XOR s1 XOR s2   # generator 7
y1 = u XOR s2          # generator 5
```

然后状态更新为 `(u, s1)`。若 `terminate=True`，在消息后追加两个零输入，使
编码器回到全零状态；这些尾比特的输出也必须保留。

契约：

- `bits` 是一维 array-like，元素只能是整数 0 或 1；否则抛出 `ValueError`。
- `terminate` 必须是 Python `bool`，否则抛出 `ValueError`。
- 返回一维 `numpy.ndarray`，dtype 必须为 `np.uint8`。
- 不修改输入；只提交完整可导入模块，不要添加外部 I/O。
""",
        "oracle": """import numpy as np

def convolutional_encode_7_5(bits, terminate=True):
    raw = np.asarray(bits)
    if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "biu") or not np.all((raw == 0) | (raw == 1)):
        raise ValueError("bits must be a one-dimensional integer binary sequence")
    if type(terminate) is not bool:
        raise ValueError("terminate must be bool")
    stream = list(map(int, raw))
    if terminate:
        stream += [0, 0]
    s1 = s2 = 0
    output = []
    for bit in stream:
        output.extend((bit ^ s1 ^ s2, bit ^ s2))
        s1, s2 = bit, s1
    return np.asarray(output, dtype=np.uint8)
""",
        "tests": r"""import numpy as np
import pytest
from candidate import convolutional_encode_7_5

def reference(bits, terminate):
    stream = list(bits) + ([0, 0] if terminate else [])
    s1 = s2 = 0
    out = []
    for bit in stream:
        out.extend([bit ^ s1 ^ s2, bit ^ s2])
        s1, s2 = bit, s1
    return np.array(out, dtype=np.uint8)

def test_published_komm_example_without_termination():
    np.testing.assert_array_equal(convolutional_encode_7_5([1,1,1,1], False), [1,1,0,1,1,0,1,0])

def test_single_one_with_termination():
    np.testing.assert_array_equal(convolutional_encode_7_5([1], True), [1,1,1,0,1,1])

def test_empty_without_termination():
    got = convolutional_encode_7_5([], False)
    assert got.shape == (0,) and got.dtype == np.uint8

def test_empty_with_termination():
    np.testing.assert_array_equal(convolutional_encode_7_5([], True), [0,0,0,0])

def test_mixed_message_without_termination():
    bits = [1,0,1,1,0,0,1]
    np.testing.assert_array_equal(convolutional_encode_7_5(bits, False), reference(bits, False))

def test_mixed_message_with_termination_and_dtype():
    bits = np.array([0,1,0,1,1], dtype=np.uint8)
    got = convolutional_encode_7_5(bits)
    assert got.dtype == np.uint8 and got.ndim == 1 and len(got) == 2 * (len(bits) + 2)
    np.testing.assert_array_equal(got, reference(bits, True))

def test_input_not_mutated():
    bits = np.array([1,0,1], dtype=np.int64)
    original = bits.copy()
    convolutional_encode_7_5(bits)
    np.testing.assert_array_equal(bits, original)

def test_invalid_bits_rejected():
    for bits in ([0,2], [0.0,1.0], [[0,1]]):
        with pytest.raises(ValueError):
            convolutional_encode_7_5(bits)

def test_invalid_terminate_rejected():
    for terminate in (1, 0, None, "yes"):
        with pytest.raises(ValueError):
            convolutional_encode_7_5([1,0], terminate)
""",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def main() -> None:
    manifest_tasks = []
    total_tests = 0
    for task in TASKS:
        task_root = ROOT / "tasks" / task["task_id"]
        problem_path = task_root / "public/problem.md"
        metadata_path = task_root / "public/task.json"
        oracle_path = task_root / "private/oracle.py"
        test_path = task_root / "private/test_official.py"
        write(problem_path, task["problem"])
        write(oracle_path, task["oracle"])
        write(test_path, task["tests"])
        test_count = sum(
            line.startswith("def test_") for line in task["tests"].splitlines()
        )
        total_tests += test_count
        metadata = {
            "task_id": task["task_id"],
            "suite": "AeroComm-Small8",
            "split": "main-c-test",
            "domain": task["domain"],
            "scenario": task["scenario"],
            "entry_point": task["entry_point"],
            "outputs": [task["entry_point"]],
            "expected_nonblank_loc": task["expected_loc"],
            "dependencies": ["numpy", "pytest"],
            "source_ids": task["source_ids"],
            "official_test_count": test_count,
        }
        write(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2))
        manifest_tasks.append(
            {
                **metadata,
                "public_problem": str(problem_path.relative_to(ROOT)),
                "public_metadata": str(metadata_path.relative_to(ROOT)),
                "private_tests": str(test_path.relative_to(ROOT)),
                "private_oracle": str(oracle_path.relative_to(ROOT)),
                "public_problem_sha256": sha256_file(problem_path),
                "private_tests_sha256": sha256_file(test_path),
                "private_oracle_sha256": sha256_file(oracle_path),
            }
        )

    source_registry = {
        "frozen_at": "2026-07-29",
        "provenance_note": (
            "Tasks are newly authored small implementation problems grounded in "
            "the listed domain sources; they are not copied benchmark instances."
        ),
        "sources": SOURCES,
    }
    write(
        ROOT / "source_registry.json",
        json.dumps(source_registry, ensure_ascii=False, indent=2),
    )
    revision_payload = json.dumps(
        {"tasks": manifest_tasks, "sources": SOURCES},
        ensure_ascii=False,
        sort_keys=True,
    ).encode()
    dataset_revision = hashlib.sha256(revision_payload).hexdigest()
    manifest = {
        "dataset_name": "AeroComm-Small8",
        "dataset_version": "1.0.0",
        "task_count": len(manifest_tasks),
        "official_test_count": total_tests,
        "domains": {"aerospace": 4, "communications": 4},
        "source": {
            "kind": "source-grounded newly authored executable tasks",
            "dataset_revision": dataset_revision,
            "revision": dataset_revision,
            "frozen_at": "2026-07-29",
        },
        "protocol": {
            "scope": "single Python function per task",
            "attempts_per_task": 1,
            "hidden_test_feedback": False,
            "resolved_rule": "all hidden tests pass and module imports without leakage",
            "dependencies": "Python standard library plus NumPy; pytest only in evaluator",
        },
        "tasks": manifest_tasks,
        "artifacts": {
            "source_registry": "source_registry.json",
            "source_registry_sha256": sha256_file(ROOT / "source_registry.json"),
        },
    }
    write(
        ROOT / "dataset_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2),
    )
    print(
        json.dumps(
            {
                "dataset": manifest["dataset_name"],
                "tasks": manifest["task_count"],
                "tests": manifest["official_test_count"],
                "revision": dataset_revision,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

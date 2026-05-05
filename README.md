# 🔆 Taichi 光线追踪：从硬阴影到镜面反射


---

## 📌 实验目的

- **理论进阶**：区分光线投射（Direct Lighting）与光线追踪（全局光照）的本质差异。
- **核心技术**：掌握次级射线的两种典型应用 —— **Shadow Ray**（硬阴影）与 **Reflection Ray**（镜面反射）。
- **GPU 编程思维**：将传统递归光线追踪改造为 **迭代（循环）** 模式，适配 GPU 并行架构。

## 🖥️ 实验环境

| 组件        | 要求                                                         |
| ----------- | ------------------------------------------------------------ |
| 语言        | Python 3.8+                                                 |
| 核心库      | [Taichi](https://taichi-lang.org/) ≥ 1.7.0                  |
| 硬件        | 支持 Metal / CUDA / Vulkan 的 GPU（集成/独立均可）          |
| 开发工具    | 任意文本编辑器 + 终端 / VS Code / PyCharm                   |

## 🧠 实验原理

本项目采用 **经典 Whitted-Style 光线追踪模型**。对每一像素发射一条主光线，遇到物体时：

1. **阴影测试**：从交点向光源发射暗影射线，若途中被遮挡 → 仅保留环境光（硬阴影）。

2. **材质分支**：
   - **漫反射 (Diffuse)**：计算 Phong 直接光照（环境光 + 漫反射），**终止**该光线。
   - **镜面反射 (Mirror)**：根据反射定律生成新射线，继续传播，能量随反射率衰减。

---

### 反射向量公式：

$R = L_{in} - 2(L_{in} \cdot N)N$

---

**关键避坑**：射线起点需沿法线方向偏移 $\epsilon = 1 \times 10^{-4}$，否则会与自身表面相交，产生“阴影粉刺”（Shadow Acne）。

## ✨ 实现功能

### 功能模块

- [x] 隐式几何场景  
- [x] 硬阴影  
- [x] 镜面反射  
- [x] 迭代光线弹射  
- [x] 交互式 UI  
- [x] GPU 加速渲染  

---

### 说明

- 无限大棋盘格平面 + 漫反射红球 + 镜面银球，均在 Taichi kernel 中定义  
- 从光源方向发射暗影射线，实时产生清晰的阴影边缘  
- 支持多级反射（最大弹射数 1~5），银色球反射环境  
- 使用 `for` 循环模拟递归，避免 GPU 递归开销  
- 动态调整光源位置（X/Y/Z）与最大弹射次数，实时更新渲染  
- 利用 Taichi 自动并行，每帧独立计算光线

---

## 🏗️ 场景结构

- **Camera** `(0, 1, 5)`
  - **左侧球**：红色漫反射球 `(-1.2, 0, 0)`
  - **右侧球**：银色镜面球 `(1.2, 0, 0)`
  - **地面**：无限大棋盘格平面 `y = -1.0`（所有射线最终向下到达）

- **红色漫反射球**：材质 `MAT_DIFFUSE`，基色 `(0.8, 0.1, 0.1)`
- **银色镜面球**：材质 `MAT_MIRROR`，反射率 `0.8`
- **棋盘平面**：漫反射，材质颜色由 `floor(x*2) + floor(z*2)` 奇偶性决定

## 🎮 交互控制面板

运行程序后，右下角会弹出控制窗口：

- **Light X / Y / Z**：左右拖动 → 阴影实时移动  
- **Max Bounces**：调节 `1～5`  
  - `=1`：镜面球只显示黑色（无反射）  
  - `≥2`：镜面球开始反射红球/棋盘格  

> 提示：弹射次数越高，反射越丰富，但 GPU 计算量线性增加。

---

## 🧩 核心代码及说明

### 1️⃣ 迭代光线弹射（替代递归）

```python
for bounce in range(max_bounces[None]):
    t, N, obj_color, mat_id = scene_intersect(ro, rd)
    if t > 1e9:   # 未命中 → 累加背景色
        final_color += throughput * bg_color
        break

    if mat_id == MAT_MIRROR:
        # 更新射线为反射方向，继续循环
        ro = p + N * 1e-4
        rd = normalize(reflect(rd, N))
        throughput *= 0.8 * obj_color

    elif mat_id == MAT_DIFFUSE:
        # 直接光照 + 阴影检测
        L = normalize(light_pos - p)
        shadow_t, _, _, _ = scene_intersect(p + N*1e-4, L)
        in_shadow = shadow_t < (light_pos - p).norm()
        direct_light = ambient + (0.8 * diff * obj_color if not in_shadow else 0)
        final_color += throughput * direct_light
        break   # 漫反射终止
```

### 2️⃣ 阴影痤疮的修正（关键偏移）

```python
shadow_ray_orig = p + N * 1e-4   # ✨ 法线方向微小偏移
shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)
```

### 3️⃣ 棋盘纹理平面求交

```python
@ti.func
def intersect_plane(ro, rd, plane_y):
    if ti.abs(rd.y) > 1e-5:
        t = (plane_y - ro.y) / rd.y
        if t > 0:
            # 计算交点并依据 x,z 坐标生成棋盘色
            p = ro + rd * t
            ix = ti.floor(p.x * 2.0)
            iz = ti.floor(p.z * 2.0)
            color = ti.Vector([0.3,0.3,0.3]) if (ix+iz)%2==0 else ti.Vector([0.8,0.8,0.8])
            return t, ti.Vector([0,1,0]), color, MAT_DIFFUSE
    return -1.0, ti.Vector([0,0,0]), ti.Vector([0,0,0]), MAT_DIFFUSE
```

---

## 实验效果

![Taichi光线追踪演示](https://github.com/mzthynx/work4/raw/master/work4.gif)

| 弹射次数 = 1 | 弹射次数 = 3 |
|---|---|
| 镜面球无反射，呈黑色 | 镜面球清晰反射红球和棋盘格 |
| 阴影硬边，光源移动效果明显 | 反射链完整，符合 Whitted 模型预期 |


---

## 实验总结

通过本项目，我们成功实现了 **基于 GPU 的 Whitted 风格光线追踪**，并解决了以下关键问题：

- **递归 → 迭代**：用 `for` 循环 + `throughput` 变量安全地在 GPU 上表达递归光线弹射。
- **阴影精度**：严格使用法线偏移 `1e-4`，彻底消除自相交伪影。
- **交互实时性**：`taichi.ui` 滑动条修改参数后 `render()` 即时重绘，展示阴影与反射的动态变化。
- **材质系统扩展性**：通过 `MAT_DIFFUSE` / `MAT_MIRROR` 枚举，未来可轻松添加折射、粗糙反射等材质。

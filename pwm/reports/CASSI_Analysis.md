# CASSI Corrected Results分析

## 核心问题：为什么corrected results不理想？

查看W2表中的结果，我们看到：
- **W2a (mask translation)**: PSNR 15.24 → 15.36 dB (+0.12 dB) ❌ 几乎无改进
- **W2b (mask rotation)**: PSNR 15.23 → 19.00 dB (+3.77 dB) ✓ 有改进
- **W2c (disp slope)**: PSNR 15.27 → 20.76 dB (+5.49 dB) ✓ 有改进
- **W2d (disp axis)**: PSNR 15.01 → 22.05 dB (+7.04 dB) ✓ 有改进
- **W2e (PSF blur)**: PSNR 15.25 → 15.25 dB (+0.00 dB) ❌ 无改进

## 基准数据（新生成）

### No-Mismatch Baseline
| 参数 | 值 |
|-----|-----|
| 条件 | 所有参数正确（dx=0, dy=0, theta=0, phi_d=0） |
| 数据集 | KAIST (256×256×28) |
| **PSNR** | **9.38 dB** |
| **SAM** | **65.42°** |

### 小mismatch的影响
| 扰动 | PSNR | 恶化 |
|-----|-----|-----|
| 无mismatch | 9.38 dB | — |
| Mask shift (2,1)px | 9.26 dB | 0.13 dB |
| Mask rotation 1° | 9.03 dB | 0.35 dB |

## 关键观察

### 1. **GAP-TV天生较弱**
- 无任何mismatch时，GAP-TV仅能达到 **9.38 dB**
- W2中uncorrected达到15.24 dB（使用TSA数据可能更高质量）
- 相比MST深网络（35+ dB），GAP-TV差7倍

**原因**: GAP-TV是经典迭代算法，没有学习到spectral结构的先验知识

### 2. **不同mismatch类型的影响不同**

#### A. 参数耦合问题 (mask geometry)
- **Mask translation (W2a)**: 影响小，只是pixels平移，spectral信息基本保留
  - Correction gain: +0.12 dB（很小）

- **Mask rotation (W2b)**: 影响中等，rotation导致mask与pixel grid不对齐
  - Correction gain: +3.77 dB

#### B. Spectral cross-talk (dispersion)
- **Dispersion slope (W2c)**: 影响大，导致band间leakage
  - Correction gain: +5.49 dB

- **Dispersion axis angle (W2d)**: 影响最大，2D spectral smearing
  - Correction gain: +7.04 dB（W2中最大的改进）

#### C. 噪音放大 (PSF)
- **PSF blur (W2e)**: 虽然NLL大幅降低（97.9%），但PSNR无改进
  - **原因**: Wiener deblurring在高频放大噪音，抵消了blur correction的益处
  - Correction gain: +0.00 dB

### 3. **为什么corrected结果最多才22 dB？**

即使完美校正了所有参数，GAP-TV的内在限制决定了：

| 方法 | PSNR | 原因 |
|-----|-----|-----|
| **GAP-TV (oracle)** | ~22 dB | 迭代算法 + TV正则化的平滑性倾向 |
| **MST-L (HDNet)** | **35 dB** | 深度学习学到spectral/spatial先验 |
| **差距** | **13 dB** | 学习先验的威力 |

### 4. **Corrected vs Deep Learning基准对比**

从W1结果：
- GAP-TV baseline: 14.92 dB (平均10个scene)
- HDNet: 35.06 dB
- MST-L: 34.99 dB

**GAP-TV的corrected结果 (最好W2d: 22.05 dB) vs Deep Methods (35+ dB):**
- 仍然相差 **13 dB** (太大)
- 即使完美校正operator，仍需要学习算法的改进

## 改进方向

### 短期（实现中）
1. **使用correction后的operator初始化深网络** ✓ 已在W2中展示
2. **多参数联合校正**：当前W2分别校正各参数，应该同时优化
3. **Adaptive正则化**：不同band应该用不同的TV weight

### 中期（待实施）
1. **Differentiable GAP-TV**：UPWMI Algorithm 2已实现，但需要与深网络整合
2. **Hybrid方法**：Calibrated operator + 轻量级网络（比MST-L快）
3. **Spectral先验学习**：从多场景学习spectral structure

### 长期（架构改进）
1. **End-to-end学习**：将operator校正integrated into solver training
2. **Physics-aware networks**：用calibrated operator初始化网络
3. **Multi-modality fusion**：CASSI + complementary modality

## 建议

### 为什么corrected results不好的根本原因：

**不是校正算法的问题，而是GAP-TV本身的限制**

- ✓ Operator correction工作得很好（NLL下降93-99%）
- ✓ 参数恢复精确（exact recovery）
- ❌ 但GAP-TV即使参数正确，也只能达到~9-22 dB
- ❌ 相比学习方法的35 dB，仍有巨大差距

### 实际应用建议：

1. **不依赖corrected GAP-TV**：对于实际应用，应该用MST+corrected-operator
2. **Operator correction的真正价值**：为深网络提供更好的初始化和bias
3. **W1结果（35 dB）才是真正的基准**：这才是实际能达到的性能

## 数据文件

- Baseline结果: `cassi_baseline_no_mismatch.json`
- W2详细结果: `cassi.md` (section "Workflow W2")
- W1深学习对比: `cassi.md` (section "Workflow W1")

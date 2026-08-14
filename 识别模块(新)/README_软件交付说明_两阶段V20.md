# 射频干扰两阶段识别模块（软件交付说明）

## 1. 模块做什么

输入为复数 IQ 数据（推荐 `.npz`）。算法严格分为两个阶段：

1. **阶段 1：V13 开集识别**：只根据已知类模型判断 IQ 块是否属于已知分布；输出“已知 / 未知”。未知块会被拒识，并可在离线测试时做未知簇聚类。
2. **阶段 2：V20 闭集四分类**：只读取阶段 1 判为“已知”的 IQ 块，在 `2ASK、2FSK、BPSK、16QAM` 四类中进一步分类。

因此，AM、FM 等目标未知类别**不参与 V13/V20 训练和阈值选择**。测试数据中的 `labels` 仅用于离线计算指标与混淆矩阵，不会被算法读取来作判断。

> V20 的判决窗口为连续 **32 个**阶段 1 接受块；每个块 16384 点，即每次四分类判决使用 524288 个 IQ 点。软件实时调用时应累计到 32 个接受块后再显示一个 V20 分类结果。

## 2. 软件方应保留的文件

请整体复制下列结构，不要只复制单个 `.py` 文件：

```text
识别模块提供的数据集模型及代码/
├─ model/
│  ├─ v13_open_set_model.npz                 # 第一阶段开集模型
│  └─ v20_closed_set_long_window_model.joblib # 第二阶段四分类模型
├─ recognition_core/                          # 两阶段算法核心（必须整体保留）
├─ 01_extract_open_set_known_npz.py           # 第一步：开集筛选并导出新 IQ 数据集
├─ 02_run_v20_closed_set_from_extracted_npz.py # 第二步：读取导出的数据集做四分类
├─ evaluate_recognition_npz.py                 # 第一阶段完整离线评估：指标、ROC、未知簇图
├─ evaluate_two_stage_v20_npz.py               # 一条命令完成两阶段正式离线评估
├─ demo_software_api_v20.py                    # Python / PyQt 调用示例
├─ requirements.txt                            # 运行依赖
└─ README_软件交付说明_两阶段V20.md             # 本说明
```

可选随交付复制：

```text
test_data/                                    # 教师演示或联调所用 NPZ
reference_results/v20_two_stage/              # 当前正式结果和示例图
```

不需要交付给软件端：训练用 HDF5 数据集、`artifacts*` 历史训练目录、旧版本实验脚本、`quick_stage_pipeline` 临时检查输出。

**请勿从 `recognition_core/` 中单独删文件。** 当前 V13 模型的实际依赖链为
`core.py → v6_nonlinear.py → v5_strict_center.py`，其 IQ 特征由
`v7_cumulant.py` 提取；第二阶段 V20 则依赖
`v14_matched_features.py`、`v16_cyclic_features.py`、`v18_bandlimited_features.py`
和 `v20_stage2.py`。因此，`v5`、`v6`、`v7` 虽是早期编号，仍是当前模型运行所必需的底层实现。

## 3. 输入 NPZ 格式

正式调用最少只需要：

```python
np.savez_compressed("input_iq.npz", iq=iq)
```

其中 `iq` 为 `float32` 数组，形状为：

```text
[块数, 2, 16384]
```

第 2 维依次为 I、Q。对于**离线指标测试**，可额外提供：

```text
labels       int64，长度等于块数
class_names  例如 ["2ASK", "2FSK", "BPSK", "16QAM", "AM", "FM"]
```

标签可用于绘制 V20 混淆矩阵和计算准确率；它们不影响任何推理结果。

## 4. 推荐的软件批处理流程

在本目录打开终端，先安装依赖：

```powershell
python -m pip install -r requirements.txt
```

### 第一步：开集识别并自动生成“已接受 IQ 数据集”

```powershell
python 01_extract_open_set_known_npz.py `
  --input "D:\\你的路径\\input_iq.npz" `
  --output-dir "D:\\你的路径\\recognition_output" `
  --accepted-name "stage1_accepted_known_iq.npz" `
  --cluster-unknown
```

最常改的三个位置已经在脚本 `parse_args()` 中用 `Software integration` 注释标出：

- `--input`：用户导入的原始 IQ NPZ；
- `--output-dir`：生成结果的目录；
- `--accepted-name`：新数据集名称。默认 `stage1_accepted_known_iq.npz`，可以按软件需求改名。

此步生成：

```text
stage1_accepted_known_iq.npz       # 第二阶段唯一需要读取的新数据集
stage1_rejected_unknown_iq.npz     # 被第一阶段拒识的未知候选块
stage1_open_set_records.csv        # 每个块的已知/未知、得分、未知簇编号
stage1_open_set_summary.json        # 块数和所有输出文件位置
```

### 第二步：对第一阶段接受数据做闭集四分类

```powershell
python 02_run_v20_closed_set_from_extracted_npz.py `
  --input "D:\\你的路径\\recognition_output\\stage1_accepted_known_iq.npz" `
  --output-dir "D:\\你的路径\\recognition_output" `
  --result-name "stage2_v20_result.json"
```

该脚本的 `--input`、`--output-dir`、`--result-name` 同样可直接改为软件实际路径与名称。它输出：

```text
stage2_v20_result.json             # 四分类数量、窗口长度、若有标签则含准确率
stage2_v20_group_records.csv       # 每个 32 块窗口的预测类别
stage2_v20_confusion_matrix.png    # 仅带标签测试集可生成
```

## 5. 一键正式测试（教师演示）

第一阶段指标、开集混淆矩阵、ROC 图、未知簇特征空间图和未知簇指标：

```powershell
python evaluate_recognition_npz.py
```

两阶段串联的 V20 四分类矩阵与汇总：

```powershell
python evaluate_two_stage_v20_npz.py
```

当前基准结果（固定正式测试集）：

| 指标 | 当前结果 |
|---|---:|
| 阶段 1 已知接受率 | 94.65% |
| 阶段 1 未知拒识率 | 92.30% |
| 阶段 1 AUROC | 0.98045 |
| 未知簇 NMI | 0.98270 |
| 未知簇准确率 | 99.84% |
| 阶段 1 → V20 四类闭集准确率 | 92.86% |

上述 V20 闭集准确率以“第一阶段接受且真实类别单一的 32 块窗口”为统计单位。

## 6. Python / PyQt 直接接口

软件不是必须调用两个命令行脚本，也可参考：

```powershell
python demo_software_api_v20.py
```

关键调用是：

```python
from recognition_core import TwoStageV20RecognitionService

service = TwoStageV20RecognitionService(
    "model/v13_open_set_model.npz",
    "model/v20_closed_set_long_window_model.joblib",
)
result = service.predict_array(iq_blocks, cluster_unknown=True)
```

`result["stage1"]` 返回每块已知/未知、得分和未知簇；`result["stage2"]` 返回已知接受块每 32 块形成一个窗口后的四类分类结果。UI 可据此显示“已知/未知判断、已知接受率、未知拒识率、簇编号、V20 四分类结果”。

## 7. 结果图建议

演示页面可从以下三张图中选择一至两张：

1. `open_set_confusion_matrix.png`：第一阶段开集“已知 / 未知”判断结果；
2. `unknown_cluster_feature_space.png`：第一阶段拒识后的未知样本形成两个簇的特征空间图；
3. `stage2_v20_confusion_matrix.png`：第二阶段四类闭集识别结果。

`recognition_summary.json` 中还会保存未知簇 NMI、未知簇准确率等数值指标。
特征空间图中 AM/FM 真值颜色只用于带标签的离线测试展示，模型聚类时不读取这些标签。
软件实时界面可按需要只显示一张图和右侧关键指标。

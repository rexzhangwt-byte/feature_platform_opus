# Feature Mining Platform · 特征挖掘平台

一个基于 Python (FastAPI + PyTorch + scikit-learn) 的端到端特征挖掘 / 模型训练平台。提供 8 大模块的 Web 可视化配置，每个模块都可以**新建命名的配置**，并被新建训练任务时自由复用、组合成训推 pipeline。

## ✨ 特性总览（对应需求 8 大模块）

| 模块 | 能力 |
|---|---|
| ① 数据集配置 | 本地 json/pkl/npy 上传 · SSH 远端 SFTP 拉取（密码 / 私钥）· 结构可视化（key 数量、value 类型、向量长度、数值统计） |
| ② 特征计算 | 直接将数据集作为特征集；或用**画布拖拽**搭建计算图 — `standardize / concat / add / sub / mul / div / mean / aggregate_* / filter_keys / scale / log1p / abs`；每个 `output` 节点归档为一个新特征集（可下载 JSON）。**画布提供完整的拖拽使用说明** |
| ③ 标签集导入 | 上传时自动新增配置 + 显示标签分布（计数与比例） |
| ④ 特征选择 | 过滤策略（值过滤 / 方差过滤 / 缺失剔除 / Top-K 方差）+ 重要性策略（uniform / variance / manual 权重）+ 实时预览 |
| ⑤ 特征分布可视化 | 单 key 时间曲线（如 48 维工作日/周末曲线）+ 全量分布直方图 + 列均值/标准差曲线 |
| ⑥ 模型算法选型 | random_forest · decision_tree · logistic_regression · linear_regression · knn · naive_bayes · kmeans · **autoencoder_torch** · **mlp_torch**；每个算法都有参数 schema 和**参数范围**字段，供寻优搜索使用 |
| ⑦ 寻优方法 | 三种策略：`grid`（无反馈朴素搜索，可指定参数优先级）/ `random` / `bayes`（基于 scikit-optimize，可配置 `n_initial_points` / `acq_func` / 核函数等） |
| ⑧ 训练 Pipeline | 可串联多模型节点（如 **AE → MLP**）；多特征集**自动取样本交集**并支持 `concat / sum / mean` 组装为 X；CPU/GPU 设备选择；自动早停与每次迭代理由记录 |
| ⑨ 结果归档 & 评估 | 最优权重（`.pt`/`.pkl`）+ 最优参数 + 训练 / 验证混淆矩阵 + F1 / Accuracy / Precision / Recall + 验证集 **TP / FP / FN / TN keys（json）** + HPO 全部试验记录 + AE 训练记录 + 迭代收敛理由日志 + 特征重要性 / 聚类结果 |

## 🗂️ 目录结构
```
webapp/
├── backend/
│   ├── app.py              # FastAPI 入口（52 个 REST 端点）
│   ├── data_io.py          # JSON / PKL / NPY 加载与结构摘要
│   ├── feature_compute.py  # 计算图执行引擎（拓扑排序 + 14 种算子）
│   ├── feature_select.py   # 过滤 / 权重策略
│   ├── models.py           # 9 种算法注册（含 PyTorch AE & MLP）
│   ├── hpo.py              # grid / random / bayes 搜索
│   ├── training.py         # Pipeline 编排 + 评估 + 归档
│   ├── ssh_loader.py       # paramiko SFTP 拉取
│   └── storage_utils.py    # JSON 元数据 + 物件存储
├── frontend/               # 单页 SPA（vanilla JS + Chart.js）
│   ├── index.html          # 9 个 Tab 页 + 拖拽画布
│   ├── style.css
│   └── app.js              # 调用所有 REST 端点 + 画布交互
├── scripts/
│   ├── gen_mall_data.py    # 生成商场分类合成数据
│   └── e2e_mall_test.py    # 端到端测试脚本
└── README.md
```

## 🚀 快速启动

```bash
cd /home/user/webapp
pip install fastapi "uvicorn[standard]" python-multipart paramiko \
            scikit-learn scikit-optimize numpy pandas
pip install torch --index-url https://download.pytorch.org/whl/cpu

python -m uvicorn backend.app:app --host 0.0.0.0 --port 8765
```
然后浏览器打开 `http://localhost:8765/`。

## 🧪 端到端「商场分类」任务验证

平台**通过新建 pipeline 的方式**可以支持需求中描述的商场分类任务（也可支持其它任意自定义任务名）。下面这个端到端脚本会按需求中的特征结构（人数变化 + 驻留时长，每个 key 对应 48 维向量；标签 `商场`/`非商场`）跑完整流程：

```bash
python scripts/gen_mall_data.py        # 生成合成数据 (mall + non-mall 各 80)
python scripts/e2e_mall_test.py        # 走通 9 个模块
```

测试覆盖：
1. 上传两个数据集（people_count, dwell_time）
2. 提升为特征集；通过特征计算图做 `standardize → concat(原始, 标准化) → filter_keys(by label keys) → output`
3. 上传标签集
4. 创建特征选择配置（缺失剔除 + 方差过滤）
5. 创建 AE 配置（latent=16）+ MLP 配置
6. 创建 random search HPO 配置（搜索 hidden_dim / lr）
7. 创建 pipeline：`AE → MLP`，2 个特征集做 concat 组装，CPU 设备
8. 运行 + 轮询日志（含每个迭代的收敛理由）
9. 校验归档：混淆矩阵 / F1 / TP-FP-FN-TN keys / HPO 历史

实测结果（合成数据，5 epochs 早停）：
```
Confusion matrix: [[16, 0], [0, 16]]
accuracy=1.000  f1_macro=1.000
TP: 16   FP: 0   FN: 0   TN: 16
```
**「Best params / Best weights / 验证集 TP/FP/FN/TN keys」全部归档可下载。**

## 🎨 画布拖拽使用说明（特征计算）

> 这部分在前端 ② 特征计算 Tab 内的 **📖 画布拖拽计算使用说明** 折叠区可查看完整版本。

1. 左侧 **节点库** 点击 `+ 数据集/特征集` 添加输入节点；从下拉框选择已上传的数据集 / 特征集。
2. 点击 `+ 计算节点`，下拉选择数学算子（`standardize` / `concat` / `add` / ...）。
3. 节点左侧 ● 是输入端口、右侧 ● 是输出端口。**按住右端口拖动**到下游节点的左端口即可连线。
4. 双击「参数 ⚙」按钮，用 JSON 设置该算子的参数（例：`filter_keys` 的 `allow_keys` 或者 `allow_from_ref` 引用某标签集）。
5. 添加 `+ 输出节点` 标记你希望归档的结果。每个 output 节点 = 一个新特征集。
6. 点击 `保存为配置`，给配置命名后即可在右侧列表「运行」生成新特征集（在「② 特征计算」末尾下载）。
7. 右键节点删除；右上角 ＋／－／⤢ 缩放画布。
8. 商场分类示例图：`input(人数) → standardize`；`input(人数) → concat`；`standardize → concat`；`concat → filter_keys(by label keys) → output`。

## 🔌 REST API 速览

| 模块 | 主要端点 |
|---|---|
| Datasets | `POST /datasets/upload` · `POST /datasets/ssh` · `GET /datasets[/{id}/structure]` |
| Features | `POST /features/from_dataset` · `GET /features` · `GET /features/{id}/download` |
| Compute  | `GET /feature_ops` · `POST /feature_compute` · `POST /feature_compute/{id}/run` |
| Labels   | `POST /labels/upload` · `GET /labels/{id}/distribution` |
| Select   | `POST /feature_select` · `POST /feature_select/{id}/preview` |
| Viz      | `POST /viz/single` · `POST /viz/distribution` |
| Models   | `GET /algos` · `POST /models` |
| HPO      | `POST /hpo` |
| Pipelines| `POST /pipelines` · `POST /pipelines/{id}/run` · `GET /pipelines/{id}/run_status` |
| Archives | `GET /archives` · `GET /archives/{id}` · `GET /archives/{id}/file/{name}` |

API 文档自动暴露在 `/docs` (Swagger UI)。

## 💡 设计要点

- **样本交集**：训练 pipeline 选择多个特征集时，自动 `set.intersection(...)` 所有 key（再与标签 keys 取交集），保证每个样本在所有视图都齐全。
- **AE→MLP 串联**：`model_chain` 列表的前序模型若为 `autoencoder_torch`，先训练 AE 把特征压缩到 latent，再把 latent 喂给最后一个分类器（MLP / RF / ...），完美对应商场任务的「先 AE 降维 32 维，再 MLP 分类」需求。
- **每次迭代理由**：训练日志 `iteration_log.json` 记录每个事件（含时间戳和原因，例如 `HPO trial 2: params=... score=0.5 (random sample 2/3)` 或 `AE early stop at epoch 9 (best loss ...)`），用于人工过程校验。
- **TP/FP/FN/TN 导出**：默认以 `商场`、`mall`、`1`、`True` 等作为 positive class 的启发式选择；归档为 `tp_fp_fn_tn.json`，含各类样本 keys 列表，便于人工抽样校验预测结果。
- **可扩展**：新增算法 = 在 `backend/models.py` 的 `MODEL_SPECS` 中注册一个 spec（`param_schema` + `factory`）；新增特征算子 = 在 `backend/feature_compute.py` 添加一个 `op_xxx` 并在 `execute_graph` 分支注册。

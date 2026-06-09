# WheatCASH
WheatCASH：融合双重注意力机制与沙漏网络的小麦结构变异精准预测
# WheatCASH: Accurate Prediction of Structural Variations in Wheat Fusing Dual-Attention Mechanisms and Stacked Hourglass Networks

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.10-orange.svg)](https://pytorch.org/)

WheatCASH 是一种专为面包小麦等大基因组、高重复、多倍体植物背景设计的深度学习结构变异（SV）检测工具。该工具面向 PacBio CCS (HiFi) 长读长测序数据，通过将一维序列比对信息转化为二维多通道图像特征，并采用改进的**融合沙漏网络（RF-HGNet）**预测断点置信图，实现了缺失（DEL）、倒位（INV）和重复（DUP）等结构变异的精准预测。

---

## 核心特性

- **多通道图像特征编码**：整合分割比对读段（SM）、读段深度（RD）、拆分读段（SR）与剪切读段（CR）四类核心信号，构建 2D 空间结构表征。
- **先进的 RF-HGNet 架构**：在经典堆叠沙漏网络基础上，引入通道-空间双注意力机制（CBAM）、系统化 Dropout 正则化，并改进了特征融合模块（RF-FFM）。
- **高保真特征融合**：抛弃传统逐元素相加，采用通道维度特征拼接（Concat）辅以双线性上采样，大幅提升边界断点细节还原度。
- **动态学习率调度**：引入余弦退火学习率调度策略（Cosine Annealing LR），确保训练高效稳定收敛。

---

## 环境依赖

### 1. 生物信息学工具（确保已加入系统环境变量）
- `Minimap2` (v2.24+)
- `Samtools` (v1.21+)
- `BCFtools`
- `SURVIVOR`
- `PBSIM2`
- `Truvari`

### 2. Python 依赖项
- Python >= 3.8
- PyTorch >= 1.10 (支持 CUDA)
- OpenCV-Python, NumPy, Pandas, Pysam, SciPy

---

## 📂 示例数据获取与复现 (Example Data Setup)

本项目提供独立的验证集染色体（以 `Chr6D` 为例）供用户进行跑通测试与泛化性验证。请激活环境并执行以下命令构建或迁移您的示例测试数据：

# 激活测试环境
source ~/.bashrc
conda activate WheatCASH_env

# 创建独立验证集的标准示例数据目录
mkdir -p ./example_data/Chr6D_test/

# 1. 将您自主模拟生成的示例结构变异真值 VCF 文件移动至工作目录
cp /path/to/your/simulated_data/Chr6D_simulated.vcf ./example_data/Chr6D_test/

# 2. 将模型在验证集上生成的预测 VCF 结果文件迁移至该目录，准备后续的 Truvari 基准评估
cp /path/to/your/predict_reports/svs.vcf ./example_data/Chr6D_test/

快速上手与通用上机操作流程
1. 数据预处理与比对流水线1.1 为参考基因组构建索引Bash# 构建 Minimap2 索引
minimap2 -x map-hifi -d <REF_INDEX>.mmi <REF_GENOME>.fa

# 构建 Samtools 索引，用于快速随机访问参考基因组
samtools faidx <REF_GENOME>.fa
1.2 使用预构建索引文件进行序列比对 (推荐，速度更快)Bash# 提交比对任务至后台运行
nohup minimap2 -ax map-hifi -t 128 <REF_INDEX>.mmi <SAMPLE_READS>.fastq > <SAMPLE>.sam 2> minimap2_progress.log &

# 查看比对进程运行状态
ps -ef | grep minimap2
1.3 格式转换、排序与构建 CSI 索引Bash# 1. SAM 转 BAM (多线程压缩)
nohup samtools view -@ 128 -b -o <SAMPLE>.bam <SAMPLE>.sam 2> samtools_view.log &

# 2. 按基因组坐标升序排序
nohup samtools sort -@ 128 -T <TMP_PREFIX> -o <SAMPLE>.sorted.bam <SAMPLE>.bam 2> samtools_sort.log &

# 3. 为排序后的 BAM 创建 CSI 索引（-c 参数更适合小麦等大基因组）
nohup samtools index -c <SAMPLE>.sorted.bam 2> samtools_index.log &

# 4. 检查 samtools 转换与排序后台任务的整体进度
ps -ef | grep samtools
💡 高阶单行操作示例（如配合 384 高线程调度）：samtools view -@ 384 -b -o <SAMPLE>.bam <SAMPLE>.sam && samtools sort -@ 384 -T <TMP_PREFIX> -o <SAMPLE>.sorted.bam <SAMPLE>.bam && samtools index -c <SAMPLE>.sorted.bam2. 多通道图像特征生成 (Feature Image Generation)本阶段将滑窗扫描染色体，提取特征信号并渲染为 256×256 像素的多通道特征图像。
2.1 特征生成配置文件 (data_config.yaml) 标准规范在运行特征生成脚本前，需修改对应的 YAML 配置文件。请根据您的本地实际路径替换包含 /path/to/... 的配置项：
YAML
#### REQUIRED ####
bam: "/path/to/your/work_dir/<SAMPLE>.sorted.bam"       # 排序后的比对 BAM 文件路径
fai: "/path/to/your/work_dir/<REF_GENOME>.fa.fai"       # 参考基因组 FAI 索引路径
bed: "/path/to/your/work_dir/<BASELINE_TRUTH>.vcf"      # 结构变异真值文件 (VCF/BED)

#### OPTIONAL ####
n_cpus: 40                                # 使用的 CPU 核心数
chr_names: ["<CHROMOSOME_NAME>"]           # 指定分析的染色体名称 (如 ["Chr6D"])
logging_level: "INFO"                     # 日志级别
store_img: True                           # 是否存储渲染的热图
allow_empty: False                        # 是否允许保存无变异区域
scan_target_intervals: False
stream: False
bins_per_block: 8000                      # 每块包含的 bin 数量
min_pair_support: 2                       # 不协调读段对的最小支持数
min_pair_distance: 4000                   # 读对支持筛选的最小距离阈值
max_pair_distance: 1000000                # 读对支持筛选的最大距离阈值

#### FIXED PARAMS (依据论文设定，请勿修改) ####
bam_type: "LONG"
signal_set: "LONG"
signal_set_origin: "LONG"
blacklist_bed: null

# 差异化特征通道饱和上限与质量过滤阈值 (MAPQ)
signal_vmax: {"RD": 600, "RD_LOW": 800, "RD_CLIPPED": 600, "SM": 200, "SR_RP": 600, "LR": 600, "LLRR": 100, "RL": 100, "LLRR_VS_LR": 1}
signal_mapq: {"RD": 20, "RD_LOW": 0, "RD_CLIPPED": 20, "SM": 20, "SR_RP": 0, "LR": 0, "LLRR": 1, "RL": 1, "LLRR_VS_LR": 1}

### Interval Configuration (滑动窗口配置) ###
bin_size: 750                             # 分辨率汇总分箱大小 (750 bp)
interval_size: 150000                     # 滑动窗口覆盖区域大小 (150 kbp)
step_size: 50000                          # 滑动窗口步长 (50 kbp，即相邻窗口重叠 100 kbp)
shift_size: [0]

### Image Generation (图像生成配置) ###
heatmap_dim: 1000                         # 初始热图渲染宽度 (1000 像素)
image_dim: 256                            # 最终降采样输入网络尺寸 (256x256 像素)
class_set: "BASIC4"                       # 变异分类集合 (DEL, DUP, INV, NEG)
num_keypoints: 1                          # 每个变异标注的关键点数量
bbox_padding: 0
empty_annotation: False                   # 核心控制：生成已标注图片设为 False；生成未标注图片设为 True

2.2 上机特征生成命令与后台管理
# 1. 刷新系统环境变量并激活 Conda 环境
source ~/.bashrc
conda activate WheatCASH_env

# 2. 进入对应的特征图像本地存放目录
cd /path/to/your/work_dir/<OUTPUT_IMAGE_DIR>

# 3. 后台挂起运行特征生成脚本
nohup python /path/to/engine/generate.py --config /path/to/your/work_dir/data_config.yaml 2> generate.log &

# 4. 检查脚本进程运行状态与日志监控
ps -ef | grep generate.py
head -n 1000000000 generate.log
head -n 1000000000 nohup.out

# 5. 统计产出的有效数据集（图像与标注文档数）
find annotated_images -type f | wc -l
find annotations -type f | wc -l

# 6. 数据压缩归档与跨服务器迁移
tar -czvf annotations.tar.gz annotations
tar -czvf annotated_images.tar.gz annotated_images
tar -czvf images.tar.gz images

# (解压命令备忘) tar -xzf annotations.tar.gz

3. 模型训练阶段 (Model Training)
3.1 编写训练配置文件 (train.yaml)
YAML
#### REQUIRED ####
dataset_dirs: ["/path/to/your/work_dir/<TRAIN_DATASET_DIR>"]  # 训练数据集根目录（包含图像与标注）
num_epochs: 32                                  # 训练总轮数 (Epochs)

#### OPTIONAL ####
batch_size: 4                                   # 每个 Batch 的图像数量
gpu_ids: [0]                                    # 指定训练使用的 GPU 设备 ID
n_cpus: 32                                      # 训练数据加载使用的 CPU 核心数
bam_type: "LONG"
signal_set: "LONG"
signal_set_origin: "LONG"
pretrained_model: null                          # 预训练模型路径
logging_level: "INFO"                           # 日志级别
report_interval: 50                             # 每隔多少个 Batch 汇报一次训练状态
model_checkpoint_interval: 10000                # 保存模型权重的 Batch 间隔
validation_ratio: 0.1                           # 验证集比例（默认切分 10% 的数据用于验证）
plot_confidence_maps: False                     # 是否输出预测置信度图
learning_rate: 0.0001                           # 初始学习率
learning_rate_decay_interval: 5                 # 学习率衰减的 Epoch 间隔
learning_rate_decay_factor: 1                   # 学习率衰减系数
sigma: 10                                       # 二维高斯分布峰值的扩散范围核
stride: 4                                       # 下采样步长超参数 (控制 256 到 64 像素的映射)
heatmap_peak_threshold: 0.4                     # 过滤置信度阈值
class_set: "BASIC4"

3.2 运行训练与数据集切分检查
# 1. 激活环境并挂起后台训练任务
source ~/.bashrc
conda activate WheatCASH_env
nohup python /path/to/engine/train.py --config /path/to/your/work_dir/train.yaml > train.log 2>&1 &

# 2. 进程状态与 Loss 日志追踪
ps aux | grep "engine/train.py" | grep -v grep
head -n 1000000000 train.log

# 3. 切换到训练根目录检查数据分布与交叉验证切分 (splits)
cd /path/to/your/work_dir/<TRAIN_DATASET_DIR>
find annotated_images -type f | wc -l
find annotations -type f | wc -l
find images -type f | wc -l

find split0 -type f | wc -l              # 检查切分后的训练/验证子集 0
find split1 -type f | wc -l              # 检查切分后的训练/验证子集 1

4. 模型性能评估阶段 (Model Evaluation & Benchmarking)模型调用并输出预测 VCF 文件后，使用 BCFtools 与 Truvari 评估指标。
4.1 变异结果排序、压缩与建立索引

# 1. 使用 bcftools 对模型预测的 VCF 按基因组坐标排序
bcftools sort -O v -o svs.sorted.vcf svs.vcf

# 2. 合并单行命令：一键完成对预测 VCF 的 bgzip 压缩与 tabix 索引
bgzip -c svs.sorted.vcf > svs.sorted.vcf.gz && tabix -p vcf svs.sorted.vcf.gz

# 3. 对模拟的基准真值 (Baseline) VCF 进行压缩与索引
bgzip -c <BASELINE_TRUTH>.vcf > <BASELINE_TRUTH>.vcf.gz && tabix -p vcf <BASELINE_TRUTH>.vcf.gz

4.2 运行 Truvari Bench 进行定量性能比对
依据论文设定，过滤尺寸 $\ge 5000\text{ bp}$ 的中大尺度变异，比对允许的断点最大匹配距离设为 1000 bp：
Bashtruvari bench \
    -b /path/to/your/work_dir/<BASELINE_TRUTH>.vcf.gz \
    -c /path/to/your/work_dir/svs.sorted.vcf.gz \
    -o /path/to/your/work_dir/<OUTPUT_DIR> \
    -f /path/to/your/work_dir/<REF_GENOME>.fa \
    --sizemin 5000 \
    --sizefilt 5000 \
    -r 1000

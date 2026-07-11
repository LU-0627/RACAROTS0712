# RDCAROTS GPU服务器部署关键修复清单

## 已修复的核心问题

### 1. Device逻辑重构 ✅
**问题**: 代码自动检测CUDA并降级CPU
**修复**: 
- RDCAROTS.__init__要求显式device参数
- 移除`torch.cuda.is_available()`自动选择
- device=None时抛出错误而非默认CPU
- 所有子模块接收显式device

### 2. IO Schema初始化 ✅
**问题**: self.io_schema未初始化就在forward中使用
**修复**:
- 在__init__中加载io_schema
- 将input_indices/output_indices转为device上的tensor
- 支持非连续索引
- forward中使用self.input_indices/self.output_indices

### 3. 合成数据生成器 (待完成)
**需要**: 使用真实状态空间递推
**当前**: 简化版C@B@u

### 4. RDCAROTSScorer接入 (待完成)
**需要**: frozen/guarded_online输出4分量
**当前**: 测试流程未完整接入

### 5. GPU服务器脚本 (待完成)
**需要**: 所有脚本默认cuda:0，无网络依赖
**当前**: 部分脚本存在但未完全GPU化

## 待修复项（按优先级）

### P0: 核心功能
1. [ ] 完成状态空间递推的合成数据生成
2. [ ] RDCAROTSScorer完整接入测试流程
3. [ ] GPU服务器脚本重构（所有脚本）
4. [ ] checkpoint完整保存恢复

### P1: 测试和文档
5. [ ] 补齐24个GPU测试文件
6. [ ] 创建GPU服务器运行文档
7. [ ] 离线依赖说明文档

## 无网GPU服务器假设
- Linux + CUDA 11.x/12.x
- 无公网连接
- PyTorch已安装
- 项目代码已上传
- 数据集已上传到DATA_ROOT
- DEVICE=cuda:0为默认

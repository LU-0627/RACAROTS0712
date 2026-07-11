# 修改文件列表

## 已修改的文件

1. **config.py**
   - 添加完整的 `_C.MODEL` 和 `_C.RDCAROTS` 配置节点
   - 添加 `_C.SYNTHETIC` 节点
   - 确保所有子节点与 YAML 配置对应

2. **models/rd_carots/modeling_rd_carots.py**
   - 修改输入输出划分逻辑
   - 使用 `cfg.RDCAROTS.N_INPUTS` 和 `cfg.RDCAROTS.N_OUTPUTS`
   - 移除 `n_vars // 2` 硬编码

3. **models/rd_carots/trainer_rd_carots.py**
   - 在 `__init__` 中添加 `self.device = next(model.parameters()).device`
   - 添加 `self.current_sim_threshold` 状态变量
   - 修改 `train_step` 使用 `prepare_inputs(inputs, device=self.device)`
   - 将 SIM_THRESHOLD 调度改为修改状态变量而非冻结的 cfg
   - 传递 `current_sim_threshold` 给损失函数

4. **models/rd_carots/loss_rd_carots.py**
   - 添加 `current_sim_threshold` 参数
   - 使用传入的阈值而非直接读取 cfg

5. **scripts/server/06_run_synthetic_smoke.sh**
   - 移除 `|| echo "CAROTS smoke failed"`
   - 移除 `|| echo "RDCAROTS smoke failed"`
   - 任一失败直接返回非零状态

6. **scripts/server/RUN_ALL_SERVER.sh**
   - 完成 Steps 6-12 的实际调用
   - 移除 "not implemented" 占位符
   - 依次调用 06 至 12 号脚本

7. **tests/rd_carots/test_one_epoch_smoke.py** (新增)
   - 生成合成数据
   - 训练 1 epoch
   - 保存和加载 checkpoint

8. **tests/rd_carots/test_guarded_update.py** (新增)
   - 验证 guarded_online 更新条件
   - 验证不使用测试标签

9. **tests/rd_carots/test_original_carots_compatibility.py** (新增)
   - 验证原 CAROTS 仍可选择和构建
   - 验证 checkpoint 目录隔离
   - 验证前向传播正常

10. **tests/rd_carots/test_synthetic_end_to_end.py** (新增)
    - 完整流程测试
    - 数据生成 → 训练 → frozen → guarded_online

---

## 服务器上应执行的命令

### 1. 环境检查
```bash
cd RDCAROTS_SERVER_PACKAGE/project
export PROJECT_ROOT="$(pwd)"
export DATA_ROOT="/path/to/data"
export OUTPUT_ROOT="$PROJECT_ROOT/results/rd_carots"
export PYTHON_BIN="python"
export CUDA_VISIBLE_DEVICES="0"

bash scripts/server/00_check_environment.sh
```

### 2. 数据检查
```bash
bash scripts/server/02_check_data.sh
```

### 3. 语法检查
```bash
bash scripts/server/03_run_compile_check.sh
```

### 4. 运行所有单元测试
```bash
bash scripts/server/04_run_tests.sh
```

### 5. 生成合成数据
```bash
bash scripts/server/05_generate_synthetic.sh
```

### 6. 运行完整测试套件
```bash
bash scripts/server/RUN_ALL_SERVER.sh
```

### 7. 仅运行 smoke test
```bash
bash scripts/server/RUN_ALL_SERVER.sh --smoke-only
```

### 8. 跳过真实数据集
```bash
bash scripts/server/RUN_ALL_SERVER.sh --skip-swat --skip-wadi
```

### 9. 收集结果
```bash
bash scripts/server/12_collect_results.sh
```

---

## 修复的问题

1. ✅ config.py 中完整定义 RDCAROTS 配置节点
2. ✅ RDCAROTSTrainer 明确初始化 self.device
3. ✅ RDCAROTS.forward() 使用配置的 N_INPUTS/N_OUTPUTS
4. ✅ SIM_THRESHOLD 调度使用状态变量而非修改冻结 cfg
5. ✅ 06_run_synthetic_smoke.sh 失败时正确返回非零状态
6. ✅ RUN_ALL_SERVER.sh 完成所有步骤的实际调用
7. ✅ 新增 4 个端到端测试文件

---

## 测试文件总数

**15 个测试文件**（原 11 个 + 新增 4 个）

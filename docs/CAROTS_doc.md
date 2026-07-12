# CAROTS 项目变更记忆文档

---

## 2026-07-12 03:44 - 上传至 RACAROTS0712 仓库（排除 data 文件夹）

### 修改文件：`.gitignore`

**修改原因：** 将项目上传至 `https://github.com/LU-0627/RACAROTS0712.git` 时，需排除 `data/` 文件夹。

```diff
 .vscode/
-**/__pycache__/
+**/__pycache__/
+data/
```

### 其他操作记录

- 从 git 跟踪中移除 `data/` 文件夹（`git rm -r --cached data/`），本地文件保留
- 更换远程仓库地址：`git@github.com:kimanki/CAROTS.git` -> `git@github.com:LU-0627/RACAROTS0712.git`
- 提交并推送至新仓库 `master` 分支

---

## 2026-07-12 04:18 - 同步更新至 RACAROTS0712 仓库

### 操作记录

将项目最新更改提交并推送到 `git@github.com:LU-0627/RACAROTS0712.git`。

本次提交包含 56 个文件变更（2681 行新增，1593 行删除），主要内容：
- 修改了 `config.py`、`datasets/build.py`、`datasets/synthetic_regime_delay.py` 等核心代码
- 修改了 `models/rd_carots/` 下多个模块（augmentors、model_bank、loss_rd_carots、modeling_rd_carots、scorer_rd_carots、trainer_rd_carots）
- 新增 `models/rd_carots/result_writer.py`
- 新增多个测试文件（test_cross_regime_mismatch、test_devices、test_guarded_update 等）
- 新增文档（ACCEPTANCE_REPORT.md、FINAL_ACCEPTANCE.md、VALIDATION_REPORT.md 等）
- 修改了 `scripts/server/` 下多个脚本
- 新增 `tools/functional_delivery_audit.py`

提交哈希：`80898e9`

---

## 2026-07-12 04:32 - 同步 GPU 服务器部署相关更新

### 操作记录

将项目最新更改提交并推送到 `git@github.com:LU-0627/RACAROTS0712.git`。

本次提交包含 9 个文件变更（445 行新增，23 行删除），主要内容：
- 修改 `models/build.py`、`models/rd_carots/modeling_rd_carots.py`、`run_rd_carots.py`、`scripts/server/00_check_environment.sh`
- 新增 `GPU_SERVER_DELIVERY_FINAL.md`、`GPU_SERVER_DEPLOYMENT_CRITICAL_FIXES.md`、`HARDBLOCK_FIXES_SUMMARY.txt`
- 新增 `scripts/server/06_run_synthetic_gpu_smoke.sh`、`scripts/server/RUN_ALL_SERVER_GPU.sh`

提交哈希：`730843d`

---

## 2026-07-12 12:39 - 同步静态验证报告

### 操作记录

将项目最新更改提交并推送到 `git@github.com:LU-0627/RACAROTS0712.git`。

本次提交包含 6 个文件变更（227 行新增），主要内容：
- 修改 `models/build.py`
- 新增静态验证报告：`FINAL_DELIVERY_STATIC_CHECK.md`、`FINAL_STATIC_CHECK.txt`、`FINAL_STATIC_VALIDATION.txt`、`FINAL_STATIC_VALIDATION_OUTPUT.txt`、`FINAL_STATUS.txt`、`STATIC_VALIDATION_RESULTS.txt`

提交哈希：`fdd8f91`

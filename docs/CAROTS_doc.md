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

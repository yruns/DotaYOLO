## conda-pack 详解

### 安装

```bash
# 在源机器安装
conda install -c conda-forge conda-pack
# 或者
pip install conda-pack
```

### 打包流程

```bash
# 打包当前激活的环境
conda pack -o myenv.tar.gz

# 打包指定名称的环境
conda pack -n myenv -o myenv.tar.gz

# 打包指定路径的环境
conda pack -p /path/to/myenv -o myenv.tar.gz

# 常用参数
conda pack -n myenv \
    -o myenv.tar.gz \
    --ignore-editable-packages \   # 忽略 editable 安装的包
    --ignore-missing-files \       # 忽略缺失文件
    --compress-level 5             # 压缩级别 0-9
```

### 目标机器解压与激活

```bash
# 创建目录并解压
mkdir -p ~/envs/myenv
tar -xzf myenv.tar.gz -C ~/envs/myenv

# 激活环境
source ~/envs/myenv/bin/activate

# 【重要】修复路径前缀
conda-unpack   # 或 source ~/envs/myenv/bin/conda-unpack

# 之后就可以正常使用了
python --version
```

### 为什么需要 `conda-unpack`？

打包时，环境中的很多脚本（如 `pip`、`python`）内部硬编码了原机器的路径，例如：

```bash
#!/home/shyue/miniconda3/envs/myenv/bin/python
```

`conda-unpack` 会自动把这些 shebang 和路径替换成目标机器的实际路径。**只需运行一次**，之后删除压缩包也没关系。

### 注意事项

| 问题 | 说明 |
|------|------|
| **平台一致性** | 源机器和目标机器的 OS 和架构必须一致（如都是 Linux x86_64） |
| **Editable 包** | `pip install -e .` 安装的包不会被打包，需要单独处理 |
| **CUDA/cuDNN** | GPU 相关库可能依赖系统路径，需要目标机器也有对应驱动 |
| **绝对路径配置** | 某些包（如 Jupyter）可能有配置文件写死了路径 |

### 完整示例

```bash
# ========== 源机器 ==========
# 假设有一个叫 spark_env 的环境
conda activate spark_env
conda install -c conda-forge conda-pack

# 打包
conda pack -n spark_env -o spark_env.tar.gz
# 输出类似：Collecting packages...
#          Packing environment to spark_env.tar.gz
#          [########################################] | 100%

# 传输到目标机器
scp spark_env.tar.gz user@target:/home/user/

# ========== 目标机器 ==========
mkdir -p ~/envs/spark_env
tar -xzf spark_env.tar.gz -C ~/envs/spark_env

# 激活并修复路径
source ~/envs/spark_env/bin/activate
conda-unpack

# 验证
which python
# /home/user/envs/spark_env/bin/python
python -c "import pyspark; print(pyspark.__version__)"
```

---

你那边是打包 PySpark 相关的环境吗？如果有 CUDA 依赖的话可能需要额外注意一些细节。
﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿# _run_python.ps1 - backend 统一 Python 调用约定
#
# 解决的问题（历史踩坑）：
#   1. 系统 python 缺 pydantic_settings，必须用 venv python
#   2. 相对路径层级容易算错（../.. 多写一级），统一用绝对路径
#   3. 复杂逻辑应写成 .py 脚本，避免超长单行 + 管道 + 中文组合
#   4. 长任务脚本应幂等可重入，避免依赖"重试"
#
# 用法：
#   1) 直接执行（推荐）：
#        .\scripts\_run_python.ps1 eval_ablation.py --top-k 8
#     脚本路径相对 backend/ 或绝对路径均可。
#   2) dot-source 加载后调用函数：
#        . .\scripts\_run_python.ps1
#        Invoke-Py eval_ablation.py --top-k 8
#        $PY                        # venv python 绝对路径
#        $BACKEND_DIR / $ROOT_DIR   # 目录绝对路径
#
# 约定：
#   - 一律用 venv python（backend\.venv\Scripts\python.exe）
#   - 统一切换到 backend/ 作为 cwd（app.* 模块可正常 import）
#   - 脚本路径统一解析为绝对路径，避免相对路径层级出错

$ErrorActionPreference = 'Stop'

# 定位目录（本脚本位于 backend/scripts/）
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$BACKEND_DIR = Split-Path -Parent $SCRIPT_DIR
$ROOT_DIR = Split-Path -Parent $BACKEND_DIR

# 唯一认可的 python 解释器（绝对路径）
$VENV_PY = Join-Path $BACKEND_DIR '.venv\Scripts\python.exe'
if (-not (Test-Path $VENV_PY)) {
    throw "未找到 venv python: $VENV_PY（请确认 backend/.venv 已创建）"
}
$PY = $VENV_PY

# 统一调用：切换到 backend cwd + venv python + 绝对脚本路径
function Invoke-Py {
    param(
        [Parameter(Mandatory = $true)][string]$Script,
        [string[]]$PyArgs = @()
    )
    $scriptPath = if ([System.IO.Path]::IsPathRooted($Script)) {
        $Script
    } else {
        Join-Path $SCRIPT_DIR $Script
    }
    if (-not (Test-Path $scriptPath)) {
        throw "脚本不存在: $scriptPath（应相对 scripts/ 或绝对路径）"
    }
    $pyArgs = @($scriptPath) + @($PyArgs)
    Push-Location $BACKEND_DIR
    try {
        & $PY @pyArgs
    } finally {
        Pop-Location
    }
}

# 直接执行模式：.\scripts\_run_python.ps1 <script> [args...]
if ($MyInvocation.InvocationName -ne '.') {
    $directArgs = @($MyInvocation.UnboundArguments)
    if ($directArgs.Count -lt 1) {
        Write-Host "用法: .\scripts\_run_python.ps1 <script.py> [args...]" -ForegroundColor Yellow
        Write-Host "或 dot-source 后调用: . .\scripts\_run_python.ps1 ; Invoke-Py <script.py> [args...]" -ForegroundColor Yellow
        exit 1
    }
    $script = $directArgs[0]
    $rest = @()
    if ($directArgs.Count -gt 1) {
        $rest = $directArgs[1..($directArgs.Count - 1)]
    }
    exit (Invoke-Py $script $rest)
}
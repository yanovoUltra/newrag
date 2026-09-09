[CmdletBinding()]
param(
    [string]$ModelPath = 'D:\newrag-reranker\models\Qwen3-Reranker-8B',
    [string]$IndicatorAdapter = 'D:\newrag-reranker\runs\qwen3-reranker-8b-cover-all-positive-lr5e5\final-adapter',
    [string]$NonIndicatorAdapter = 'D:\newrag-reranker\v2\final-non-indicator-v2\final-adapter',
    [string]$ListenHost = '127.0.0.1',
    [int]$Port = 8010
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$Python = 'D:\newrag-reranker\.venv\Scripts\python.exe'
$Server = Join-Path $RepoRoot 'backend\scripts\reranker_finetune\serve_local.py'
$StateDir = Join-Path $env:LOCALAPPDATA 'NewRAG\reranker'
$PidPath = Join-Path $StateDir 'reranker.pid'
$StdoutPath = Join-Path $StateDir 'reranker.stdout.log'
$StderrPath = Join-Path $StateDir 'reranker.stderr.log'

foreach ($required in @($Python, $Server, $ModelPath, $IndicatorAdapter, $NonIndicatorAdapter)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path does not exist: $required"
    }
}
if ($null -ne (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) {
    throw "Port $Port is already occupied"
}

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
$env:HF_HOME = 'D:\newrag-reranker\hf-home'
$env:TOKENIZERS_PARALLELISM = 'false'
$arguments = @(
    '-m', 'scripts.reranker_finetune.serve_local',
    '--model', $ModelPath,
    '--indicator-adapter', $IndicatorAdapter,
    '--non-indicator-adapter', $NonIndicatorAdapter,
    '--default-profile', 'finance-indicator',
    '--host', $ListenHost,
    '--port', $Port,
    '--max-length', '768',
    '--batch-size', '2'
)

$started = Start-Process -FilePath $Python -ArgumentList $arguments `
    -WorkingDirectory (Join-Path $RepoRoot 'backend') `
    -WindowStyle Hidden `
    -RedirectStandardOutput $StdoutPath `
    -RedirectStandardError $StderrPath `
    -PassThru
Set-Content -LiteralPath $PidPath -Value $started.Id -NoNewline

try {
    $started.WaitForExit()
    exit $started.ExitCode
}
finally {
    if (Test-Path -LiteralPath $PidPath) {
        $savedPid = (Get-Content -LiteralPath $PidPath -Raw).Trim()
        if ($savedPid -eq [string]$started.Id) {
            Remove-Item -LiteralPath $PidPath -ErrorAction SilentlyContinue
        }
    }
}

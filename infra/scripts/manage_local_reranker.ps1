[CmdletBinding()]
param(
    [ValidateSet('install', 'start', 'stop', 'status')]
    [string]$Action = 'status',
    [string]$ModelPath = 'D:\newrag-reranker\models\Qwen3-Reranker-8B',
    [string]$IndicatorAdapter = 'D:\newrag-reranker\runs\qwen3-reranker-8b-cover-all-positive-lr5e5\final-adapter',
    [string]$NonIndicatorAdapter = 'D:\newrag-reranker\v2\final-non-indicator-v2\final-adapter',
    [string]$ListenHost = '127.0.0.1',
    [int]$Port = 8010,
    [int]$ReadyTimeoutSeconds = 600,
    [string]$TaskName = 'NewRAG Local Reranker'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$Python = 'D:\newrag-reranker\.venv\Scripts\python.exe'
$Server = Join-Path $RepoRoot 'backend\scripts\reranker_finetune\serve_local.py'
$Runner = Join-Path $RepoRoot 'infra\scripts\run_local_reranker.ps1'
$StateDir = Join-Path $env:LOCALAPPDATA 'NewRAG\reranker'
$PidPath = Join-Path $StateDir 'reranker.pid'
$StdoutPath = Join-Path $StateDir 'reranker.stdout.log'
$StderrPath = Join-Path $StateDir 'reranker.stderr.log'
$HealthUrl = "http://127.0.0.1:$Port/healthz"

function Get-RerankerHealth {
    try {
        return Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2
    }
    catch {
        return $null
    }
}

function Get-OwnedProcess {
    $process = $null
    if (Test-Path -LiteralPath $PidPath) {
        $savedPid = [int](Get-Content -LiteralPath $PidPath -Raw).Trim()
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$savedPid" -ErrorAction SilentlyContinue
    }
    if ($null -eq $process) {
        $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -ne $listener) {
            $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)" -ErrorAction SilentlyContinue
        }
    }
    if ($null -eq $process) { return $null }
    if ($process.CommandLine -notlike '*scripts.reranker_finetune.serve_local*') {
        throw "Port/PID state points to a process not owned by NewRAG reranker: $($process.ProcessId)"
    }
    return $process
}

function Get-RerankerTask {
    return Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
}

function Install-RerankerTask {
    foreach ($required in @($Python, $Server, $Runner, $ModelPath, $IndicatorAdapter, $NonIndicatorAdapter)) {
        if (-not (Test-Path -LiteralPath $required)) { throw "Required path does not exist: $required" }
    }
    $powerShell = (Get-Command powershell.exe -ErrorAction Stop).Source
    $taskArguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$Runner`""
    $taskAction = New-ScheduledTaskAction -Execute $powerShell -Argument $taskArguments -WorkingDirectory (Join-Path $RepoRoot 'backend')
    $taskTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $taskSettings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -MultipleInstances IgnoreNew
    $taskSettings.DisallowStartIfOnBatteries = $false
    $taskSettings.StopIfGoingOnBatteries = $false
    $principal = New-ScheduledTaskPrincipal `
        -UserId "$env:USERDOMAIN\$env:USERNAME" `
        -LogonType Interactive `
        -RunLevel Limited
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $taskAction `
        -Trigger $taskTrigger `
        -Settings $taskSettings `
        -Principal $principal `
        -Description 'NewRAG loopback Qwen3 financial reranker service' `
        -Force `
        -ErrorAction Stop | Out-Null
    if ($null -eq (Get-RerankerTask)) {
        throw "Scheduled task registration did not create '$TaskName'"
    }
}

if ($Action -eq 'install') {
    Install-RerankerTask
    $task = Get-RerankerTask
    [pscustomobject]@{
        status = 'installed'
        task_name = $TaskName
        task_state = [string]$task.State
        endpoint = $HealthUrl
    } | ConvertTo-Json -Compress
    exit 0
}

if ($Action -eq 'status') {
    $health = Get-RerankerHealth
    $process = Get-OwnedProcess
    $task = Get-RerankerTask
    [pscustomobject]@{
        status = if ($null -ne $health) { 'healthy' } elseif ($null -ne $process) { 'starting_or_unhealthy' } else { 'stopped' }
        pid = if ($null -ne $process) { $process.ProcessId } else { $null }
        task_name = $TaskName
        task_state = if ($null -ne $task) { [string]$task.State } else { 'not_installed' }
        endpoint = $HealthUrl
        model = if ($null -ne $health) { $health.model } else { $null }
        default_profile = if ($null -ne $health) { $health.default_profile } else { $null }
        profiles = if ($null -ne $health) { @($health.profiles) } else { @() }
    } | ConvertTo-Json -Compress
    exit 0
}

if ($Action -eq 'stop') {
    $task = Get-RerankerTask
    if ($null -ne $task -and $task.State -eq 'Running') {
        Stop-ScheduledTask -TaskName $TaskName
        Start-Sleep -Seconds 2
    }
    $process = Get-OwnedProcess
    if ($null -eq $process) {
        Remove-Item -LiteralPath $PidPath -ErrorAction SilentlyContinue
        Write-Output '{"status":"already_stopped"}'
        exit 0
    }
    $ownedProcesses = @($process)
    $parents = @($process.ProcessId)
    while ($parents.Count -gt 0) {
        $children = @(Get-CimInstance Win32_Process | Where-Object { $parents -contains $_.ParentProcessId })
        foreach ($child in $children) {
            if ($child.CommandLine -notlike '*scripts.reranker_finetune.serve_local*') {
                throw "Refusing to stop an unrelated child process: $($child.ProcessId)"
            }
        }
        $ownedProcesses += $children
        $parents = @($children.ProcessId)
    }
    foreach ($owned in @($ownedProcesses | Sort-Object ProcessId -Descending)) {
        Stop-Process -Id $owned.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $PidPath -ErrorAction SilentlyContinue
    Write-Output ('{"status":"stopped","pid":' + $process.ProcessId + '}')
    exit 0
}

$existingHealth = Get-RerankerHealth
if ($null -ne $existingHealth) {
    Write-Output ($existingHealth | ConvertTo-Json -Compress)
    exit 0
}
if ($null -ne (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) {
    throw "Port $Port is occupied by an unhealthy or unrelated process"
}
$task = Get-RerankerTask
if ($null -eq $task) {
    Install-RerankerTask
}
Start-ScheduledTask -TaskName $TaskName

$deadline = (Get-Date).AddSeconds($ReadyTimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    $health = Get-RerankerHealth
    if ($null -ne $health) {
        Write-Output ($health | ConvertTo-Json -Compress)
        exit 0
    }
    $task = Get-RerankerTask
    if ($null -eq $task -or $task.State -ne 'Running') {
        $taskInfo = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction SilentlyContinue
        throw "Local reranker task stopped during startup (result=$($taskInfo.LastTaskResult)); inspect $StderrPath"
    }
    Start-Sleep -Seconds 3
}
throw "Local reranker did not become healthy within $ReadyTimeoutSeconds seconds"

<#
.SYNOPSIS
    Start the GR00T N1.7 policy server (scripts/groot_server.py) on loopback,
    with the environment it needs scoped to the server process ONLY.

.DESCRIPTION
    The real server loaded on this machine only with (logs/e2e/groot_server2.txt):

      PYTHONPYCACHEPREFIX   the GR00T venv has corrupt .pyc files; a fresh cache
                            prefix bypasses them
      HF_HUB_OFFLINE=1      transformers otherwise calls the Hub for the
      TRANSFORMERS_OFFLINE=1  Cosmos-Reason2-2B tokenizer even when it is cached
      GROOT_PATCH_MISTRAL=1 Isaac-GR00T's own guard for that call

    These must NEVER reach Isaac Sim: exporting PYTHONPYCACHEPREFIX into the
    Isaac process rewrote 3504 .pyc files and broke it ("source code string
    cannot contain null bytes"). So this script never assigns $env:*. The
    variables are placed on a ProcessStartInfo for the child alone; the calling
    shell -- even when the script is dot-sourced -- is left untouched.

    The model is resolved to a LOCAL snapshot directory from the HuggingFace
    cache (refs/main -> snapshots/<commit>), because on Windows a repo id is
    mangled by Path() inside Gr00tPolicy, and offline mode cannot download.

.PARAMETER GrootEnv
    The GR00T venv. Default: %USERPROFILE%\groot_env

.PARAMETER GrootPython
    The interpreter to run. Default: <GrootEnv>\Scripts\python.exe

.PARAMETER Model
    A snapshot directory. Overrides the cache lookup.

.PARAMETER RepoId
    The repo to look up in the cache. Default: nvidia/GR00T-N1.7-3B

.PARAMETER HfCache
    The HuggingFace hub cache. Default: $env:HF_HUB_CACHE, else
    $env:HF_HOME\hub, else %USERPROFILE%\.cache\huggingface\hub

.PARAMETER PycachePrefix
    Where the server's bytecode goes. Default: %LOCALAPPDATA%\groot_pycache

.PARAMETER DryRun
    Print the resolved command and the child-only environment, start nothing.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_groot_server.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_groot_server.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [string]$GrootEnv = (Join-Path $env:USERPROFILE 'groot_env'),
    [string]$GrootPython = '',
    [string]$Model = '',
    [string]$RepoId = 'nvidia/GR00T-N1.7-3B',
    [string]$HfCache = '',
    [string]$PycachePrefix = '',
    [int]$Port = 5555,
    [string]$Device = 'cuda:0',
    [string]$Embodiment = 'oxe_droid_relative_eef_relative_joint',
    [string]$ServerScript = '',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

function Fail([string]$message) {
    [Console]::Error.WriteLine("start_groot_server: $message")
    exit 2
}

# --- interpreter -----------------------------------------------------------
if (-not $GrootPython) {
    $GrootPython = Join-Path $GrootEnv 'Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $GrootPython -PathType Leaf)) {
    Fail "GR00T python not found at '$GrootPython' (pass -GrootEnv or -GrootPython)"
}

# --- server script -----------------------------------------------------------
if (-not $ServerScript) {
    $ServerScript = Join-Path $PSScriptRoot 'groot_server.py'
}
if (-not (Test-Path -LiteralPath $ServerScript -PathType Leaf)) {
    Fail "server script not found at '$ServerScript'"
}

# --- model snapshot ----------------------------------------------------------
if (-not $HfCache) {
    if ($env:HF_HUB_CACHE) {
        $HfCache = $env:HF_HUB_CACHE
    } elseif ($env:HF_HOME) {
        $HfCache = Join-Path $env:HF_HOME 'hub'
    } else {
        $HfCache = Join-Path $env:USERPROFILE '.cache\huggingface\hub'
    }
}

if ($Model) {
    if (-not (Test-Path -LiteralPath $Model -PathType Container)) {
        Fail "-Model '$Model' is not a directory"
    }
    $snapshot = (Resolve-Path -LiteralPath $Model).Path
} else {
    $repoDir = Join-Path $HfCache ('models--' + ($RepoId -replace '/', '--'))
    $snapshotsDir = Join-Path $repoDir 'snapshots'
    if (-not (Test-Path -LiteralPath $snapshotsDir -PathType Container)) {
        Fail "'$RepoId' is not in the HuggingFace cache at '$HfCache' (pass -Model <snapshot dir> or -HfCache)"
    }
    $snapshot = $null
    $ref = Join-Path $repoDir 'refs\main'
    if (Test-Path -LiteralPath $ref -PathType Leaf) {
        $commit = (Get-Content -LiteralPath $ref -Raw).Trim()
        $candidate = Join-Path $snapshotsDir $commit
        if (Test-Path -LiteralPath (Join-Path $candidate 'config.json') -PathType Leaf) {
            $snapshot = $candidate
        }
    }
    if (-not $snapshot) {
        # refs/main missing or stale: newest snapshot with a config.json.
        $complete = Get-ChildItem -LiteralPath $snapshotsDir -Directory |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'config.json') -PathType Leaf } |
            Sort-Object LastWriteTime -Descending
        if (-not $complete) {
            Fail "no complete snapshot (with config.json) of '$RepoId' under '$snapshotsDir'"
        }
        $snapshot = @($complete)[0].FullName
    }
}

if (-not $PycachePrefix) {
    $PycachePrefix = Join-Path $env:LOCALAPPDATA 'groot_pycache'
}

# --- the child-only environment ------------------------------------------------
$childEnv = [ordered]@{
    'PYTHONPYCACHEPREFIX'  = $PycachePrefix
    'HF_HUB_OFFLINE'       = '1'
    'TRANSFORMERS_OFFLINE' = '1'
    'GROOT_PATCH_MISTRAL'  = '1'
}

if ($env:PYTHONPYCACHEPREFIX) {
    Write-Warning ("PYTHONPYCACHEPREFIX is set in THIS shell ('$env:PYTHONPYCACHEPREFIX'). " +
        "Anything else started from here -- Isaac Sim included -- inherits it, and Isaac " +
        "breaks with it. Unset it in this shell; this launcher sets it for the server only.")
}

$arguments = @(
    '-u', $ServerScript,
    '--model', $snapshot,
    '--host', '127.0.0.1',
    '--port', "$Port",
    '--device', $Device,
    '--embodiment', $Embodiment
)

function Quote([string]$value) {
    if ($value -match '[\s"]') { return '"' + ($value -replace '"', '\"') + '"' }
    return $value
}
$argumentLine = ($arguments | ForEach-Object { Quote $_ }) -join ' '

Write-Output "PYTHON=$GrootPython"
Write-Output "MODEL=$snapshot"
foreach ($name in $childEnv.Keys) { Write-Output "CHILD_ENV $name=$($childEnv[$name])" }
Write-Output "COMMAND=$(Quote $GrootPython) $argumentLine"

if ($DryRun) {
    exit 0
}

New-Item -ItemType Directory -Force -Path $PycachePrefix | Out-Null

$info = New-Object System.Diagnostics.ProcessStartInfo
$info.FileName = $GrootPython
$info.Arguments = $argumentLine
$info.UseShellExecute = $false          # required for EnvironmentVariables; shares this console
$info.WorkingDirectory = (Split-Path -Parent $PSScriptRoot)
foreach ($name in $childEnv.Keys) {
    $info.EnvironmentVariables[$name] = [string]$childEnv[$name]
}

$process = [System.Diagnostics.Process]::Start($info)
$process.WaitForExit()
exit $process.ExitCode

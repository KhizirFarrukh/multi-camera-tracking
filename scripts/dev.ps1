<#
.SYNOPSIS
    Windows-native equivalent of the Makefile targets.

.DESCRIPTION
    GNU make is not present on a default Windows install, and the exit criteria
    for every stage are expressed as make targets. This script mirrors them
    one-for-one so a Windows developer runs the same gates as CI.

.PARAMETER Target
    The target to run. Omit to list available targets.

.EXAMPLE
    .\scripts\dev.ps1 install
    .\scripts\dev.ps1 ci
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet(
        'help', 'install', 'install-vision', 'lint', 'format', 'format-check',
        'typecheck', 'test-unit', 'test-integration', 'test-all', 'coverage',
        'db-up', 'db-down', 'db-reset', 'db-logs', 'clean', 'ci'
    )]
    [string]$Target = 'help',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Invoke-Step {
    param([string]$Description, [scriptblock]$Action)
    Write-Host "==> $Description" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Description (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

function Show-Help {
    Write-Host "Targets:" -ForegroundColor Yellow
    @(
        @{ n = 'install';          d = 'Create the venv and install core + dev dependencies' },
        @{ n = 'install-vision';   d = 'Additionally install the CV stack (slow)' },
        @{ n = 'lint';             d = 'Run ruff lint checks' },
        @{ n = 'format';           d = 'Rewrite files with the ruff formatter' },
        @{ n = 'format-check';     d = 'Verify formatting without rewriting' },
        @{ n = 'typecheck';        d = 'Run mypy in strict mode' },
        @{ n = 'test-unit';        d = 'Run unit tests only (fast, no docker)' },
        @{ n = 'test-integration'; d = 'Run integration tests (requires docker)' },
        @{ n = 'test-all';         d = 'Run the whole suite with the coverage gate' },
        @{ n = 'coverage';         d = 'Run the suite and write an HTML coverage report' },
        @{ n = 'db-up';            d = 'Start Postgres+pgvector and wait for health' },
        @{ n = 'db-down';          d = 'Stop Postgres, preserving the data volume' },
        @{ n = 'db-reset';         d = 'Stop Postgres and DESTROY the data volume' },
        @{ n = 'db-logs';          d = 'Tail the Postgres logs' },
        @{ n = 'clean';            d = 'Remove caches and build artefacts' },
        @{ n = 'ci';               d = 'Everything CI runs, in CI order' }
    ) | ForEach-Object { Write-Host ("  {0,-18} {1}" -f $_.n, $_.d) }
}

switch ($Target) {
    'help' { Show-Help }

    'install'        { Invoke-Step 'uv sync (core + dev)' { uv sync --extra dev } }
    'install-vision' { Invoke-Step 'uv sync (core + dev + vision)' { uv sync --extra dev --extra vision } }

    'lint'         { Invoke-Step 'ruff check' { uv run ruff check . } }
    'format'       {
        Invoke-Step 'ruff format' { uv run ruff format . }
        Invoke-Step 'ruff check --fix' { uv run ruff check --fix . }
    }
    'format-check' { Invoke-Step 'ruff format --check' { uv run ruff format --check . } }
    'typecheck'    { Invoke-Step 'mypy' { uv run mypy } }

    'test-unit'        { Invoke-Step 'pytest (unit)' { uv run pytest tests/unit -m "not integration" } }
    'test-integration' { Invoke-Step 'pytest (integration)' { uv run pytest tests/integration -m integration } }
    'test-all'         { Invoke-Step 'pytest (all)' { uv run pytest } }
    'coverage'         {
        Invoke-Step 'pytest (html coverage)' { uv run pytest --cov-report=html }
        Write-Host 'Report written to htmlcov/index.html'
    }

    'db-up'    { Invoke-Step 'docker compose up' { docker compose up -d --wait postgres } }
    'db-down'  { Invoke-Step 'docker compose down' { docker compose down } }
    'db-reset' { Invoke-Step 'docker compose down -v' { docker compose down -v } }
    'db-logs'  { docker compose logs -f postgres }

    'clean' {
        $paths = @(
            '.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov',
            '.coverage', 'coverage.xml', 'build', 'dist'
        )
        foreach ($path in $paths) {
            if (Test-Path $path) { Remove-Item -Recurse -Force $path }
        }
        Get-ChildItem -Recurse -Directory -Filter '__pycache__' |
            Remove-Item -Recurse -Force
        Write-Host 'Cleaned.' -ForegroundColor Green
    }

    'ci' {
        Invoke-Step 'ruff check' { uv run ruff check . }
        Invoke-Step 'ruff format --check' { uv run ruff format --check . }
        Invoke-Step 'mypy' { uv run mypy }
        Invoke-Step 'pytest (all)' { uv run pytest }
        Write-Host 'CI passed.' -ForegroundColor Green
    }
}

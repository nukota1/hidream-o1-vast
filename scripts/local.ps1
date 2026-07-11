param(
    [ValidateSet("start", "stop", "restart", "logs", "status", "build")]
    [string]$Action = "start"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $ProjectRoot "docker-compose.local.yml"

if (-not (Test-Path (Join-Path $ProjectRoot ".env"))) {
    throw "Missing .env. Copy .env.example to .env and set the R2 credentials."
}

if (-not (Test-Path (Join-Path $ProjectRoot ".env.local"))) {
    Copy-Item (Join-Path $ProjectRoot ".env.local.example") (Join-Path $ProjectRoot ".env.local")
    Write-Host "Created .env.local. Adjust settings if needed, then run this command again."
    exit 0
}

$compose = @("compose", "-f", $ComposeFile)
switch ($Action) {
    "start"   { & docker @compose up --build --detach }
    "stop"    { & docker @compose down }
    "restart" { & docker @compose up --build --detach --force-recreate }
    "logs"    { & docker @compose logs --follow --tail 200 }
    "status"  { & docker @compose ps }
    "build"   { & docker @compose build }
}

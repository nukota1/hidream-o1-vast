param(
    [ValidateSet("start", "stop", "restart", "logs", "status", "build")]
    [string]$Action = "start"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $ProjectRoot "docker-compose.local.yml"
$ComposeProject = "janku-local"
$ModelsVolume = "janku-models-local"
$RuntimeVolume = "janku-python-local"

if (-not (Test-Path (Join-Path $ProjectRoot ".env"))) {
    throw "Missing .env. Copy .env.example to .env and set the R2 credentials."
}

if (-not (Test-Path (Join-Path $ProjectRoot ".env.local"))) {
    Copy-Item (Join-Path $ProjectRoot ".env.local.example") (Join-Path $ProjectRoot ".env.local")
    Write-Host "Created .env.local. Adjust settings if needed, then run this command again."
    exit 0
}

$legacyCompose = Get-Command "docker-compose" -ErrorAction SilentlyContinue
if ($legacyCompose) {
    $composeCommand = $legacyCompose.Source
    $composePrefix = @()
} else {
    $composeCommand = "docker"
    $composePrefix = @("compose")
}
$compose = @("-p", $ComposeProject, "-f", $ComposeFile)

function Invoke-LocalCompose {
    param([string[]]$ComposeActionArgs)

    & $composeCommand @composePrefix @compose @ComposeActionArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose failed with exit code $LASTEXITCODE."
    }
}

function Ensure-LocalVolume {
    param([string]$VolumeName)

    $existingVolumes = @(
        & docker volume ls --filter "name=$VolumeName" --format "{{.Name}}"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Could not list Docker volumes."
    }
    if ($existingVolumes -contains $VolumeName) {
        return
    }
    & docker volume create $VolumeName | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create Docker volume $VolumeName."
    }
}

function Ensure-LocalVolumes {
    Ensure-LocalVolume $ModelsVolume
    Ensure-LocalVolume $RuntimeVolume
}

switch ($Action) {
    "start"   { Ensure-LocalVolumes; Invoke-LocalCompose @("up", "--build", "--detach") }
    "stop"    { Invoke-LocalCompose @("down") }
    "restart" { Ensure-LocalVolumes; Invoke-LocalCompose @("up", "--build", "--detach", "--force-recreate") }
    "logs"    { Invoke-LocalCompose @("logs", "--follow", "--tail", "200") }
    "status"  { Invoke-LocalCompose @("ps") }
    "build"   { Invoke-LocalCompose @("build") }
}

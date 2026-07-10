param(
  [Parameter(Mandatory = $true)]
  [string]$Image,

  [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"

if ($Image -notmatch "^ghcr\.io/") {
  throw "Image must start with ghcr.io/, for example ghcr.io/YOUR_GITHUB_OWNER/janku-image-studio"
}

$fullTag = "${Image}:${Tag}"

Write-Host "Building $fullTag"
docker build -f Dockerfile.ghcr -t $fullTag .

Write-Host "Pushing $fullTag"
docker push $fullTag

Write-Host "Done: $fullTag"

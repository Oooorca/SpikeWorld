param([string]$TeXBin = '')
$ErrorActionPreference = 'Stop'

$paperRoot = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $paperRoot 'build'
if ($TeXBin) {
    $pdflatex = Join-Path $TeXBin 'pdflatex.exe'
    $bibtex = Join-Path $TeXBin 'bibtex.exe'
}
else {
    $pdflatex = (Get-Command pdflatex -ErrorAction Stop).Source
    $bibtex = (Get-Command bibtex -ErrorAction Stop).Source
}

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
Push-Location $paperRoot
try {
    & $pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex |
        Tee-Object -FilePath (Join-Path $buildDir 'compile_pass1.log')
    if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 1 failed: $LASTEXITCODE" }

    & $bibtex build/main |
        Tee-Object -FilePath (Join-Path $buildDir 'bibtex.log')
    if ($LASTEXITCODE -ne 0) { throw "bibtex failed: $LASTEXITCODE" }

    foreach ($pass in 2..4) {
        & $pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex |
            Tee-Object -FilePath (Join-Path $buildDir "compile_pass$pass.log")
        if ($LASTEXITCODE -ne 0) { throw "pdflatex pass $pass failed: $LASTEXITCODE" }
    }

    Copy-Item -LiteralPath (Join-Path $buildDir 'main.pdf') -Destination (Join-Path $paperRoot 'main.pdf') -Force
    Write-Host "Built: $(Join-Path $paperRoot 'main.pdf')"
}
finally {
    Pop-Location
}

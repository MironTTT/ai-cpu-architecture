# build.ps1 — сборка graph_evolve.cpp под Windows.
#
# Использование (из папки src/cpp или откуда угодно):
#     pwsh -File build.ps1
#
# Скрипт сам находит MSVC (через vswhere) и вызывает cl.exe с нужными флагами.
# Если MSVC не найден — пробует g++ из PATH.

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$src  = Join-Path $here 'graph_evolve.cpp'
$exe  = Join-Path $here 'graph_evolve.exe'
$obj  = Join-Path $here 'graph_evolve.obj'

if (-not (Test-Path $src)) { throw "Не найден исходник: $src" }

function Find-Vcvars {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $inst = & $vswhere -latest -products * `
                    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                    -property installationPath 2>$null
        if ($inst) {
            $candidate = Join-Path $inst 'VC\Auxiliary\Build\vcvars64.bat'
            if (Test-Path $candidate) { return $candidate }
        }
    }

    $known = @(
        'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat',
        'C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat',
        'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat',
        'C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
        'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat',
        'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
    )
    foreach ($p in $known) { if (Test-Path $p) { return $p } }
    return $null
}

$vcvars = Find-Vcvars

if ($vcvars) {
    Write-Host "MSVC:  $vcvars"
    $cmdLine = 'call "{0}" >nul && cl /nologo /std:c++17 /O2 /W4 /utf-8 /EHsc "{1}" /Fe:"{2}" /Fo:"{3}"' -f `
        $vcvars, $src, $exe, $obj
    & cmd.exe /c $cmdLine
    if ($LASTEXITCODE -ne 0) { throw "cl.exe завершился с кодом $LASTEXITCODE" }
} else {
    $gpp = Get-Command g++ -ErrorAction SilentlyContinue
    if (-not $gpp) {
        throw "Не найден ни MSVC (vswhere/vcvars64.bat), ни g++ в PATH."
    }
    Write-Host "g++:   $($gpp.Source)"
    & $gpp.Source -std=c++17 -O3 -march=native -Wall -Wextra -o $exe $src
    if ($LASTEXITCODE -ne 0) { throw "g++ завершился с кодом $LASTEXITCODE" }
}

if (Test-Path $obj) { Remove-Item $obj -Force }

Write-Host "Собрано: $exe"

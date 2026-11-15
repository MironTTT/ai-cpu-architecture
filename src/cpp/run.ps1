# run.ps1 — собрать (если нужно) и запустить graph_evolve, сохранив лог.
#
# Примеры:
#     .\run.ps1
#     .\run.ps1 --json
#     .\run.ps1 --evolve --from-random --gens 50
#     .\run.ps1 -Rebuild            # принудительно пересобрать
#     .\run.ps1 -Graphs             # исполнить графы, обученные в Python
#     .\run.ps1 -Train              # обучить графы в Python и сразу проверить
#
# Лог пишется в results/last_run.log и одновременно печатается в консоль.

param(
    [switch]$Rebuild,
    [switch]$Graphs,
    [switch]$Train,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'

# Кириллица в консоли: без этого вывод exe и Python может поехать.
try {
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $env:PYTHONIOENCODING = 'utf-8'
} catch { }

$here    = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe     = Join-Path $here 'graph_evolve.exe'
$root    = Split-Path -Parent (Split-Path -Parent $here)
$results = Join-Path $root 'results'

if ($Rebuild -or -not (Test-Path $exe)) {
    & (Join-Path $here 'build.ps1')
    if ($LASTEXITCODE -ne 0) { throw "Сборка не удалась" }
}

if (-not (Test-Path $results)) { New-Item -ItemType Directory -Path $results | Out-Null }
$log = Join-Path $results 'last_run.log'

# -Train: обучить структуру графа в Python (PyTorch), затем сразу проверить.
if ($Train) {
    $py = Join-Path $root '.venv\Scripts\python.exe'
    if (-not (Test-Path $py)) {
        $found = Get-Command python -ErrorAction SilentlyContinue
        if (-not $found) { throw "Не найден Python. Ожидался .venv\Scripts\python.exe" }
        $py = $found.Source
        Write-Host "Внимание: venv не найден, использую $py"
    }

    # Проверяем именно импорт: find_spec не поймает битую установку.
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $py -c "import torch" 2>&1 | Out-Null
    $hasTorch = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $saved
    if (-not $hasTorch) {
        throw "В интерпретаторе $py нет torch. Установи: & `"$py`" -m pip install torch"
    }

    $pyDir = Join-Path $root 'src\python'
    # Без --task/--all обучим все задачи: это то, чего ждёт человек по умолчанию.
    if ($null -eq $Rest) { $Rest = @() }
    $hasTask = $false
    $hasJson = $false
    foreach ($a in $Rest) {
        if ($a -eq '--task' -or $a -eq '--all') { $hasTask = $true }
        if ($a -eq '--json') { $hasJson = $true }
    }
    if (-not $hasTask) { $Rest = @('--all') + $Rest }
    # Сводку пишем всегда: иначе results/diff_graph_training.json отстаёт от кода.
    if (-not $hasJson) { $Rest += '--json' }

    Write-Host "Обучение структуры графа"
    Write-Host "  python: $py"
    Write-Host ""

    Push-Location $pyDir
    try {
        & $py -u -m diffgraph @Rest
        $trainCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($trainCode -ne 0) {
        # Это не поломка скрипта: обучение вернуло «не все задачи решены».
        Write-Host ""
        Write-Host "Обучение завершилось с кодом $trainCode — не все задачи решены на 100%."
        Write-Host "Смотрите на итог выше, а не на эту строку. Типичный случай:"
        Write-Host "  --refine 0 оставляет add2 на 93.75% — это и есть вклад добивки."
        exit $trainCode
    }

    $Graphs = $true
    $Rest   = @()
    Write-Host ""
}

# -Graphs: исполнить все графы, которые обучил Python (results/graphs/*.json).
if ($Graphs) {
    $graphDir = Join-Path $results 'graphs'
    $files = @(Get-ChildItem -Path $graphDir -Filter *.json -ErrorAction SilentlyContinue |
               Where-Object { $_.Name -notlike '*.unsolved.json' } |
               Sort-Object Name)
    if ($files.Count -eq 0) {
        throw "В $graphDir нет графов. Сначала: python -m diffgraph --all (из src/python)"
    }
    $Rest = @()
    foreach ($f in $files) { $Rest += '--load'; $Rest += $f.FullName }
    $Rest += '--json'
}

# Аргументы передаются как есть: .\run.ps1 --digits 9 --examples 500
$output = & $exe @Rest 2>&1
$code   = $LASTEXITCODE

$output | ForEach-Object { Write-Host $_ }
$output | Out-File -FilePath $log -Encoding utf8

Write-Host ""
Write-Host "Лог: $log"
exit $code

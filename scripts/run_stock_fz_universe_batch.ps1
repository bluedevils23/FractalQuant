param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForPid
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$inputRoot = 'D:\workspace\stockdata\stock-data\行情数据\stock_1min'
$dailyRoot = 'D:\workspace\stockdata\stock-data\行情数据\stock_daily.parquet'
$universeRoot = 'D:\workspace\stockdata\stock-factors\stock_universe'
$factorRoot = 'D:\workspace\stockdata\stock-factors'
$stageRoot = Join-Path $factorRoot '_tmp_fz_stage'
$generator = Join-Path $PSScriptRoot 'generate_fz_daily_factors.py'

function Test-UniverseOutput {
    param([string]$Universe, [string]$OutputRoot)

    $script = @'
import sys
from pathlib import Path
import pandas as pd

universe = {symbol.strip() for symbol in Path(sys.argv[1]).read_text(encoding="utf-8-sig").splitlines() if symbol.strip()}
output_root = Path(sys.argv[2])
files = {path.stem for path in output_root.glob("*.parquet")}
if files != universe:
    raise SystemExit(f"file set mismatch: expected={len(universe)}, actual={len(files)}")
for symbol in sorted(universe):
    frame = pd.read_parquet(output_root / f"{symbol}.parquet")
    if frame.shape[1] != 39 or frame.index.name != "factor_date" or not frame.index.is_unique:
        raise SystemExit(f"invalid output schema: {symbol}")
'@
    uv run python -c $script (Join-Path $universeRoot "$Universe.txt") $OutputRoot
    if ($LASTEXITCODE -ne 0) { throw "Validation failed for $Universe" }
}

Wait-Process -Id $WaitForPid
Test-UniverseOutput -Universe 'csi300' -OutputRoot (Join-Path $factorRoot 'stock_fz_daily_factors')

foreach ($universe in @('csi500', 'csi1000')) {
    $outputRoot = Join-Path $factorRoot "stock_fz_daily_factors_$universe"
    uv run python $generator --asset-type stock --input-root $inputRoot --daily-root $dailyRoot --symbols-file (Join-Path $universeRoot "$universe.txt") --date-from 2022-01-01 --output-root $outputRoot --stage-root $stageRoot --workers 8
    if ($LASTEXITCODE -ne 0) { throw "Generation failed for $universe" }
    Test-UniverseOutput -Universe $universe -OutputRoot $outputRoot
}

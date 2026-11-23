# Build the Toby 2 (version 2.0) Windows packages Toby-light and Toby-full from source.
# This is the recipe that built Toby-light-20261122-2.0-test.zip (sha256 e5eab25f...) and
# Toby-full-20261122-2.0-test.zip (sha256 62ec2bae...) from toby-dev commit d61cd73, on Windows Server 2022.
#
# Usage (PowerShell, as administrator, on a fresh Windows 10/11 or Server 2022 x64):
#   .\build_toby2_windows.ps1 -Source C:\src\toby-dev -Nets C:\src\toby2_nets [-Work C:\build] [-ReleaseDate 20261122-2.0]
#     -Source  a checkout of the toby-dev repository at the release commit
#     -Nets    a folder with the six Toby 2 files (their sha256 are checked below):
#              toby2_b.pt, toby2_lhead.pt, toby2_level.pt, toby2_level_cal.json,
#              toby2_standard.json (= nn/pass3/product/toby2_standard_c2gap.json),
#              toby2_accurate.json (= nn/pass3/product/toby2_accurate_c2gap.json)
#     toby2_level_cal.json can be made with: python nn/level_t2/lh/merge_ea_display.py <out>
# Output: <Work>\toby-dev\package_build\zips\Toby-{light,full}-<ReleaseDate>.zip and manifest_windows.json
param(
  [Parameter(Mandatory = $true)][string]$Source,
  [Parameter(Mandatory = $true)][string]$Nets,
  [string]$Work = 'C:\build',
  [string]$ReleaseDate = '20261122-2.0'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = 'Tls12'

# Pinned inputs
$PythonUrl = 'https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe'
$VcRedistUrl = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
$StockfishUrl = 'https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-windows-x86-64-universal.zip'
$StockfishSha = '45bc8e4969147db9c2eb533810637994619bff0eacc81ccfd9854394901bcbd0'   # stockfish-windows-x86-64-universal.exe
$Maia3Commit = '1e13597c42d4858b7cfd7cfdae01e297263364b2'   # https://github.com/CSSLab/maia3 (its Python code is frozen in; no Maia weights ship)
$Lc0Url = 'https://github.com/LeelaChessZero/lc0/releases/download/v0.32.1/lc0-v0.32.1-windows-gpu-nvidia-cuda12.zip'
$Lc0Sha = 'e32164ceb85ab128e6fe5e02d34cf17608d638335ae7d519f82a58c372e9666b'   # lc0.exe
$Bt3Url = 'https://storage.lczero.org/files/networks-contrib/BT3-768x15x24h-swa-2790000.pb.gz'
$Bt3Sha = 'e3067757d1fc2dfc66947b21d15ace0cedf4c54254fc1de83d77c378a3e8b8e1'
$NetSha = @{
  'toby2_b.pt'           = '3219cfc6f01bcd218441f67466475f27b05174fd462c40c8a4053d9753a1b281'
  'toby2_lhead.pt'       = 'd0c278bba6b87369f79e1cd8d89ba11227fdfd173734aeba52f001d2e78497da'
  'toby2_level.pt'       = '6e83ce50c7d688847f5ad8be5c855c2638b6b678004b617ec70e8a639bb7f6ce'
  'toby2_level_cal.json' = '1a104601c5b093ec021d3eb63c057d1206c8f36fc7f1e040765ef56f2efd1e22'
  'toby2_standard.json'  = '395670f85615c0f638c9284e1115a8579fac979d5d7b477a6257147a2f17a1a4'
  'toby2_accurate.json'  = 'b1b896986eb14f3bfcea20e77be7da1de10df2b3897138a3a62768e4424210be'
}
# Python packages, the versions frozen into the 2.0 packages
$Torch = '2.14.1'
$Pip = @('chess==1.11.2', 'numpy==2.5.5', 'scipy==1.18.1', 'huggingface_hub==2.4.0', 'pyinstaller==6.22.3', 'psutil')

function Sha($p) { (Get-FileHash $p -Algorithm SHA256).Hash.ToLower() }
function Need($p, $want) { $got = Sha $p; if ($got -ne $want) { throw "sha256 mismatch: $p is $got, expected $want" } }

New-Item -ItemType Directory -Force $Work, "$Work\dl", "$Work\t2", "$Work\lc0", "$Work\stockfish" | Out-Null

# 1. Toby 2 files
foreach ($f in $NetSha.Keys) { Copy-Item (Join-Path $Nets $f) "$Work\t2\$f" -Force; Need "$Work\t2\$f" $NetSha[$f] }

# 2. Python 3.12.8 and the VC++ 2015-2022 runtime
if (-not (Test-Path C:\Python312\python.exe)) {
  Invoke-WebRequest -UseBasicParsing $PythonUrl -OutFile "$Work\dl\python.exe"
  Start-Process -Wait "$Work\dl\python.exe" -ArgumentList '/quiet InstallAllUsers=1 TargetDir=C:\Python312 PrependPath=0 Include_test=0 Include_launcher=0'
}
Invoke-WebRequest -UseBasicParsing $VcRedistUrl -OutFile "$Work\dl\vc_redist.x64.exe"
Start-Process -Wait "$Work\dl\vc_redist.x64.exe" -ArgumentList '/install /quiet /norestart'

# 3. Stockfish 19 (official universal build)
Invoke-WebRequest -UseBasicParsing $StockfishUrl -OutFile "$Work\dl\sf.zip"
Expand-Archive -Force "$Work\dl\sf.zip" "$Work\dl\sfx"
Copy-Item (Get-ChildItem -Recurse "$Work\dl\sfx" -Filter 'stockfish*.exe' | Select-Object -First 1).FullName "$Work\stockfish\stockfish.exe" -Force
Need "$Work\stockfish\stockfish.exe" $StockfishSha

# 4. Maia-3 code at the pinned commit
Invoke-WebRequest -UseBasicParsing "https://codeload.github.com/CSSLab/maia3/zip/$Maia3Commit" -OutFile "$Work\dl\maia3.zip"
Expand-Archive -Force "$Work\dl\maia3.zip" "$Work\dl\maia3x"
Remove-Item -Recurse -Force "$Work\maia3" -ErrorAction SilentlyContinue
Move-Item (Get-ChildItem "$Work\dl\maia3x" -Directory | Select-Object -First 1).FullName "$Work\maia3"

# 5. lc0 v0.32.1 (official CUDA 12 build) and the BT3 network, for Toby-full (Accurate)
Invoke-WebRequest -UseBasicParsing $Lc0Url -OutFile "$Work\dl\lc0.zip"
Expand-Archive -Force "$Work\dl\lc0.zip" "$Work\dl\lc0x"
$exe = Get-ChildItem -Recurse "$Work\dl\lc0x" -Filter lc0.exe | Select-Object -First 1
Copy-Item "$($exe.DirectoryName)\*" "$Work\lc0" -Recurse -Force
Need "$Work\lc0\lc0.exe" $Lc0Sha
Invoke-WebRequest -UseBasicParsing $Bt3Url -OutFile "$Work\lc0\BT3-768x15x24h-swa-2790000.pb.gz"
Need "$Work\lc0\BT3-768x15x24h-swa-2790000.pb.gz" $Bt3Sha

# 6. Two virtual environments: PyTorch CPU (toby.exe of Toby-light) and CUDA 12.6 (toby.exe of Toby-full)
foreach ($v in 'cpu', 'cuda') {
  & C:\Python312\python.exe -m venv "$Work\venv_$v"
  $vp = "$Work\venv_$v\Scripts\python.exe"
  & $vp -m pip install --quiet --disable-pip-version-check --upgrade pip
  $idx = if ($v -eq 'cpu') { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu126' }
  & $vp -m pip install --quiet --disable-pip-version-check "torch==$Torch" --index-url $idx
  & $vp -m pip install --quiet --disable-pip-version-check @Pip
  & $vp -c "import torch, PyInstaller, chess, numpy, scipy, huggingface_hub; print('$v', torch.__version__, PyInstaller.__version__, chess.__version__, numpy.__version__, scipy.__version__, huggingface_hub.__version__)"
}

# 7. Build: public release tree, freeze (CPU, then CUDA), assemble, zip. Maia-3 weights are not shipped (TOBY_SHIP_MAIA unset).
Remove-Item -Recurse -Force "$Work\toby-dev" -ErrorAction SilentlyContinue
Copy-Item $Source "$Work\toby-dev" -Recurse
$env:TOBY_MAIA_REPO_WIN = "$Work\maia3"
$env:TOBY_LC0_DIR_WIN = "$Work\lc0"
$env:TOBY_STOCKFISH_WIN = "$Work\stockfish\stockfish.exe"
$env:TOBY2_WEIGHTS_WIN = "$Work\t2\toby2_b.pt"   # the other five Toby 2 files are read from the same folder
$env:TOBY_RELEASE_DATE = $ReleaseDate
Remove-Item Env:TOBY_SHIP_MAIA, Env:TOBY_MAIA_WEIGHTS_WIN -ErrorAction SilentlyContinue
$ErrorActionPreference = 'Continue'   # PyInstaller writes its log to stderr
Set-Location "$Work\toby-dev"
$py = "$Work\venv_cpu\Scripts\python.exe"
& $py make_public_release.py
& $py build_package.py freeze *> "$Work\freeze_cpu.log"; if ($LASTEXITCODE) { throw "freeze cpu failed, see $Work\freeze_cpu.log" }
& "$Work\venv_cuda\Scripts\python.exe" build_package.py freeze cuda *> "$Work\freeze_cuda.log"; if ($LASTEXITCODE) { throw "freeze cuda failed, see $Work\freeze_cuda.log" }
& $py build_package.py assemble; if ($LASTEXITCODE) { throw 'assemble failed' }
& $py build_package.py zip; if ($LASTEXITCODE) { throw 'zip failed' }
Get-ChildItem package_build\zips | ForEach-Object { '{0}  {1}  {2}' -f (Sha $_.FullName), $_.Length, $_.Name }

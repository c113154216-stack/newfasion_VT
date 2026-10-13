# 衣物生成環境一鍵安裝（對應第四週週報「所需工具」）
#
# 會做的事（已經有的會跳過，可以重複執行）：
#   1. 抓 ChatGarment、GarmentCodeRC、DeepBump 原始碼到 third_party/（固定在測試過的版本）
#   2. 在 ChatGarment 建 assets 連結，指向 GarmentCodeRC/assets（作者 README 的步驟 6）
#   3. 建 .venv（GarmentCode 環境）並安裝 tools/chatgarment/requirements.txt
#   4. 建 .venv-chatgarment（ChatGarment 環境，PyTorch CUDA 12.8）並安裝 requirements-ai.txt
#   5. 檢查不能自動下載的檔案（ChatGarment 權重、SMPL-X 權重），缺的話告訴你去哪裡拿
#
# 用法（在 newfashion_VT 根目錄，PowerShell）：
#   powershell -ExecutionPolicy Bypass -File tools\setup\setup.ps1
# 需要先裝好：Git、Python 3.12（指令 py -3.12 或 python 要能用）

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# 找 Python 3.12
$Py = $null
foreach ($cand in @("py -3.12", "python")) {
    try {
        $ver = Invoke-Expression "$cand --version" 2>$null
        if ($ver -match "3\.12") { $Py = $cand; break }
    } catch {}
}
if (-not $Py) { throw "找不到 Python 3.12，請先安裝（https://www.python.org/downloads/）" }
Write-Host "Python：$Py ($ver)"

# ---- 1. 第三方原始碼 ----
Step "1. 第三方原始碼（third_party/）"
$Repos = @(
    @{ Name = "ChatGarment";   Url = "https://github.com/biansy000/ChatGarment";   Commit = "798c34ac04319668aaa0e3a3a02d44284fbd70ec" },
    @{ Name = "GarmentCodeRC"; Url = "https://github.com/biansy000/GarmentCodeRC"; Commit = "e3d4476d6f9bcb0af5ee0eccd18b0433e3df398a" },
    # 法線貼圖比較用（GPL-3：只當外部工具呼叫，不併進專案程式碼）
    @{ Name = "DeepBump";      Url = "https://github.com/HugoTini/DeepBump";      Commit = "fad19ba87daed12b1d0410a57e74f3d79e82f78d" }
)
New-Item -ItemType Directory -Force "third_party" | Out-Null
foreach ($r in $Repos) {
    $dir = "third_party\$($r.Name)"
    if (Test-Path "$dir\.git") {
        Write-Host "$($r.Name)：已存在，跳過"
    } else {
        git clone $r.Url $dir
        git -C $dir checkout $r.Commit
    }
}

# ---- 2. assets 連結 ----
Step "2. ChatGarment\assets → GarmentCodeRC\assets"
$link = "third_party\ChatGarment\assets"
if (Test-Path $link) {
    Write-Host "連結已存在，跳過"
} else {
    # junction 不需要系統管理員權限
    cmd /c mklink /J "$link" "$Root\third_party\GarmentCodeRC\assets" | Out-Null
    Write-Host "已建立"
}

# ---- 3. GarmentCode 環境 ----
Step "3. .venv（GarmentCode 環境）"
if (-not (Test-Path ".venv\Scripts\python.exe")) { Invoke-Expression "$Py -m venv .venv" }
& .venv\Scripts\python -m pip install --upgrade pip -q
& .venv\Scripts\python -m pip install -r tools\chatgarment\requirements.txt

# ---- 4. ChatGarment 環境 ----
Step "4. .venv-chatgarment（ChatGarment 環境，約 3～4 GB）"
if (-not (Test-Path ".venv-chatgarment\Scripts\python.exe")) { Invoke-Expression "$Py -m venv .venv-chatgarment" }
& .venv-chatgarment\Scripts\python -m pip install --upgrade pip -q
& .venv-chatgarment\Scripts\python -m pip install -r tools\chatgarment\requirements-ai.txt

# ---- 5. 不能自動下載的檔案 ----
Step "5. 需要手動放置的檔案"
$Manual = @(
    @{ Path = "third_party\ChatGarment\checkpoints\try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final\pytorch_model.bin";
       What = "ChatGarment 權重（14 GB）";
       How  = "到 ChatGarment README 的 Pretrained weights 連結（SharePoint，要用瀏覽器下載）" },
    @{ Path = "legacy\geometry_physics\assets\smplx\SMPLX_NEUTRAL.npz";
       What = "SMPL-X 權重";
       How  = "到 https://smpl-x.is.tue.mpg.de/ 註冊下載（授權禁止散布，不放 GitHub）" }
)
$missing = 0
foreach ($m in $Manual) {
    if (Test-Path $m.Path) { Write-Host "✔ $($m.What)" -ForegroundColor Green }
    else { Write-Host "✘ $($m.What)：缺少 $($m.Path)`n    取得方式：$($m.How)" -ForegroundColor Yellow; $missing++ }
}

Step "完成"
Write-Host "接著執行環境檢查：.venv\Scripts\python tools\setup\check_env.py"
if ($missing -gt 0) { Write-Host "還有 $missing 個檔案要手動放置（見上方）" -ForegroundColor Yellow }

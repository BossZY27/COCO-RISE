# เปิดระบบ RISE บนเครื่องนี้
#   .\run.ps1            เปิดเซิร์ฟเวอร์
#   .\run.ps1 -Reindex   สร้างเวกเตอร์ใหม่ก่อนเปิด (ใช้เมื่อเพิ่งอนุมัติเอกสารใหม่)
#   .\run.ps1 -Check     ตรวจความพร้อมอย่างเดียว ไม่เปิดเซิร์ฟเวอร์
#   .\run.ps1 -FriendDemoIp 192.168.1.25   เปิดเดโมให้เพื่อนใน LAN เดียวกัน
param(
    [switch]$Reindex,
    [switch]$Check,
    [ValidateRange(1, 65535)]
    [int]$Port = 8077,
    [string]$FriendDemoIp = "",
    [switch]$FriendDemoAudit
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$serverHost = "127.0.0.1"
$friendMode = -not [string]::IsNullOrWhiteSpace($FriendDemoIp)
$savedEnvironment = @{}
$dotenvEnvironment = @{}

function Set-DemoEnvironment([string]$Name, [string]$Value) {
    $script:savedEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, "Process")
    [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
}

function Restore-ProcessEnvironment([string]$Name, $Value) {
    if ($null -eq $Value) {
        Remove-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
        return
    }
    [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
}

function Restore-DemoEnvironment {
    foreach ($entry in $script:savedEnvironment.GetEnumerator()) {
        Restore-ProcessEnvironment $entry.Key $entry.Value
    }
}

function Restore-DotEnvEnvironment {
    foreach ($entry in $script:dotenvEnvironment.GetEnumerator()) {
        Restore-ProcessEnvironment $entry.Key $entry.Value
    }
}

function Import-DotEnv([string]$Path) {
    foreach ($rawLine in Get-Content -LiteralPath $Path) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#")) { continue }
        $parts = $line -split "=", 2
        if ($parts.Count -ne 2 -or $parts[0] -notmatch "^[A-Za-z_][A-Za-z0-9_]*$") { continue }
        $name = $parts[0]
        $value = $parts[1].Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        # Explicit process values win over .env. Friend-demo overrides are applied below.
        if ($null -eq [Environment]::GetEnvironmentVariable($name, "Process")) {
            $script:dotenvEnvironment[$name] = $null
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

try {
if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
    Write-Host "  สร้าง .env จากตัวอย่างแล้ว เปิดไฟล์ไปตั้ง RISE_ADMIN_TOKEN ก่อนใช้หน้า /admin" -ForegroundColor Yellow
}
Import-DotEnv ".env"

if ($friendMode) {
    try {
        $address = [System.Net.IPAddress]::Parse($FriendDemoIp)
    } catch {
        throw "FriendDemoIp ต้องเป็น IP ของเครื่องนี้ เช่น 192.168.1.25"
    }
    if ($address.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork) {
        throw "โหมดเดโมเพื่อนรองรับ private IPv4 ของ LAN เท่านั้น"
    }
    $bytes = $address.GetAddressBytes()
    $privateAddress = $bytes[0] -eq 10 -or
        ($bytes[0] -eq 172 -and $bytes[1] -ge 16 -and $bytes[1] -le 31) -or
        ($bytes[0] -eq 192 -and $bytes[1] -eq 168)
    if (-not $privateAddress) {
        throw "FriendDemoIp ต้องเป็น private LAN IP ช่วง 10.x, 172.16-31.x หรือ 192.168.x"
    }
    $localAddresses = [System.Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() |
        ForEach-Object { $_.GetIPProperties().UnicastAddresses } |
        ForEach-Object { $_.Address.IPAddressToString }
    if ($FriendDemoIp -notin $localAddresses) {
        throw "ไม่พบ $FriendDemoIp บน network adapter ของเครื่องนี้ ตรวจ IP แล้วลองใหม่"
    }

    $serverHost = $FriendDemoIp
    Set-DemoEnvironment "RISE_TRUSTED_HOSTS" "$FriendDemoIp,127.0.0.1,localhost"
    Set-DemoEnvironment "RISE_ADMIN_ENABLED" "off"
    Set-DemoEnvironment "RISE_ADMIN_TOKEN" ""
    Set-DemoEnvironment "RISE_EXPOSE_DOCS" "off"
    Set-DemoEnvironment "RISE_PROFILE_SOURCE" "demo"
    Set-DemoEnvironment "RISE_TRUST_PROXY_USER" "off"
    Set-DemoEnvironment "RISE_LLM_CONCURRENCY" "1"
    Set-DemoEnvironment "RISE_AUDIT" $(if ($FriendDemoAudit) { "on" } else { "off" })
}

if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("RISE_ANSWER_ENABLED", "Process"))) {
    Set-DemoEnvironment "RISE_ANSWER_ENABLED" "true"
}
if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("RISE_SLOT_LLM_FALLBACK", "Process"))) {
    Set-DemoEnvironment "RISE_SLOT_LLM_FALLBACK" "on"
}

Write-Host "== ตรวจความพร้อม ==" -ForegroundColor Cyan

$base = [Environment]::GetEnvironmentVariable("RISE_LLM_BASE", "Process")
if ([string]::IsNullOrWhiteSpace($base)) { $base = "http://127.0.0.1:11434" }
$base = $base.TrimEnd("/")
try {
    $null = Invoke-WebRequest -UseBasicParsing -Uri "$base/api/version" -TimeoutSec 3
    Write-Host "  Ollama: ตอบรับ" -ForegroundColor Green
} catch {
    Write-Host "  Ollama: ติดต่อไม่ได้ที่ $base" -ForegroundColor Red
    Write-Host "  ระบบยังตอบได้จากข้อมูลที่ตรวจแล้ว และจะใช้ trigram แทน semantic embedding" -ForegroundColor Yellow
}

$ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
$embedModel = [Environment]::GetEnvironmentVariable("RISE_EMBED_MODEL", "Process")
if ([string]::IsNullOrWhiteSpace($embedModel)) { $embedModel = "bge-m3:latest" }
$embedModelForList = if ($embedModel.Contains(":")) { $embedModel } else { "${embedModel}:latest" }
$modelNames = @()
if ($null -ne $ollamaCommand) {
    $modelNames = @(
        & ollama list 2>$null |
            Select-Object -Skip 1 |
            ForEach-Object { ($_.Trim() -split "\s+")[0] } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
} else {
    Write-Host "  ไม่พบคำสั่ง ollama: ข้ามการตรวจโมเดลเสริม" -ForegroundColor Yellow
}
if ($embedModelForList -in $modelNames) {
    Write-Host "  โมเดล ${embedModel}: มีแล้ว" -ForegroundColor Green
} else {
    Write-Host "  โมเดลเสริม ${embedModel}: ยังไม่พร้อม (ระบบยังใช้ trigram ได้)" -ForegroundColor Yellow
}

if ($Reindex) {
    Write-Host "== สร้างเวกเตอร์ใหม่ ==" -ForegroundColor Cyan
    python reindex.py
    if ($LASTEXITCODE -ne 0) { throw "reindex.py สร้างเวกเตอร์ไม่สำเร็จ (exit code $LASTEXITCODE)" }
}

if ($Check) {
    python check.py
    if ($LASTEXITCODE -ne 0) { throw "check.py ตรวจพบข้อผิดพลาด" }
    return
}

Write-Host "== เปิดเซิร์ฟเวอร์ที่ http://${serverHost}:$Port ==" -ForegroundColor Cyan
if ($friendMode) {
    Write-Host "   โหมดเพื่อน: ใช้ persona สมมติ, ปิด /admin, ปิด API docs และใช้ worker เดียว" -ForegroundColor Green
    Write-Host "   audit: $(if ($FriendDemoAudit) { 'เปิดตามคำสั่ง' } else { 'ปิด' })" -ForegroundColor Gray
    Write-Host "   สคริปต์นี้ไม่เปิด Windows Firewall และไม่สร้าง tunnel" -ForegroundColor Yellow
    Write-Host "   หากเพื่อนเข้าไม่ได้ ให้เจ้าของเครื่องเปิดพอร์ต $Port เฉพาะเครือข่าย Private ด้วยตนเอง และปิดกฎหลังจบเดโม" -ForegroundColor Yellow
} else {
    Write-Host "   หน้าถามตอบ /   หน้าตรวจข้อมูล /admin" -ForegroundColor Gray
}
# Keep request.client as the real TCP peer. The app validates that exact IP
# before trusting X-User-Id from a configured reverse proxy.
python -m uvicorn app.main:app --host $serverHost --port $Port --workers 1 --no-proxy-headers
if ($LASTEXITCODE -ne 0) { throw "เซิร์ฟเวอร์หยุดด้วย exit code $LASTEXITCODE" }
} finally {
    try {
        Restore-DemoEnvironment
    } finally {
        Restore-DotEnvEnvironment
    }
}

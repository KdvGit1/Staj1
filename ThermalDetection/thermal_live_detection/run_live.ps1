param(
    [string]$CameraIp = $env:THERMAL_CAMERA_IP,
    [string]$CameraUser = $(if ($env:THERMAL_CAMERA_USER) {
        $env:THERMAL_CAMERA_USER
    } else {
        "admin"
    }),
    [ValidateSet("201", "202")]
    [string]$Channel = "202",
    [ValidateSet("edsr", "bicubic", "source")]
    [string]$DetectorInput = "edsr",
    [ValidateSet("off", "manual", "interval", "detections", "hybrid")]
    [string]$CaptureMode = "manual"
)

if (-not $CameraIp) {
    throw "CameraIp veya THERMAL_CAMERA_IP gerekli."
}

$securePassword = Read-Host "Kamera parolası" -AsSecureString
$passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR(
    $securePassword
)
try {
    $env:THERMAL_CAMERA_PASSWORD = (
        [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
    )
    python -m thermal_live_detection.app `
        --camera-ip $CameraIp `
        --camera-user $CameraUser `
        --channel $Channel `
        --detector-input $DetectorInput `
        --capture-mode $CaptureMode
}
finally {
    $env:THERMAL_CAMERA_PASSWORD = $null
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
}

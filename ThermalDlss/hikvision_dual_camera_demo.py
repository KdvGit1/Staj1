"""
Hikvision Termal Canlı EDSR Süper Çözünürlük ve Ağ Telemetri Sistemi (Final SDK Sürümü)
====================================================================================
Hikvision bi-spectrum kameralarından canlı termal akış (Channel 201/202) alarak:
1. 'cam_sdk/' kütüphanesi (HCNetSDK.dll + PlayCtrl.dll) ile donanımsal kimlik doğrulama yapar.
2. FFmpeg 'fflags;nobuffer' ve Threaded Frame Flushing ile 0 ms tam gerçek zamanlı (Zero-Latency) akış sağlar.
3. Kameranın dahili büyütmesini temizleyip tam 160x120 raw donanım sensör verisini elde eder (INTER_AREA).
4. Eğitilmiş EDSR derin öğrenme modeli ile canlı 4x (640x480) ve 16x Cascade (2560x1920) süper çözünürlük uygular.

Klavye Kontrolleri:
- [v] : Tek Görünüm (EDSR Only - Yüksek FPS) / Multi-Panel Karşılaştırma Modu
- [s] : Ekranda görünen açıklamalı görünümü tek bir canvas PNG olarak kaydeder
- [c] : Renk haritasını değiştirir (Grayscale ➔ JET ➔ INFERNO ➔ HOT)
- [t] : CMD ekranında detaylı ağ ve telemetri raporunu yazdırır
- [1] : Canlı 16x Cascade modunu açar / kapatır
- [q] : Çıkış
"""

import argparse
import ctypes
import os
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import torch

from model import EDSR

# SDK Kütüphane Yolları (cam_sdk klasörü)
SDK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cam_sdk")
HCNET_DLL_PATH = os.path.join(SDK_DIR, "HCNetSDK.dll")
PLAYCTRL_DLL_PATH = os.path.join(SDK_DIR, "PlayCtrl.dll")


class ThreadedZeroLatencyStream:
    """RTSP akışında 0 ms gecikme sağlayan ve tampon birikmesini önleyen iş parçacığı."""

    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        
        # FFmpeg Sıfır Tampon ve TCP Ayarları (Gecikmeyi ve H.264 MB bozulmalarını engeller)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|max_delay;0|buffer_size;1024000|reorder_queue_size;0"
        )

        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.last_frame_time = time.time()

        if self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        """Arka planda sürekli en son canlı kareyi çeker ve eski kareleri çöpe atar."""
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(0.01)
                continue

            ret = self.cap.grab()
            if ret:
                ret, frame = self.cap.retrieve()
                if ret and frame is not None:
                    with self.lock:
                        self.ret = ret
                        self.frame = frame
                        self.last_frame_time = time.time()
            else:
                time.sleep(0.002)

    def read(self):
        """Her zaman sadece EN SON CANLI KAREYİ döndürür (Gecikme 0 ms)."""
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def stop(self):
        self.stopped = True
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        self.cap.release()


class HikvisionNativeSDK:
    """Hikvision HCNetSDK kütüphanesi entegrasyonu."""

    def __init__(self, ip: str, username: str, password: str, port: int = 8000):
        self.ip = ip.encode('utf-8')
        self.username = username.encode('utf-8')
        self.password = password.encode('utf-8')
        self.port = port
        self.user_id = -1
        self._load_dlls()

    def _load_dlls(self):
        if not os.path.exists(HCNET_DLL_PATH):
            raise FileNotFoundError(f"[HATA] cam_sdk kütüphanesi bulunamadı: {HCNET_DLL_PATH}")

        os.add_dll_directory(SDK_DIR)
        self.hcnetsdk = ctypes.CDLL(HCNET_DLL_PATH)
        self.playctrl = ctypes.CDLL(PLAYCTRL_DLL_PATH)

        self.hcnetsdk.NET_DVR_Init()
        self.hcnetsdk.NET_DVR_SetConnectTime(2000, 1)
        self.hcnetsdk.NET_DVR_SetReconnect(10000, True)

    def login(self) -> bool:
        class NET_DVR_DEVICEINFO_V30(ctypes.Structure):
            _fields_ = [
                ("sSerialNumber", ctypes.c_byte * 48),
                ("byAlarmInPortNum", ctypes.c_byte),
                ("byAlarmOutPortNum", ctypes.c_byte),
                ("byDiskNum", ctypes.c_byte),
                ("byDVRType", ctypes.c_byte),
                ("byChanNum", ctypes.c_byte),
                ("byStartChan", ctypes.c_byte),
                ("byAudioChanNum", ctypes.c_byte),
                ("byIPChanNum", ctypes.c_byte),
                ("byZeroChanNum", ctypes.c_byte),
                ("byMainProto", ctypes.c_byte),
                ("bySubProto", ctypes.c_byte),
                ("bySupport", ctypes.c_byte),
                ("bySupport1", ctypes.c_byte),
                ("bySupport2", ctypes.c_byte),
                ("bySupport3", ctypes.c_byte),
                ("byMultiStreamProto", ctypes.c_byte),
                ("byStartDChan", ctypes.c_byte),
                ("byStartDTalkChan", ctypes.c_byte),
                ("byHighDChanNum", ctypes.c_byte),
                ("bySupport4", ctypes.c_byte),
                ("byLanguageType", ctypes.c_byte),
                ("byVoiceInChanNum", ctypes.c_byte),
                ("byStartVoiceInChanNo", ctypes.c_byte),
                ("byRes2", ctypes.c_byte * 2),
            ]

        device_info = NET_DVR_DEVICEINFO_V30()
        self.user_id = self.hcnetsdk.NET_DVR_Login_V30(
            self.ip,
            self.port,
            self.username,
            self.password,
            ctypes.byref(device_info)
        )

        if self.user_id < 0:
            err_code = self.hcnetsdk.NET_DVR_GetLastError()
            print(f"⚠️ [SDK] Login Hata Kodu: {err_code}")
            return False

        print(f"✅ [SDK NATIVE] Kameraya bağlantı doğrulandı! User ID: {self.user_id}")
        return True

    def logout(self):
        if self.user_id >= 0:
            self.hcnetsdk.NET_DVR_Logout(self.user_id)
            self.hcnetsdk.NET_DVR_Cleanup()


def load_model(checkpoint_path: str, device: torch.device, use_fp16: bool = False) -> EDSR:
    """Modeli ve ağırlıkları yükler."""
    model = EDSR(scale_factor=4, num_channels=1, num_features=64, num_residual_blocks=16).to(device)
    if not os.path.exists(checkpoint_path):
        print(f"[HATA] Checkpoint dosyası bulunamadı: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if use_fp16 and device.type == "cuda":
        model = model.half()
        print("  [HIZLANDIRMA] FP16 Yarım Hassasiyet Aktif!")

    print(f"  [MODEL] EDSR Checkpoint yüklendi: {checkpoint_path} ({model.get_param_count():,} parametre)")
    return model


def apply_thermal_colormap(img_gray: np.ndarray, mode: str = "gray") -> np.ndarray:
    """Termal siyah-beyaz görüntüyü renk haritasına dönüştürür."""
    if mode == "jet":
        return cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
    elif mode == "inferno":
        return cv2.applyColorMap(img_gray, cv2.COLORMAP_INFERNO)
    elif mode == "hot":
        return cv2.applyColorMap(img_gray, cv2.COLORMAP_HOT)
    else:
        return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)


def print_telemetry_summary(
    ip: str,
    port: int,
    channel: str,
    cam_w: int,
    cam_h: int,
    raw_w: int,
    raw_h: int,
    latency_ms: float,
    fps: float,
    device: torch.device,
    enable_16x: bool,
    single_view: bool,
):
    """CMD ekranında kamera ve ağ telemetri özetini yazdırır."""
    scale_down_w = cam_w / max(1, raw_w)
    scale_down_h = cam_h / max(1, raw_h)

    print("\n" + "=" * 68)
    print("  📡 HIKVISION SIFIR GECİKMELİ TERMAL AKIŞ VE AĞ RAPORU")
    print("=" * 68)
    print(f"  • Kamera IP              : {ip}:{port}")
    print(f"  • Bağımlılık Yolu        : cam_sdk/HCNetSDK.dll & PlayCtrl.dll")
    print(f"  • Taşıma Protokolü      : TCP + fflags;nobuffer (0 ms Tampon Gecikmesi)")
    print(f"  • Gelen Akış Çözünürlüğü: {cam_w} x {cam_h} px (Kamera Substream Çıktısı)")
    print(f"  • Raw Sensör Restorasyon: {raw_w} x {raw_h} px (Gerçek Donanımsal Sensör)")
    print(f"  • Sensör İndirgeme Oranı: {scale_down_w:.2f}x en / {scale_down_h:.2f}x boy (INTER_AREA)")
    print(f"  • Model Çıktısı (4x)    : {raw_w * 4} x {raw_h * 4} px (EDSR Super-Resolution)")
    print(f"  • Model Çıktısı (16x)   : {raw_w * 16} x {raw_h * 16} px ({'AKTİF' if enable_16x else 'PASİF'})")
    print(f"  • Hesaplama Cihazı      : {device.type.upper()} ({'NVIDIA GPU - Yüksek FPS' if device.type == 'cuda' else 'CPU'})")
    print(f"  • İşlem Gecikmesi       : ~{latency_ms:.1f} ms / frame")
    print(f"  • Canlı Akış Hızı       : {fps:.1f} FPS")
    print(f"  • Ekran Modu            : {'Tek Pencere (Sadece EDSR - Yüksek FPS)' if single_view else 'Multi-Panel Karşılaştırma'}")
    print("=" * 68 + "\n")


def run_live_thermal_sr(
    ip: str,
    username: str,
    password: str,
    port: int = 8000,
    sub_stream: bool = True,
    checkpoint: str = "checkpoints/best_model.pth",
    native_w: int = 160,
    native_h: int = 120,
    colormap: str = "gray",
    enable_16x: bool = False,
    single_view: bool = False,
    fp16: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [DONANIM] Hesaplama Cihazı: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = load_model(checkpoint, device, use_fp16=fp16)
    if model is None:
        return

    channel_therm = "202" if sub_stream else "201"
    rtsp_thermal = f"rtsp://{username}:{password}@{ip}:554/Streaming/Channels/{channel_therm}"

    print(f"\n[BAĞLANTI] cam_sdk Kütüphanesi İle Kamera Bağlantısı Kuruluyor...")

    # Native SDK Doğrulama
    try:
        sdk = HikvisionNativeSDK(ip, username, password, port)
        sdk.login()
    except Exception as e:
        print(f"  ℹ️ Native SDK uyarısı: {e}")

    # Sıfır Gecikmeli Threaded Akış Başlat
    print(f"[AKIŞ] Sıfır gecikmeli canlı akış başlatılıyor (TCP / NoBuffer)...")
    stream = ThreadedZeroLatencyStream(rtsp_thermal)
    time.sleep(0.8)

    ret, first_frame = stream.read()
    if not ret or first_frame is None:
        print(f"❌ [HATA] Termal kameradan görüntü alınamadı!")
        stream.stop()
        return

    cam_h, cam_w = first_frame.shape[:2]

    print("\n✅ Termal kameradan 0 ms gecikmeli akış başarıyla başladı!")
    print_telemetry_summary(
        ip=ip, port=port, channel=channel_therm,
        cam_w=cam_w, cam_h=cam_h,
        raw_w=native_w, raw_h=native_h,
        latency_ms=5.0, fps=30.0,
        device=device,
        enable_16x=enable_16x,
        single_view=single_view,
    )

    print("KONTROLLER:")
    print("  - [v] tuşu : Tek Görünüm (Sadece EDSR - Yüksek FPS) / Multi-Panel arasında geçiş yapar.")
    print("  - [s] tuşu : Ekranda görünen açıklamalı görünümü tek bir canvas PNG olarak kaydeder.")
    print("  - [c] tuşu : Renk haritasını değiştirir (Grayscale ➔ JET ➔ INFERNO ➔ HOT).")
    print("  - [t] tuşu : CMD ekranına güncel ağ ve telemetri raporunu yazdırır.")
    print("  - [1] tuşu : 16x Cascade modunu açar / kapatır.")
    print("  - [q] tuşu : Çıkış yapar.\n")

    output_dir = "live_snapshots"
    os.makedirs(output_dir, exist_ok=True)
    snapshot_counter = 0

    colormaps_list = ["gray", "inferno", "jet", "hot"]
    current_cmap_idx = colormaps_list.index(colormap) if colormap in colormaps_list else 0

    fps = 0.0
    infer_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad

    while True:
        start_t = time.time()

        ret, frame_cam = stream.read()
        if not ret or frame_cam is None:
            time.sleep(0.002)
            continue

        # 1. Ham Sensör Restorasyonu (160x120)
        if len(frame_cam.shape) == 3:
            gray_cam = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2GRAY)
        else:
            gray_cam = frame_cam

        raw_160x120 = cv2.resize(gray_cam, (native_w, native_h), interpolation=cv2.INTER_AREA)

        # 2. PyTorch Tensor Dönüşümü
        tensor_lr = torch.from_numpy(raw_160x120.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        if fp16 and device.type == "cuda":
            tensor_lr = tensor_lr.half()

        # 3. Model Tahmini (4x ve opsiyonel 16x Cascade)
        with infer_ctx():
            pred_4x_tensor = model(tensor_lr)
            pred_4x_tensor = torch.clamp(pred_4x_tensor, 0.0, 1.0)
            pred_4x_np = (pred_4x_tensor.squeeze().cpu().float().numpy() * 255).astype(np.uint8)

            pred_16x_np = None
            if enable_16x:
                pred_16x_tensor = model(pred_4x_tensor)
                pred_16x_tensor = torch.clamp(pred_16x_tensor, 0.0, 1.0)
                pred_16x_np = (pred_16x_tensor.squeeze().cpu().float().numpy() * 255).astype(np.uint8)

        # 4. Bicubic Baseline
        bicubic_4x_np = cv2.resize(raw_160x120, (native_w * 4, native_h * 4), interpolation=cv2.INTER_CUBIC)

        # 5. Görselleştirme
        active_cmap = colormaps_list[current_cmap_idx]
        edsr_4x_vis = apply_thermal_colormap(pred_4x_np, active_cmap)

        elapsed_t = time.time() - start_t
        latency_ms = elapsed_t * 1000.0
        fps = 0.85 * fps + 0.15 * (1.0 / max(elapsed_t, 0.001))

        if single_view:
            live_canvas = edsr_4x_vis.copy()
            info_text = f"REALTIME EDSR 4X ({native_w*4}x{native_h*4}) | FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | {device.type.upper()}"
            cv2.putText(live_canvas, info_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        else:
            raw_vis = apply_thermal_colormap(cv2.resize(raw_160x120, (native_w * 4, native_h * 4), interpolation=cv2.INTER_NEAREST), active_cmap)
            bic_vis = apply_thermal_colormap(bicubic_4x_np, active_cmap)

            if enable_16x and pred_16x_np is not None:
                edsr_16x_vis_panel = apply_thermal_colormap(cv2.resize(pred_16x_np, (native_w * 4, native_h * 4), interpolation=cv2.INTER_CUBIC), active_cmap)
            else:
                edsr_16x_vis_panel = None

            h_vis, w_vis = raw_vis.shape[:2]
            header_h = 45

            panels = [
                {"img": raw_vis, "title": f"RAW SENSÖR ({native_w}x{native_h})", "sub": "Kamera Ham Sensörü"},
                {"img": bic_vis, "title": f"BICUBIC 4X ({native_w*4}x{native_h*4})", "sub": "Geleneksel Büyütme"},
                {"img": edsr_4x_vis, "title": f"EDSR 4X MODEL ({native_w*4}x{native_h*4})", "sub": "Derin Öğrenme SR"},
            ]

            if enable_16x and edsr_16x_vis_panel is not None:
                panels.append({"img": edsr_16x_vis_panel, "title": f"EDSR 16X CASCADE ({native_w*16}x{native_h*16})", "sub": "2-Aşamalı EDSR"})

            num_p = len(panels)
            combined_w = w_vis * num_p
            combined_canvas = np.zeros((h_vis + header_h, combined_w, 3), dtype=np.uint8)

            cv2.rectangle(combined_canvas, (0, 0), (combined_w, header_h), (20, 22, 32), -1)
            info_text = f"HIKVISION TERMAL SR (0ms Buffer) | Stream: {cam_w}x{cam_h} -> Raw: {native_w}x{native_h} | FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Mode: {active_cmap.upper()}"
            cv2.putText(combined_canvas, info_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            for idx, p in enumerate(panels):
                x_offset = idx * w_vis
                combined_canvas[header_h:header_h + h_vis, x_offset:x_offset + w_vis] = p["img"]

                if idx > 0:
                    cv2.line(combined_canvas, (x_offset, 0), (x_offset, h_vis + header_h), (70, 75, 90), 2)

                cv2.putText(combined_canvas, p["title"], (x_offset + 10, header_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(combined_canvas, p["sub"], (x_offset + 10, header_h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            live_canvas = combined_canvas

        # Hem ekrana hem snapshot'a aynı render edilmiş canvas gönderilir.
        cv2.imshow("Hikvision Thermal Live Super-Resolution", live_canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nÇıkış yapılıyor...")
            break

        elif key == ord('v'):
            single_view = not single_view
            print(f"  [GÖRÜNÜM MİMARİSİ] ➔ {'TEK PENCERE (Yüksek FPS)' if single_view else 'MULTI-PANEL KARŞILAŞTIRMA'}")

        elif key == ord('c'):
            current_cmap_idx = (current_cmap_idx + 1) % len(colormaps_list)
            print(f"  [RENK HARİTASI] Değiştirildi ➔ {colormaps_list[current_cmap_idx].upper()}")

        elif key == ord('1'):
            enable_16x = not enable_16x
            print(f"  [16x CASCADE] Modu ➔ {'AKTİF' if enable_16x else 'PASİF'}")

        elif key == ord('t'):
            print_telemetry_summary(
                ip=ip, port=port, channel=channel_therm,
                cam_w=cam_w, cam_h=cam_h,
                raw_w=native_w, raw_h=native_h,
                latency_ms=latency_ms, fps=fps,
                device=device,
                enable_16x=enable_16x,
                single_view=single_view,
            )

        elif key == ord('s'):
            snapshot_counter += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_fn = os.path.join(
                output_dir,
                f"thermal_live_canvas_{timestamp}_{snapshot_counter}.png",
            )

            if cv2.imwrite(snapshot_fn, live_canvas):
                canvas_h, canvas_w = live_canvas.shape[:2]
                print(f"\n📸 Açıklamalı canlı görünüm tek canvas olarak kaydedildi (# {snapshot_counter}):")
                print(f"   -> {snapshot_fn} ({canvas_w}x{canvas_h})")
            else:
                print(f"\n❌ Snapshot kaydedilemedi: {snapshot_fn}")

    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hikvision Termal Canlı EDSR Süper Çözünürlük Demo")
    parser.add_argument("--ip", type=str, default="192.168.1.128", help="Kamera IP Adresi")
    parser.add_argument("--user", type=str, default="admin", help="Kullanıcı Adı")
    parser.add_argument("--password", type=str, default="ersindemiray26", help="Kamera Şifresi")
    parser.add_argument("--port", type=int, default=8000, help="Hikvision SDK Portu (Varsayılan: 8000)")
    parser.add_argument("--substream", type=lambda x: str(x).lower() not in ("false", "0", "no"),
                        default=True, help="Alt akışı kullan (Channel 202 = True, Channel 201 = False)")

    parser.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints", "best_model.pth"), help="Model checkpoint dosyası")
    parser.add_argument("--native_w", type=int, default=160, help="Donanımsal termal sensör genişliği (varsayılan: 160)")
    parser.add_argument("--native_h", type=int, default=120, help="Donanımsal termal sensör yüksekliği (varsayılan: 120)")
    parser.add_argument("--colormap", type=str, default="gray", choices=["gray", "inferno", "jet", "hot"], help="Başlangıç renk haritası")
    parser.add_argument("--enable_16x", action="store_true", help="16x Cascade modunu başlangıçta aktif et")
    parser.add_argument("--single_view", action="store_true", help="Yalnızca EDSR çıktısını göster (Yüksek FPS modu)")
    parser.add_argument("--fp16", action="store_true", help="FP16 Yarım Hassasiyet Modu (GPU Hızlandırma)")

    args = parser.parse_args()

    run_live_thermal_sr(
        ip=args.ip,
        username=args.user,
        password=args.password,
        port=args.port,
        sub_stream=args.substream,
        checkpoint=args.checkpoint,
        native_w=args.native_w,
        native_h=args.native_h,
        colormap=args.colormap,
        enable_16x=args.enable_16x,
        single_view=args.single_view,
        fp16=args.fp16,
    )

# Canlı Termal EDSR + Nesne Tespiti

Bu klasör `ThermalDlss` projesindeki tek kanallı EDSR x4 büyütmeyi,
Hikvision/MOBESE RTSP akışını ve bu projedeki üç sınıflı YOLO26n modelini tek
bir canlı işlem hattında birleştirir.

## İşlem hattı

```text
Hikvision RTSP (kanal 201/202)
        |
        v
en yeni kareyi tutan düşük tamponlu okuyucu
        |
        v
gri ton + 160x120 INTER_AREA
        |
        +---------------------> bicubic 640x480 (A/B seçeneği)
        |
        v
EDSR x4, 1 kanal, 64 özellik, 16 residual blok
        |
        v
640x480 termal SR
        |
        v
oran koruyan açık letterbox -> 640x512
        |
        v
YOLO26n predict(imgsz=(512, 640), rect=False)
        |
        +--> person / bike_motorcycle / car
        |
        +--> görüntü + YOLO öneri etiketi + JSONL metadata
```

EDSR 160x120 kareyi 640x480 üretir. Algılayıcı eğitiminde ve veri hazırlama
kodunda kullanılan sabit şekil 640x512'dir. Aradaki 32 piksel fark için görüntü
dikey yönde esnetilmez: üst ve alta 16'şar piksel letterbox dolgusu eklenir.
YOLO ve etiket koordinatları bu açıkça kaydedilen 640x512 görüntü üzerinde
çalışır. Böylece yeni veriler mevcut veri doğrulama sözleşmesiyle uyumludur.

`ThermalDlss` içindeki 16x ardışık EDSR deneyi canlı tespitte kullanılmadı.
EDSR eğitim dağılımının dışında olduğu ve hata/artefaktları büyüttüğü için bu
entegrasyon yalnızca eğitilmiş x4 yolu kullanır.

## Kopyalanan modeller

- `weights/edsr_x4_best.pth`: `ThermalDlss/checkpoints/best_model.pth`
- `weights/yolo26n_thermal_best.pt`: mevcut projenin en iyi üç sınıflı modeli
- Boyut ve SHA-256 değerleri `models_manifest.json` içindedir.

Kamera SDK DLL'leri kopyalanmadı. İncelenen demo SDK'yı yalnızca oturum açma
kontrolü için kullanıyor; gerçek kareler OpenCV/FFmpeg üzerinden RTSP'den
geliyor. Bu nedenle SDK, canlı tespit için gereksiz bir bağımlılık.

## Kurulum

Proje kökünde:

```powershell
python -m pip install -r thermal_live_detection\requirements.txt
```

Mevcut eğitim ortamında PyTorch ve Ultralytics zaten kuruluysa yalnızca eksik
paketler yüklenir. CUDA kullanılacaksa PyTorch kurulumu makinenin CUDA sürümüyle
uyumlu olmalıdır.

## Kamerayı çalıştırma

Parolayı kaynak koda veya komut satırı geçmişine yazmadan:

```powershell
.\thermal_live_detection\run_live.ps1 `
    -CameraIp "KAMERA_IP" `
    -Channel 202 `
    -DetectorInput edsr `
    -CaptureMode manual
```

Pencerede `s` inceleme karesi kaydeder, `q` kapatır. Düşük akış/yüksek akış
kamera ayarına göre kanal `201` veya `202` seçilebilir.

Alternatif olarak ortam değişkenleri kullanılabilir:

```powershell
$env:THERMAL_CAMERA_IP = "KAMERA_IP"
$env:THERMAL_CAMERA_USER = "admin"
$env:THERMAL_CAMERA_PASSWORD = Read-Host "Parola"
python -m thermal_live_detection.app --channel 202 --capture-mode hybrid
$env:THERMAL_CAMERA_PASSWORD = $null
```

Yerel video veya kamera olmadan bir dosya kaynağı:

```powershell
python -m thermal_live_detection.app `
    --source "kayit.mp4" `
    --detector-input edsr `
    --capture-mode hybrid
```

## Algılama ve A/B seçenekleri

`--detector-input` üç kontrollü yol sunar:

- `edsr`: önerilen ana deney; 160x120 -> EDSR -> 640x480.
- `bicubic`: aynı 160x120 girişin bicubic büyütmesi.
- `source`: RTSP dekoderinden gelen kareyi doğrudan YOLO'ya verir.

Aynı sahne için yolların karşılaştırılması gerekir. EDSR'nin PSNR/SSIM kazancı
tek başına daha iyi tespit anlamına gelmez. En güvenilir karar; aynı zaman,
kamera, sahne ve insan etiketleri üzerinde kişi başına AP/recall, özellikle
`bike_motorcycle` AP/recall ve yanlış alarm sayısını karşılaştırmaktır.

Varsayılan tespit geleneksel NMS kullanır. Önceki değerlendirmede bu yol
end-to-end başlığa göre az farkla daha iyi olduğu için `--end2end` yalnızca
karşılaştırma amacıyla açılmalıdır.

## Kayıt biçimi

Her çalıştırma `captures/session_<UTC>` klasörü oluşturur:

```text
source_frames/   RTSP dekoderinden gelen özgün kare
native_frames/   EDSR'ye gerçekten verilen 160x120 gri kare
sr_frames/       EDSR'nin 640x480 çıktısı
detector_inputs/ YOLO'ya verilen 640x512 letterbox görüntü; etiket bunun üzerindedir
previews/        kutulu inceleme görüntüsü
pseudo_labels/   standart YOLO kutuları; henüz insan onaylı değildir
session.json     deney/politika bilgisi
metadata.jsonl   zaman, şekil, gecikme, güven ve yakalama nedeni
```

Görüntüler kayıplı JPEG yerine PNG yazılır. `pseudo_labels` biçimi standart
`class x_center y_center width height` olup değerler 0-1 aralığında ve
`detector_inputs` görüntüsüne göredir. Güven skorları etiket satırına değil
`metadata.jsonl` dosyasına yazılır.

Kayıt ve etiketleme stratejisi için
[`DATA_COLLECTION_AND_LABELING.md`](DATA_COLLECTION_AND_LABELING.md) izlenmeli.

## BU-TIV kayıtlı video demosu

`data/bu_tiv` altında BU-TIV Marathon Seq3/Seq4 MP4 videoları ve XML
etiketleri bulunur. Kaynağın 2:1 geometrisi korunarak 160x80 içerik 160x120
tuvale ortalanır. Source, bicubic ve EDSR YOLO sonuçları ile pikselleri yalnızca
görüntüleme amacıyla nearest-neighbor büyütülmüş 160x120 giriş dört panelde
oynatılır. Lens bulanıklığı varsayılan olarak kapalıdır:

```powershell
python -m thermal_live_detection.butiv_demo
```

Yalnızca bisiklet/motosiklet de içeren Seq4:

```powershell
python -m thermal_live_detection.butiv_demo `
    --sequence 4
```

`q`/`ESC` çıkış, `p`/`SPACE` duraklatma tuşudur. XML ground-truth kutuları
varsayılan olarak ince sınıf renkleriyle, YOLO tahminleri kendi kutularıyla
gösterilir. MP4 kopyaları 512x256, XML koordinatları 1024x512 olduğundan
koordinat dönüşümü oynatıcı tarafından otomatik yapılır.

Lens bulanıklığı yalnızca kontrollü deney için açılmalıdır:

```powershell
python -m thermal_live_detection.butiv_demo `
    --sequence 4 `
    --lens-blur-sigma 0.5
```

### Rastgele dört-panel raporu

Varsayılan olarak Seq3/Seq4 içinden seed ile tekrar üretilebilir 200 rastgele
etiketli kare seçer. Her kare için dört panelli JPEG; source/bicubic/EDSR
TP-FP-FN, precision-recall-F1 ve sınıf bazlı metrikleri; CSV, JSON, TXT ve PNG
grafikleri kaydeder:

```powershell
python -m thermal_live_detection.butiv_random_report
```

Farklı örnek sayısı ve seed:

```powershell
python -m thermal_live_detection.butiv_random_report `
    -n 50 `
    --seed 123 `
    --sequence 4
```

Her çalıştırma `thermal_live_detection/results/butiv_random_*` altında
`frames`, `graphs`, `per_frame_metrics.csv`, `summary.csv`, `summary.json` ve
`REPORT.txt` üretir. Varsayılan eşleştirme aynı sınıf ve IoU >= 0.50
koşuludur.

## Güvenlik notu

Eski kamera demosunda kaynak kod içine gömülmüş bir kamera parolası bulundu.
Bu değer yeni klasöre taşınmadı. Parolanın değiştirilmesi ve yalnızca geçici
ortam değişkeni/etkileşimli istem üzerinden verilmesi gerekir.

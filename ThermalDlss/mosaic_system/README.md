# ThermalDlss Mosaic System

Bu klasör native `640×512 → 2560×2048` EDSR ×4 fine-tuning hattını ana
projeyi değiştirmeden uygular.

Temel ilkeler:

- `thermal database` içindeki görüntüler yerinde okunur; kopyalanmaz.
- 16 görüntülü `4×4` pseudo-HR tuval varsayılan olarak yalnız RAM'de oluşur.
- Aynı görüntü bir epoch içinde iki mozaikte kullanılmaz.
- Aynı 16'lı kombinasyon farklı epoch'larda yeniden üretilmez.
- Ana projenin `EDSR`, `ThermalSRDataset`, `SobelEdgeLoss`, PSNR/SSIM ve tiled
  inference kodları import edilerek kullanılır.
- Eski checkpoint `--pretrained` ile yalnız ağırlık kaynağıdır; optimizer yeni
  kurulur. `--resume` yalnız bu sistemin checkpoint'ları içindir.

## Hızlı kullanım

Komutlar proje kökünde çalıştırılmalıdır:

```powershell
python -m mosaic_system inspect --epochs 3

python -m mosaic_system train

python -m mosaic_system evaluate `
  --checkpoint mosaic_system/runs/native_x4/best_model.pth `
  --split val
```

RTX 3060 üzerinde aynı seed ve aynı evaluation manifestiyle otomatik
hiperparametre araması:

```powershell
python -m mosaic_system tune `
  --device cuda `
  --trials 24 `
  --epochs-per-trial 10 `
  --seed 42 `
  --output-dir mosaic_system/runs/optuna_rtx3060
```

Kurulum, kesintiden devam, raporlar ve seçilen ayarla 60 epoch final eğitim:

**[`OPTUNA_KULLANIM_REHBERI.md`](OPTUNA_KULLANIM_REHBERI.md)**

Yavaş bir bilgisayarda eski modeli 20–30 epoch fine-tune etmek, baseline ile
karşılaştırmak ve grafik üretmek için doğrudan:

**[`YAVAS_PC_KULLANIM_REHBERI.md`](YAVAS_PC_KULLANIM_REHBERI.md)**

dosyasındaki sıralı komutları kullanın.

`python` PATH'te değilse projede PyTorch kurulu Python executable'ı doğrudan
kullanılabilir:

```powershell
C:\path\to\python.exe -m mosaic_system train
```

## Varsayılan eğitim

```text
pretrained: checkpoints/best_model.pth
patch: 96×96 LR / 384×384 HR
learning rate: 1e-5
loss: L1 + 0.01 × Sobel
paired replay: %70
16'lı mosaic: %30
cache: memory
AMP: CUDA varsa açık
epoch: 60
```

Otomatik benzersiz örnek sınırında train epoch'u yaklaşık:

```text
1.822 mevcut paired örnek + 781 benzersiz mozaik = 2.603 örnek
```

oluşturur. Mozaik sayısını yapay biçimde yükseltip aynı birleşimi aynı epoch
içinde tekrar etmez.

## RAM ve rolling disk seçenekleri

Varsayılan ve önerilen yol:

```powershell
python -m mosaic_system train --cache-mode memory
```

Her mozaik geçici olarak RAM'de oluşur, crop tensörü çıkarıldıktan sonra serbest
bırakılır. Bir tam uint8 LR/HR çiftinin ham yükü yaklaşık `5,57 MB`'dır.

RAM/üretim süresi dengesi için sıradaki `N` örneklik geçici cache:

```powershell
python -m mosaic_system train `
  --cache-mode rolling_disk `
  --cache-size 8
```

Rolling cache:

- yalnız `mosaic_system/.cache` altında çalışır;
- sadece geçerli pencerenin `.npz` dosyalarını tutar;
- yeni pencereye geçerken eskisini siler;
- normal kapanışta tüm payload'u kaldırır;
- crash sonrasındaki kalıntıyı bir sonraki başlangıçta temizler;
- marker bulunmayan bir dizinde silme yapmayı reddeder.

Mozaik görüntüleri dataset veya proje içine kalıcı olarak yazılmaz.

## CPU, CUDA ve CuPy

Model cihazı:

```text
--device auto   CUDA varsa CUDA, yoksa CPU
--device cuda   CUDA zorunlu
--device cpu    CPU zorunlu
```

Dataset ön-işleme:

```text
--preprocess-backend auto   CuPy+CUDA varsa CuPy, yoksa CPU
--preprocess-backend cupy   CuPy+CUDA zorunlu
--preprocess-backend cpu    NumPy/PyTorch CPU
```

Görüntü decode ve referans bicubic işlemi, ana projeyle aynı sonuç karakterini
korumak için PIL/CPU'da yapılır. CuPy yolu crop sonrasındaki tensör aktarımı,
flip ve rotasyonu GPU'da yapar; bu ana `dataset.py` davranışıyla uyumludur.
Model forward/backward CUDA üzerinde çalışır.

Windows, CuPy veya rolling cache kullanımında DataLoader worker sayısı güvenli
biçimde `0` yapılır. Diğer CPU/Linux durumlarında `--num-workers` kullanılabilir.

## Mozaik ve dikiş seçenekleri

```text
--seam-mode avoid    Patch tek LR karo içinde kalır (varsayılan)
--seam-mode mask     Dikiş kesişebilir; dikiş bandı loss'tan çıkarılır
--seam-mode include  Dikiş normal piksel gibi loss'a girer (kontrol deneyi)
```

`--patch-size 128 --seam-mode avoid --seam-margin-lr 4` geometrik olarak
`160×128` LR karo içine sığmaz. Bu deneyde `--seam-mode mask` kullanılmalı veya
marj sıfırlanmalıdır.

Örnek ablation komutları:

```powershell
python -m mosaic_system train `
  --patch-size 128 `
  --seam-mode mask `
  --edge-weight 0.03 `
  --output-dir mosaic_system/runs/patch128_edge003

python -m mosaic_system train `
  --no-post-shuffle-relu `
  --output-dir mosaic_system/runs/no_post_shuffle_relu
```

## Checkpoint davranışı

Fine-tuning:

```powershell
python -m mosaic_system train `
  --pretrained checkpoints/best_model.pth
```

Sıfırdan kontrol:

```powershell
python -m mosaic_system train `
  --from-scratch `
  --output-dir mosaic_system/runs/from_scratch
```

Kesilmiş mosaic eğitimine devam:

```powershell
python -m mosaic_system train `
  --resume mosaic_system/runs/native_x4/last_checkpoint.pth `
  --output-dir mosaic_system/runs/native_x4
```

`--resume`, optimizer/scheduler/AMP scaler durumunu da yükler.

## Değerlendirme çıktıları

`evaluate` varsayılan olarak sabit ve tekrarlanabilir **validation pseudo-HR**
manifestinde tiled inference kullanır:

```text
hedef: 16 × 640×512 görüntünün 4×4 birleşimi = 2560×2048
girdi: aynı hedefin bicubic ×4 küçültülmüş hali = 640×512
model: 640×512 → 2560×2048
baseline: aynı girdinin bicubic ×4 büyütülmüş hali
```

`testFoto` sayısal evaluation'a dahil edilmez.

- `metrics.csv`: görüntü bazında PSNR/SSIM ve artefakt ölçümleri;
- `summary.json`: ortalama sonuçlar ve checkpoint bilgisi;
- `protocol.json`: hedef, girdi ve baseline tanımının makinece okunabilir kaydı;
- `evaluated_manifest.jsonl`: yalnız kaynak dosya adları ve grup imzaları;
- `previews/`: sınırlı sayıda küçültülmüş karşılaştırma paneli.

Pseudo-HR veya LR mozaiklerin kendisi değerlendirme klasörüne yazılmaz.

Grafik ve eski-yeni model karşılaştırma raporu:

```powershell
python -m mosaic_system report `
  --run-dir mosaic_system/runs/native_x4 `
  --evaluation-dir mosaic_system/runs/native_x4/evaluation `
  --baseline-evaluation-dir mosaic_system/runs/baseline_evaluation
```

Bu komut loss, PSNR/SSIM, bicubic kazanç histogramı, artefakt grafiği ve
eşleşen gruplarda yeni-eski model PSNR farkını PNG olarak üretir.

Tam test:

```powershell
python -m mosaic_system evaluate `
  --checkpoint mosaic_system/runs/native_x4/best_model.pth `
  --split val `
  --max-samples 0 `
  --save-previews 6
```

## Testler

Bağımsız saf-Python testleri:

```powershell
python -m unittest discover -s mosaic_system/tests -v
```

Manifest denetimi gerçek database'i salt okunur kullanır:

```powershell
python -m mosaic_system inspect --epochs 3
```

## Dosya yapısı

| Dosya | Görev |
|---|---|
| `manifest.py` | Deterministik, video dengeli, tekrarsız 16'lı plan |
| `mosaic_io.py` | RAM'de tuval/bicubic üretimi ve rolling cache |
| `data.py` | Mosaic dataset, mevcut dataset adapter'ı, mixed replay |
| `backend.py` | CPU/CUDA/CuPy seçimi |
| `modeling.py` | Kök EDSR importu ve checkpoint yönetimi |
| `losses_ext.py` | Kök Sobel kodunu kullanan seam-maskeli loss |
| `train.py` | AMP/accumulation/early-stop fine-tuning |
| `evaluate.py` | Tiled pseudo-HR test ve artefakt raporu |
| `report.py` | Training/evaluation CSV'lerinden PNG grafik ve Markdown rapor |
| `tune.py` | Optuna araması, aynı-seed evaluation, adalet kontrolü ve final test |
| `inspect_manifest.py` | Dataset/manifest bütünlük denetimi |
| `YAVAS_PC_KULLANIM_REHBERI.md` | 20–30 epoch için kopyala-çalıştır komutları |
| `OPTUNA_KULLANIM_REHBERI.md` | RTX 3060 tuning ve uzun final eğitim rehberi |
| `UYGULAMA_RAPORU.md` | Uygulama kararları ve doğrulama raporu |

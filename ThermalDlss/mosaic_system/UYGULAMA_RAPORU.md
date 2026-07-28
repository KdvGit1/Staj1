# Mosaic System Uygulama ve Doğrulama Raporu

## 2026-07-28 — Optuna ve adil evaluation genişletmesi

`tune.py` ile yeniden başlatılabilir Optuna/TPE hiperparametre optimizasyonu
eklendi. Her tamamlanan trial:

1. aynı eski checkpoint'ten başlar;
2. aynı seed, epoch ve örnek bütçesiyle eğitilir;
3. kendi en iyi checkpoint'iyle sabit validation mozaiklerinde tiled
   `evaluate.py` uygulamasından geçer;
4. baseline ve trial `evaluated_manifest.jsonl` dosyalarının SHA-256
   eşitliği doğrulandıktan sonra sıralamaya alınır.

Trial seçiminde test split'i kullanılmaz. `--run-final`, seçim bittikten sonra
kazanan parametrelerle temiz uzun eğitim yapar; eski ve yeni final modeli aynı
test seed'i/manifestinde değerlendirip `final_test_comparison.json` üretir.
Varsayılan `--pruner none` sayesinde istenen karşılaştırmada bütün trial'lar tam
10 epoch eğitim bütçesi alır. SQLite `study.db` kesinti sonrası devamı sağlar.

Yeni otomatik çıktılar: `trials.csv`, `best_config.json`, `REPORT.md`,
`optimization_history.png`, `parameter_importance.png` ve trial başına
training/evaluation klasörleri. Dataset veya mosaic görüntüsü kopyalanmaz.

**Tarih:** 28 Temmuz 2026  
**Kapsam:** Native `640×512 → 2560×2048` görsel EDSR ×4 fine-tuning  
**Konum:** `ThermalDlss/mosaic_system`  

## 1. Uygulanan sonuç

Planlanan 16 görüntülü pseudo-HR yaklaşımı, ana proje dosyaları
değiştirilmeden bağımsız bir pipeline olarak uygulandı.

Sistem:

1. `thermal database/thermal_dataset_split/{split}` içindeki dosyaları yerinde
   okur.
2. Aynı split içinden 16 benzersiz görüntü seçer.
3. RAM'de `4×4`, `2560×2048` pseudo-HR uint8 tuval oluşturur.
4. PIL bicubic ile `640×512` LR üretir.
5. Eşleşen LR/HR patch ve dikiş maskesini eğitim hattına verir.
6. Mevcut paired dataset örnekleriyle deterministik mixed replay uygular.
7. Kök EDSR checkpoint'inden ağırlık-only fine-tuning yapar.
8. Sabit validation mozaiklerinde tiled inference ve ayrıntılı metrik raporu
   üretir.

Kaynak dataset kopyalanmadı. Kalıcı pseudo-HR/LR mozaik veri seti
oluşturulmadı.

## 2. Yeniden kullanılan mevcut proje kodu

Tek bir değişikliğin bütün projeyi bozmasını önlemek için kök modüller import
edildi:

| Mevcut modül | Yeniden kullanılan parça |
|---|---|
| `model.py` | `EDSR`, residual bloklar ve PixelShuffle upsampler |
| `dataset.py` | `ThermalSRDataset`, CuPy tespiti ve CuPy modülü |
| `losses.py` | `SobelEdgeLoss` ve mevcut Sobel hesaplaması |
| `train.py` | `calculate_psnr`, `calculate_ssim` |
| `upscale_testfoto_x4.py` | Halo'lu `upscale_tiled` inference |

Ana `model.py`, `dataset.py`, `losses.py` ve `train.py` değiştirilmedi.
`mosaic_system/bootstrap.py`, proje kökünü import yoluna eklemekle sınırlıdır.

Yavaş bilgisayarda eğitim, baseline karşılaştırması ve grafik üretimi için
`YAVAS_PC_KULLANIM_REHBERI.md` içindeki sıralı çalışma reçetesi eklenmiştir.

## 3. Dataset ve manifest sonucu

Gerçek database üzerinde:

| Split | Kaynak | Epoch başına mozaik | Artan | Video |
|---|---:|---:|---:|---:|
| Train | 12.505 | 781 | 9 | 123 |
| Validation | 1.563 | 97 | 11 | 19 |
| Test | 1.567 | 97 | 15 | 16 |

Üç epoch gerçek dosya adlarıyla denetlendi:

- epoch içinde tekrar eden kaynak dosya: `0`;
- epoch içinde tekrar eden 16'lı birleşim: `0`;
- üç epoch arasında tekrar eden tam birleşim: `0`;
- train mozaiklerinde farklı video sayısı: minimum `15`, ortalama `15,68`,
  maksimum `16`;
- validation: minimum `12`, ortalama `12,82`, maksimum `13`;
- test: minimum `9`, ortalama `9,92`, maksimum `10`.

Validation ve test split'lerinde 16'dan az veya dengesiz video bulunduğu için
her mozaiğin 16 farklı videodan oluşması matematiksel olarak mümkün değildir.
Algoritma büyük video kovalarını bütün gruplara yayarak mümkün olan en dengeli
dağıtımı yapar.

### Tekrarsızlık kuralı

Bir epoch içinde her görüntü en fazla bir kez tüketilir. Epoch planı seed,
epoch ve nonce üzerinden deterministiktir. Her grubun karo sırasından bağımsız
SHA-256 imzası tutulur. Önceki epoch imzalarıyla çakışma görülürse epoch planı
farklı deterministik nonce ile yeniden kurulur.

Resume işleminde önceki epoch imzaları aynı seed ile yeniden hesaplanır; binlerce
dosya adını içeren büyük manifest kopyaları diske yazılmaz. Bellekte yalnız
mevcut epoch grupları ve küçük SHA-256 imza kümesi tutulur.

## 4. Disk ve RAM davranışı

### Varsayılan RAM modu

Gerçek train görüntülerinden smoke test:

```text
LR:  640×512 uint8
HR: 2560×2048 uint8
toplam ham çift: 5.570.560 byte ≈ 5,31 MiB
```

Tuval `__getitem__` sırasında kurulur. Eğitim patch'i tensöre çevrildikten sonra
tam numpy LR/HR nesnelerine kalıcı referans bırakılmaz.

### Rolling disk modu

`RollingMosaicCache` testinde kapasite `2` ve toplam örnek `4` seçildi:

1. indeks `0` istendiğinde yalnız `0–1` dosyaları oluştu;
2. indeks `2` istendiğinde `0–1` silindi, yalnız `2–3` kaldı;
3. `close()` sonrasında cache dizini kaldırıldı.

Cache temizliği marker kontrollüdür. Marker olmayan hedeflerde silme reddedilir.
Bu güvenlik, yanlış dizinin recursive temizlenmesini önler.

## 5. Eğitim hattı

Varsayılan fine-tuning:

```text
checkpoint: checkpoints/best_model.pth
model yükleme: yalnız model_state_dict
optimizer: yeni Adam
learning rate: 1e-5
patch: 96 LR / 384 HR
edge weight: 0.01
paired:mosaic: 70:30
AMP: CUDA varsa
gradient accumulation: 2
```

`DeterministicMixedDataset`, bir kaynağın kapasitesini aşan
`samples_per_epoch` değerini reddeder. Varsayılan oranlarla mozaik kapasitesi
sınırlayıcıdır:

```text
paired: 1.822 benzersiz örnek
mosaic:   781 benzersiz örnek
toplam: 2.603 örnek/epoch
```

Her epoch'ta paired dosya alt kümesi ve mozaik komşulukları seed ile değişir.
Normal epoch tekrarı, aynı eğitim örneğinin aynı epoch içinde yanlışlıkla
çoğaltılmasıyla karıştırılmaz.

### Dikiş yönetimi

- `avoid`: patch tek karoda, dikiş marjının içinde kalır;
- `mask`: dikiş kesilebilir, çevresi L1 ve Sobel kaybından çıkarılır;
- `include`: dikişi loss'a katan kontrol deneyi.

Masked loss, kök `SobelEdgeLoss._sobel_edges` hesabını yeniden kullanır.

### PixelShuffle ReLU ablation

Kök model değiştirilmez. `--no-post-shuffle-relu` seçeneği model kurulduktan ve
checkpoint yüklendikten sonra upsampler bloklarındaki parametresiz ReLU
modüllerini `nn.Identity` ile değiştirir. Böylece gövde ve upsampler convolution
ağırlıkları aktarılır.

## 6. CPU, CUDA ve CuPy

Uygulanan yollar:

| Katman | CPU | CUDA | CuPy |
|---|---|---|---|
| Görüntü decode | PIL | PIL | PIL |
| Bicubic küçültme | PIL | PIL | PIL |
| Crop/augment | NumPy/PyTorch | NumPy/PyTorch | CuPy |
| Model forward/backward | PyTorch CPU | PyTorch CUDA | PyTorch CUDA |
| AMP | Kapalı | Açık | Açık |

CuPy'nin decode veya PIL ile birebir bicubic eşleniği zorlanmadı. Bu tercih,
mevcut degradation karakterini değiştirmemek ve kök projedeki CuPy kullanım
modelini korumak içindir.

Backend istekleri sessizce yanlış cihaza düşmez:

- `--device cuda` ve CUDA yoksa hata;
- `--preprocess-backend cupy` ve CuPy+CUDA yoksa hata;
- `auto` uygun fallback'i seçer.

## 7. Değerlendirme

Değerlendirme, kök `upscale_tiled` fonksiyonunu kullanır. Varsayılan halo `40`,
minimum güvenli halo `36` olarak doğrulanır.

Bilimsel varsayılan protokol:

```text
split: validation
hedef: 4×4 native mozaik, 2560×2048
girdi: aynı hedefin PIL bicubic ×4 küçültülmüş hali, 640×512
model çıktısı: 2560×2048
bicubic baseline: aynı 640×512 girdinin ×4 büyütülmüş hali
testFoto kullanımı: yok
```

Kod her örnekte LR `(512,640)` ve HR `(2048,2560)` numpy boyutlarını doğrular.
Uyuşmazlıkta metrik üretmek yerine hata verir. Protokol `protocol.json` olarak
evaluation çıktısına yazılır.

Raporlananlar:

- EDSR ve bicubic PSNR/SSIM;
- görüntü başına kazanç;
- dikiş dışı ve dikiş bandı PSNR;
- clipping oranı;
- yatay/dikey gradyan;
- ortalama mutlak Laplacian;
- 4×4 PixelShuffle faz ortalamalarının standart sapması;
- sınırlı sayıda küçültülmüş karşılaştırma preview'u;
- kaynak adları ve grup imzası.

Tam pseudo-HR hedef veya LR mozaik evaluation klasörüne kaydedilmez.

## 8. Yapılan doğrulamalar

### Saf-Python unit testleri

```text
test_cross_epoch_groups_do_not_repeat ... ok
test_epoch_uses_each_path_at_most_once ... ok
test_same_seed_is_reproducible ... ok
test_mosaic_dimensions_and_tile_order ... ok
test_rolling_cache_keeps_only_current_window ... ok

Ran 5 tests
OK
```

### PyTorch entegrasyon testi

Mevcut `checkpoints/best_model.pth` yüklendi:

```text
checkpoint epoch: 74
parametre: 1.515.265
mosaic LR patch:  [1, 48, 48]
mosaic HR patch:  [1, 192, 192]
model çıktı:      [1, 1, 192, 192]
mixed schedule:   7 paired + 3 mosaic
forward/loss:     başarılı
```

### Bir epoch smoke training

CPU üzerinde iki örnek ve bir validation örneği:

```text
epoch: 1
train örnek: 1 paired + 1 mosaic
train loss: 0.02167
validation PSNR: 31.6456 dB
validation SSIM: 0.77195
checkpoint yazma: başarılı
```

Bu sonuç performans iddiası değildir; CLI, data, forward, backward, loss,
optimizer, validation ve checkpoint hattının birlikte çalıştığını doğrular.

### Tiled evaluation smoke testi

Bir sabit pseudo-HR mozaiği, CPU ve tiled inference:

```text
EDSR PSNR: 30.229 dB
bicubic PSNR: 28.949 dB
fark: +1.280 dB
```

CSV, JSON ve manifest raporları başarıyla üretildi. Bu geçici smoke çıktıları
nihai projede tutulmadı.

## 8.1 Otomatik grafik raporu

`python -m mosaic_system report` komutu eklendi. Araç:

- train/validation loss eğrisi;
- validation PSNR/SSIM eğrisi;
- learning-rate eğrisi;
- EDSR–bicubic PSNR scatter grafiği;
- PSNR kazanç histogramı;
- clipping, faz, gradyan ve Laplacian karşılaştırması;
- aynı test gruplarında yeni–eski model PSNR histogramı

üretir. Sayısal özet `report_summary.json`, okunabilir özet `REPORT.md` olarak
kaydedilir.

## 9. Test ortamı sınırı

Bu bilgisayarda erişilebilen `cryptai` Python ortamında:

```text
Python 3.10.18
PyTorch 2.9.1+cpu
CuPy kurulu değil
```

Bu nedenle CPU yolu uçtan uca çalıştırıldı. CUDA/CuPy yolları uygulanmış ve kök
projenin mevcut arayüzleriyle bağlanmıştır; fakat bu oturumda CUDA donanımlı
PyTorch/CuPy runtime bulunmadığı için gerçek GPU smoke testi yapılamadı. Eğitim
ortamında ilk çalıştırılacak doğrulama:

```powershell
python -m mosaic_system train `
  --device cuda `
  --preprocess-backend cupy `
  --epochs 1 `
  --samples-per-epoch 32 `
  --val-max-samples 4 `
  --output-dir mosaic_system/runs/gpu_smoke
```

## 10. Bilinçli kapsam kararları

- `testFoto` içindeki altı DJI görüntüsü eğitim verisine katılmadı. Bunların
  mevcut değerlendirme/görsel hedef örnekleri olması ve RGB pseudo-color domain
  farkı nedeniyle train contamination oluşturması önlendi.
- Ana proje checkpoint'ları veya training logları değiştirilmedi.
- Ana database'e dosya yazılmadı.
- Sıfırdan eğitim desteklenir, fakat varsayılan mevcut ağırlıklardan
  fine-tuning'dir.
- İki-model fusion bu klasörün kapsamına alınmadı; sonraki faz olarak kaldı.

## 11. Önerilen gerçek deney sırası

1. `inspect --epochs 3` ile hedef makinede manifest denetimi.
2. 32 örnekli CUDA/CuPy smoke run.
3. Baseline: patch `48`, edge `0.10`.
4. Ana aday: patch `96`, edge `0.01`.
5. Patch `128`, `seam_mode=mask`.
6. Edge ablation: `0`, `0.01`, `0.03`, `0.10`.
7. En iyi veri/loss adayında `--no-post-shuffle-relu`.
8. Sabit 97 test mozaiğinde tam evaluation.
9. Native hedef görüntülerde ayrı kör görsel değerlendirme.

# Yavaş PC İçin 20–30 Epoch Fine-tuning ve Değerlendirme Rehberi

Bu dosya doğrudan kopyalanıp çalıştırılabilecek komutları içerir. Bütün komutlar
`ThermalDlss` proje kökünde çalıştırılmalıdır.

## 1. Inspect sonucunun anlamı

Train sonucu:

```text
12.505 kaynak // 16 = 781 tam mozaik
12.505 mod 16 = 9 artan görüntü
```

Bu nedenle `groups=781` ve `leftovers=9` doğrudur.

`checked_epochs=3`, modelin üç epoch eğitildiği anlamına gelmez. Yalnız epoch
0, 1 ve 2 için mozaik planlarının denetlendiğini belirtir.

`cross_epoch_duplicate_groups=0`, bu üç plan arasında aynı 16 dosyadan oluşan
tam birleşimin tekrarlanmadığını gösterir. Tekil kaynak görüntülerin sonraki
epoch'larda farklı komşularla yeniden kullanılması normal ve istenen eğitim
davranışıdır.

Train mozaiklerinin her birinde en az 15 farklı video bulunması güçlü bir
sonuçtur. En büyük train videosunda 1.033 kare, epoch başına ise yalnız 781
mozaik vardır. Dolayısıyla bazı gruplarda o videodan iki kare bulunması
matematiksel olarak kaçınılmazdır.

Validation'da yalnız 19, testte yalnız 16 video vardır ve kare sayıları dengeli
değildir. Bu yüzden:

```text
validation: ortalama 12,82 farklı video
test:       ortalama  9,92 farklı video
```

bir kod hatası değildir. Hiçbir dosyanın aynı epoch içinde iki kez
kullanılmaması daha önemli bütünlük koşuludur.

## 2. Hangi Python kullanılmalı?

Önce:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

çalıştırın. `python` bulunamıyorsa PyTorch kurulu ortamın executable'ını
kullanın:

```powershell
C:\path\to\python.exe -m mosaic_system inspect --epochs 3
```

CUDA sonucu `True` ise komutlarda `--device auto` CUDA'yı seçer. CuPy yoksa
`--preprocess-backend cpu` kullanmak güvenlidir; model yine CUDA'da çalışabilir.

## 3. Önce eski modeli baseline olarak ölç

Fine-tuning öncesinde eski checkpoint'i sabit 10 mozaikte ölç:

```powershell
python -m mosaic_system evaluate `
  --checkpoint checkpoints/best_model.pth `
  --output-dir mosaic_system/runs/baseline_eval_10 `
  --split val `
  --max-samples 10 `
  --save-previews 4 `
  --seed 42 `
  --device auto `
  --preprocess-backend cpu
```

Bu adımı atlamayın. Yeni modelin gerçekten iyileşip iyileşmediği ancak aynı
10 validation mozaiği ve aynı seed ile karşılaştırılabilir. Hedef 16 native
görüntünün `2560×2048` birleşimi, model girdisi ise aynı hedefin bicubic ile
`640×512` boyutuna indirilmiş halidir.

Çıktılar:

```text
mosaic_system/runs/baseline_eval_10/
├── metrics.csv
├── summary.json
├── protocol.json
├── evaluated_manifest.jsonl
└── previews/
```

## 4. Bir dakikalık/ kısa smoke eğitim

Önce bütün hattın kendi bilgisayarınızda çalıştığını kontrol edin:

```powershell
python -m mosaic_system train `
  --pretrained checkpoints/best_model.pth `
  --epochs 1 `
  --patch-size 48 `
  --batch-size 1 `
  --gradient-accumulation 1 `
  --samples-per-epoch 32 `
  --val-max-samples 4 `
  --device auto `
  --preprocess-backend cpu `
  --output-dir mosaic_system/runs/local_smoke
```

Beklenen dosyalar:

```text
config.json
training_log.csv
last_checkpoint.pth
best_model.pth
```

Bu çalışma yalnız teknik kontroldür; kalite sonucu değildir.

## 5. Yavaş PC için önerilen 25 epoch fine-tuning

Bu profil hız önceliklidir. Eski modelden başlanır, sıfırdan eğitim yapılmaz:

```powershell
python -m mosaic_system train `
  --pretrained checkpoints/best_model.pth `
  --epochs 25 `
  --patch-size 48 `
  --batch-size 1 `
  --gradient-accumulation 4 `
  --samples-per-epoch 512 `
  --val-max-samples 8 `
  --learning-rate 1e-5 `
  --edge-weight 0.01 `
  --paired-ratio 0.70 `
  --mosaic-ratio 0.30 `
  --seam-mode avoid `
  --cache-mode memory `
  --patience 8 `
  --device auto `
  --preprocess-backend cpu `
  --output-dir mosaic_system/runs/slow_pc_25e
```

Bu komutta epoch başına yaklaşık:

```text
358 paired örnek + 154 benzersiz mozaik = 512 örnek
```

kullanılır. Tam profilin 2.603 örneği yerine 512 örnek seçildiği için yavaş
PC'de yaklaşık beş kat daha kısa bir epoch hedeflenir.

### Biraz daha kaliteli fakat daha yavaş profil

Sistem hızlı profili kaldırıyorsa:

```powershell
python -m mosaic_system train `
  --pretrained checkpoints/best_model.pth `
  --epochs 25 `
  --patch-size 96 `
  --batch-size 1 `
  --gradient-accumulation 4 `
  --samples-per-epoch 512 `
  --val-max-samples 8 `
  --learning-rate 1e-5 `
  --edge-weight 0.01 `
  --paired-ratio 0.70 `
  --mosaic-ratio 0.30 `
  --seam-mode avoid `
  --device auto `
  --preprocess-backend cpu `
  --output-dir mosaic_system/runs/slow_pc_25e_patch96
```

`96×96` LR patch, `48×48` patch'e göre yaklaşık dört kat daha fazla piksel
işler. Kalite planına daha uygundur fakat CPU'da belirgin yavaşlar.

## 6. Eğitim kesilirse devam et

Resume sırasında ilk komuttaki patch, oran ve örnek sayılarını aynen verin:

```powershell
python -m mosaic_system train `
  --resume mosaic_system/runs/slow_pc_25e/last_checkpoint.pth `
  --epochs 25 `
  --patch-size 48 `
  --batch-size 1 `
  --gradient-accumulation 4 `
  --samples-per-epoch 512 `
  --val-max-samples 8 `
  --learning-rate 1e-5 `
  --edge-weight 0.01 `
  --paired-ratio 0.70 `
  --mosaic-ratio 0.30 `
  --seam-mode avoid `
  --device auto `
  --preprocess-backend cpu `
  --output-dir mosaic_system/runs/slow_pc_25e
```

`--resume` yerine yanlışlıkla eski `checkpoints/best_model.pth` verilmemelidir.
Eski model yalnız `--pretrained` ile kullanılır.

25 epoch tamamlandıktan sonra ek eğitim istenirse scheduler'ı zorla uzatmak
yerine en iyi modeli yeni bir fine-tuning aşamasının başlangıcı yapın:

```powershell
python -m mosaic_system train `
  --pretrained mosaic_system/runs/slow_pc_25e/best_model.pth `
  --epochs 5 `
  --patch-size 48 `
  --batch-size 1 `
  --gradient-accumulation 4 `
  --samples-per-epoch 512 `
  --val-max-samples 8 `
  --output-dir mosaic_system/runs/slow_pc_extra_5e
```

## 7. Yeni modeli aynı 10 validation mozaiğinde evaluate et

```powershell
python -m mosaic_system evaluate `
  --checkpoint mosaic_system/runs/slow_pc_25e/best_model.pth `
  --output-dir mosaic_system/runs/slow_pc_25e/evaluation_10 `
  --split val `
  --max-samples 10 `
  --save-previews 6 `
  --seed 42 `
  --device auto `
  --preprocess-backend cpu
```

Baseline ile aynı `max-samples=10` ve `seed=42` kullanıldığı için
`group_id`'ler eşleşir.

### Tam 97 mozaik testi

Kısa test iyi görünüyorsa:

```powershell
python -m mosaic_system evaluate `
  --checkpoint mosaic_system/runs/slow_pc_25e/best_model.pth `
  --output-dir mosaic_system/runs/slow_pc_25e/evaluation_full `
  --split val `
  --max-samples 0 `
  --save-previews 10 `
  --seed 42 `
  --device auto `
  --preprocess-backend cpu
```

CPU'da tiled `2560×2048` değerlendirme zaman alır. Hızlı kontrol için
`--skip-ssim` eklenebilir; nihai raporda SSIM istendiği için tam koşuda
kullanılmamalıdır.

## 8. Sayısal sonuçlar nasıl okunur?

`summary.json` içindeki başlıca alanlar:

| Alan | Yorum |
|---|---|
| `psnr_model` | EDSR ortalama PSNR; yüksek daha iyi |
| `psnr_bicubic` | Bicubic referans PSNR |
| `psnr_gain` | Model − bicubic; pozitif olmalı |
| `ssim_model` | EDSR yapısal benzerliği; yüksek daha iyi |
| `ssim_gain` | Model − bicubic |
| `model_clip_ratio` | 0/255'e kırpılan piksel oranı; aşırı artmamalı |
| `model_phase_mean_std` | 4×4 faz/ızgara göstergesi; düşük tercih edilir |
| `model_gradient_x` | Keskinlik göstergesi; aşırı artış halo anlamına gelebilir |

Yeni model kabul edilirken yalnız PSNR'a bakılmamalıdır:

```text
PSNR/SSIM artıyor
+ clipping ve faz artefaktı artmıyor
+ previews doğal görünüyor
= başarılı aday
```

## 9. Grafik ve otomatik rapor oluştur

Yeni evaluation bittikten sonra:

```powershell
python -m mosaic_system report `
  --run-dir mosaic_system/runs/slow_pc_25e `
  --evaluation-dir mosaic_system/runs/slow_pc_25e/evaluation_10 `
  --baseline-evaluation-dir mosaic_system/runs/baseline_eval_10
```

Üretilenler:

```text
mosaic_system/runs/slow_pc_25e/report/
├── REPORT.md
├── report_summary.json
├── training_losses.png
├── validation_psnr_ssim.png
├── learning_rate.png
├── evaluation_psnr_scatter.png
├── evaluation_psnr_gain_histogram.png
├── evaluation_artifacts.png
└── new_vs_old_psnr.png
```

`new_vs_old_psnr.png`, aynı `group_id` üzerindeki yeni ve eski model farkını
gösterir. Rapor ayrıca yeni modelin eski modeli geçtiği mozaik oranını verir.

## 10. Evaluation görüntüleri nerede?

Pseudo-HR evaluation önizlemeleri:

```text
mosaic_system/runs/slow_pc_25e/evaluation_10/previews/
```

Her panel:

```text
Pseudo-HR hedef | Bicubic | Yeni EDSR
```

`testFoto` bu bilimsel evaluation hattında kullanılmaz. Sayısal metriklerin
tamamı validation split'inden oluşturulan pseudo-HR hedef ile aynı hedefin
bicubic küçültülmüş girdisi arasında hesaplanır.

## 11. Hangi checkpoint kullanılmalı?

```text
last_checkpoint.pth  = son tamamlanan epoch; resume için
best_model.pth       = en yüksek validation PSNR; evaluation/inference için
```

Model çıktısı üretirken her zaman önce `best_model.pth` denenmelidir.

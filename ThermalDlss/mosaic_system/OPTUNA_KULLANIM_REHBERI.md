# RTX 3060 ile Adil Hiperparametre Optimizasyonu

Bu sistem her aday modeli aynı koşullarda karşılaştırır:

- aynı `checkpoints/best_model.pth` başlangıç ağırlığı;
- aynı train/validation seed'i (`42`);
- aynı epoch ve örnek bütçesi;
- aynı validation mozaik grupları;
- her trial sonunda aynı tiled `evaluate` uygulaması;
- evaluation manifestlerinin SHA-256 eşitlik kontrolü;
- test split'inin yalnız final model seçildikten sonra açılması.

`thermal database` kopyalanmaz. Mozaik görüntüleri kalıcı kaydedilmez; mevcut
RAM/rolling-cache sistemi aynen kullanılır. Trial klasörlerinde yalnız
checkpoint, CSV/JSON metrikleri ve sınırlı rapor dosyaları bulunur.

## 1. Ortamı hazırlama

PyTorch/CUDA'nın kurulu olduğu Python ortamını etkinleştirin. Sonra:

```powershell
python -m pip install -r mosaic_system/requirements-tuning.txt
```

CUDA'yı doğrulayın:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

İlk satır `True`, ikinci satır RTX 3060 model adını göstermelidir.

## 2. Bir epoch smoke test

Smoke test için ayrı bir çıktı klasörü kullanın:

```powershell
python -m mosaic_system tune `
  --device cuda `
  --preprocess-backend auto `
  --trials 1 `
  --epochs-per-trial 1 `
  --samples-per-epoch 256 `
  --val-max-samples 8 `
  --eval-max-samples 2 `
  --output-dir mosaic_system/runs/optuna_smoke
```

Bu çalışma kalite kararı için kullanılmaz. Amaç CUDA, veri hattı, eğitim,
checkpoint ve tiled evaluation zincirini doğrulamaktır.

## 3. Önerilen 24 trial × 10 epoch araması

```powershell
python -m mosaic_system tune `
  --device cuda `
  --preprocess-backend auto `
  --trials 24 `
  --epochs-per-trial 10 `
  --samples-per-epoch 1024 `
  --val-max-samples 32 `
  --eval-max-samples 10 `
  --seed 42 `
  --sampler-seed 2026 `
  --pruner none `
  --effective-batch-size 8 `
  --output-dir mosaic_system/runs/optuna_rtx3060
```

`--trials 24` toplam hedef trial sayısıdır. Çalışma örneğin trial 9'da
kesilirse aynı komut yeniden çalıştırıldığında 24'e kadar devam eder; 24 yeni
trial daha eklemez.

Varsayılan arama uzayı:

```text
patch_size:        48, 64, 96
learning_rate:     2e-6 ... 3e-5 (logaritmik)
edge_weight:       0, 0.005, 0.01, 0.02, 0.03
mosaic_ratio:      0.30, 0.50, 0.70
post_shuffle_relu: açık / kapalı
```

Patch değiştiğinde fiziksel batch otomatik ayarlanır, gradient accumulation
ise etkin batch'i `8` civarında sabit tutar. Böylece patch deneyi mümkün
olduğunca batch deneyi haline gelmez.

Varsayılan `--pruner none` özellikle seçilmiştir: bütün adaylar tam 10 epoch
eğitilir. Zaman kısıtı varsa `--pruner median` kullanılabilir; bu durumda
zayıf trial'lar üçüncü epoch sonrasından itibaren kesilebilir ve tamamlanmamış
trial'lara full evaluation uygulanmaz.

## 4. Sonuçları okuma

Ana dosyalar:

```text
mosaic_system/runs/optuna_rtx3060/
├── study.db
├── tuning_config.json
├── baseline_val_evaluation/
├── trials/
│   └── trial_XXXX/
│       ├── best_model.pth
│       ├── training_log.csv
│       └── evaluation/
├── trials.csv
├── best_config.json
├── REPORT.md
├── optimization_history.png
├── trial_vs_starting_model.png
├── quality_constraint_violations.png
└── parameter_importance.png
```

Seçim kuralı:

1. SSIM başlangıç modelinden düşük olmamalı.
2. Phase artefaktı başlangıç modelinin `%10` üst sınırını aşmamalı.
3. Clipping başlangıç modelinin `%10` üst sınırını aşmamalı.
4. Bu şartları geçenler içinde aynı-manifest evaluation PSNR'si en yüksek
   trial seçilmeli.

Optuna'nın epoch içi `val_psnr` değeri yalnız arama/pruning sinyalidir. Nihai
trial sıralaması patch validation metriğiyle değil, trial bitiminde çalışan tam
tiled mosaic evaluation PSNR'siyle yapılır.

## 5. Seçilen ayarla uzun eğitim ve tarafsız test

Arama bittikten sonra aynı komuta `--run-final` ekleyin:

```powershell
python -m mosaic_system tune `
  --device cuda `
  --preprocess-backend auto `
  --trials 24 `
  --epochs-per-trial 10 `
  --samples-per-epoch 1024 `
  --val-max-samples 32 `
  --eval-max-samples 10 `
  --seed 42 `
  --sampler-seed 2026 `
  --pruner none `
  --effective-batch-size 8 `
  --output-dir mosaic_system/runs/optuna_rtx3060 `
  --run-final `
  --final-epochs 60 `
  --final-patience 20 `
  --final-samples-per-epoch 0 `
  --final-val-max-samples 97 `
  --final-test-max-samples 0
```

Bu aşama:

1. seçilen parametrelerle eski checkpoint'ten temiz bir 60 epoch eğitim açar;
2. test split'inde eski modeli seed `42` ile değerlendirir;
3. aynı test gruplarında yeni final modeli seed `42` ile değerlendirir;
4. grup bazında PSNR/SSIM farkını ve kazanma sayılarını yazar.

Final sonuç:

```text
mosaic_system/runs/optuna_rtx3060/final_test_comparison.json
```

`same_manifest` alanı `true` olmalıdır. Bu alan `false` ise sonuçlar bilimsel
karşılaştırma olarak kullanılmamalıdır.

## Önemli sınırlar

- Arama sırasında `test` sonucu açılmaz ve parametre seçiminde kullanılmaz.
- Başka bir başlangıç checkpoint'i veya seed denenecekse yeni `--output-dir`
  kullanılmalıdır.
- Mevcut study'nin seed, checkpoint veya trial bütçesi değiştirilirse sistem
  devam etmeyi reddeder.
- Final eğitim klasörü doluysa üzerine yazılmaz.
- `study.db` silinmedikçe çalışma kesintiden sonra devam edebilir.

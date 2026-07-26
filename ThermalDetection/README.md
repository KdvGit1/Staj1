# Thermal YOLO26n Detection

Bu proje, 640×512 tek kanallı termal görüntülerde yalnızca şu üç sınıfı
tespit etmek için hazırlanmıştır:

1. `person`
2. `bike_motorcycle` (`bike` ve `motor` birleşik)
3. `car`

Kaynak COCO verisi değiştirilmez. Dönüştürme aracı sadece `data` klasörlerini
kullanır; `analyticsData` eğitime alınmaz.

## Hızlı başlangıç

Python 3.10+ ve hedef NVIDIA sisteme uygun CUDA destekli PyTorch kurulmalıdır.
Ardından:

```powershell
python -m pip install -r requirements.txt
python check_environment.py --require-cuda
python prepare_dataset.py --backend auto
python verify_dataset.py --backend auto
python train_yolo26n.py --smoke-test --device 0
python train_yolo26n.py --device 0
```

Veri doğrulaması tamamlandığında anlaşılır SVG/HTML grafik raporu otomatik
oluşur:

```text
reports/graphs/dataset/dataset-report.html
```

CUDA 12.x ile yardımcı veri işlemlerinde CuPy kullanmak için:

```powershell
python -m pip install -r requirements-cupy-cuda12.txt
```

CuPy çalışır durumdaysa `--backend auto` onu seçer; aksi durumda güvenli şekilde
NumPy kullanılır. Model eğitimi her durumda PyTorch/CUDA tarafından yapılır.

Validation:

```powershell
python evaluate_model.py --model runs/thermal_detection/yolo26n_640x512/weights/best.pt --split val --compare-heads
```

Test, yalnızca model ve eşikler validation üzerinde kesinleştirildikten sonra:

```powershell
python evaluate_model.py --model runs/thermal_detection/yolo26n_640x512/weights/best.pt --split test
```

Inference:

```powershell
python infer.py --model path/to/best.pt --source path/to/image_or_video --device 0
```

Sabit `1×3×512×640` ONNX:

```powershell
python export_model.py --model path/to/best.pt --format onnx --device 0
```

## Grafik raporları

Veri seti raporunu yeniden üretmek:

```powershell
python generate_graph_reports.py
```

Mevcut bir eğitim koşumundan loss, metrik, learning-rate ve süre grafikleri:

```powershell
python generate_graph_reports.py --run-dir runs/thermal_detection/yolo26n_640x512
```

Eğitim, değerlendirme ve inference scriptleri kendi grafik raporlarını
otomatik üretir. Çıktılar bağımlılıksız SVG ve çevrimdışı HTML biçimindedir;
grafik üzerindeki değerlerin yanında eksiksiz sayısal tablolar da bulunur.

Tüm kararlar, parametreler, klasörler ve sorun giderme adımları
`PROJE_DOKUMANTASYONU.txt` dosyasında ayrıntılı olarak açıklanmıştır.

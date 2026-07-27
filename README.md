# staj1

Bu depo, staj çalışmaları için tek bir Git monoreposudur:

- `ThermalDetection`
- `ThermalDlss`
- `ThermalRealEsrgan`

## Veri setleri

Büyük veri setleri Git yerine DVC ile izlenir. Git'e yalnızca küçük `.dvc`
manifestleri eklenir. Mevcut veri klasörleri `.gitignore` içinde açıkça
korunur; gelecekte eklenecek DVC veri klasörleri için ortak son ek
`*.dvcdata/` olarak belirlenmiştir.

### Kurulum

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dvc.txt
```

### Google Drive uzak deposunu bağlama

`FOLDER_ID` yerine paylaşılacak Google Drive klasörünün bağlantısındaki
klasör kimliğini yazın:

```powershell
.\.venv\Scripts\dvc.exe remote add -d storage "gdrive://FOLDER_ID"
.\.venv\Scripts\dvc.exe push
```

Başka bir bilgisayarda kodu ve veri setlerini indirmek için:

```powershell
git clone https://github.com/KdvGit1/staj1.git
cd staj1
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dvc.txt
.\.venv\Scripts\dvc.exe pull
```

## DVC'yi güncelleme

### DVC programını güncelleme

Bu projede Google Drive uyumluluğu için DVC ve ilgili paketlerin çalışan
sürümleri `requirements-dvc.txt` dosyasında sabitlenmiştir. Mevcut, test edilmiş
sürümleri kurmak veya onarmak için:

```powershell
cd C:\Users\KDV\Desktop\Staj
.\.venv\Scripts\python.exe -m pip install --upgrade -r requirements-dvc.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\dvc.exe version
```

Yeni bir DVC sürümüne geçerken yalnızca `dvc` paketini tek başına yükseltmeyin.
Önce `requirements-dvc.txt` içindeki `dvc[gdrive]`, `pydrive2`, `pyOpenSSL`,
`cryptography` ve `asyncssh` sürümlerini uyumlu şekilde güncelleyin. Ardından
yukarıdaki üç komutu çalıştırıp `pip check` sonucunun
`No broken requirements found.` olduğunu doğrulayın.

### Veri seti değişince DVC manifestini güncelleme

Bir veri setine dosya ekledikten, dosya değiştirdikten veya sildikten sonra
yalnızca değişen veri seti klasörünü yeniden ekleyin. Örnek:

```powershell
cd C:\Users\KDV\Desktop\Staj
.\.venv\Scripts\dvc.exe add -f "ThermalDlss\thermal database\thermal_dataset_split"
```

Diğer DVC hedefleri:

```text
organized_videos
ThermalDetection\images_thermal_train
ThermalDetection\images_thermal_val
ThermalDetection\video_thermal_test
ThermalDetection\thermal_live_detection\data\bu_tiv
ThermalDlss\thermal database\thermal_dataset_degraded
ThermalDlss\thermal database\thermal_dataset_split
```

Ardından veriyi Google Drive'a, değişen `.dvc` manifestini de GitHub'a gönderin:

```powershell
.\.venv\Scripts\dvc.exe push -j 1
git add "*.dvc"
git commit -m "data: update DVC dataset"
git push
```

Google Drive bağlantısı geçici olarak kesilirse aynı `dvc push -j 1` komutunu
yeniden çalıştırabilirsiniz. DVC daha önce yüklenen nesneleri atlayıp eksik
olanlardan devam eder.

Uzak depo ile eşitliği kontrol etmek için:

```powershell
.\.venv\Scripts\dvc.exe status -c
```

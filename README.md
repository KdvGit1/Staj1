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

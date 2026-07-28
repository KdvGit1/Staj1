# ThermalDlss: Native 640×512 Görüntüler İçin EDSR ×4 Revizyon Planı

**Belge türü:** Projeye özgü teknik inceleme, literatür araştırması ve deney planı  
**Tarih:** 28 Temmuz 2026  
**Hedef:** Görsel iyileştirme  
**Yeni kullanım profili:** Native `640×512 → 2560×2048`  
**Başlangıç noktası:** Mevcut eğitilmiş EDSR ×4 ağırlıkları  

## 1. Yönetici özeti

Bu revizyonda hedef, 16 adet native `640×512` görüntüyü `4×4` düzende
birleştirerek `2560×2048` pseudo-HR hedef oluşturmak, bu hedefi bicubic ile
`640×512` boyutuna indirip LR–HR çifti üretmek ve mevcut EDSR ağırlıklarını yeni
kullanım profiline uyarlamaktır.

Boyut hesabı doğrudur:

```text
16 × native görüntü:          640×512
4×4 pseudo-HR tuval:        2560×2048
bicubic ile ×4 küçültme:      640×512
EDSR ×4 çıktısı:             2560×2048
```

Buradaki “×4”, genişlik ve yüksekliğin ayrı ayrı dört katına çıkmasıdır; toplam
piksel sayısı `16×` olur.

Proje incelemesinin temel sonucu şudur:

> Mevcut model, eğitildiği sentetik dağılımda çalışıyor; native yüksek çözünürlüklü
> girdilerdeki kötü görünüm yalnızca piksel sayısından kaynaklanmıyor. Ana sorun,
> eğitim ve gerçek kullanım dağılımları arasındaki farktır.

Mevcut test setinde EDSR, bicubic’e göre yaklaşık `+0.62 dB PSNR` ve
`+0.0166 SSIM` kazanıyor. Buna karşılık `testFoto` içindeki DJI görüntülerinde
EDSR yüksek frekansları ve kontrastı gereğinden fazla büyütüyor, doyuma giden
siyah/beyaz pikselleri artırıyor ve yer yer düzenli ızgara benzeri yapılar
üretiyor. Bu iki gözlem birbiriyle çelişmez: model, gördüğü dağılımda başarılı
olup farklı bir kamera, sahne ve renk işleme zincirinde başarısız olabilir.

16 görüntülü mozaik yöntemi uygulanacaktır; ancak tek başına çözüm kabul
edilmeyecektir. Mevcut ağ tam evrişimlidir. Mozaikten yine `48×48` LR yamaları
alınırsa, dikişlerden uzaktaki eğitim örnekleri mevcut
`160×128 → 640×512` eğitimine büyük ölçüde eşdeğer olur. Mozaikten yarar görmek
için şu değişiklikler birlikte denenmelidir:

- daha büyük LR eğitim yamaları (`96`, `128` ve gerekirse `160`);
- dikişleri ayrı ölçen veya kayıptan maskeleyen örnekleme;
- mevcut ana veri setiyle hedef alan verisini birlikte kullanan replay;
- Sobel kenar kaybının ağırlığını düşüren ablation;
- mevcut PixelShuffle sonrasındaki ReLU katmanlarını sınayan mimari ablation;
- yalnız PSNR/SSIM değil, görsel artefakt ve hedef alan değerlendirmesi.

Ana öneri, **mevcut ağırlıklardan düşük öğrenme oranıyla fine-tuning** yapmaktır.
Sıfırdan eğitim yalnızca kontrol deneyi olmalıdır. İki yapay zekâ modelinin
rekabetine dayalı fusion yaklaşımı gelecek çalışma olarak korunacak, bu belgede
tasarlanmayacaktır.

## 2. Projede doğrulanan mevcut durum

Bu bölüm genel bir EDSR açıklaması değil, depodaki gerçek kod ve çıktılara
dayanmaktadır.

### 2.1 Model

`model.py` içindeki model:

- tek kanallı giriş ve tek kanallı çıkış kullanıyor;
- `64` özellik kanalı ve `16` residual blok içeriyor;
- Batch Normalization kullanmıyor;
- toplam yaklaşık `1.52 milyon` parametreye sahip;
- iki ardışık `×2` PixelShuffle katmanıyla toplam `×4` büyütüyor;
- her PixelShuffle sonrasında ReLU uyguluyor.

Model tam evrişimli olduğundan teorik olarak değişken boyutlu girdileri kabul
edebilir. Bu nedenle eğitimde `160×128` görmesi, çıkarımda `640×512` tensor
alamayacağı anlamına gelmez. Ancak tensor boyutunun kabul edilmesi ile o
görüntünün istatistiksel dağılımına genellenmesi farklı konulardır.

### 2.2 Kayıp fonksiyonu

`losses.py` içindeki toplam kayıp:

```text
L_total = L1 + 0.1 × L_Sobel
```

Son eğitim dönemindeki loglara göre Sobel bileşeni toplam kaybın yaklaşık
`%27`’sini oluşturuyor. Bu oran sentetik bicubic veride kenarları güçlendirebilir;
fakat zaten yüksek kontrastlı ve yüksek frekanslı native girdide aşırı keskinlik,
halelenme ve kırpılma riskini artırabilir. Bu bir hipotezdir ve kontrollü
ablation ile sınanmalıdır.

### 2.3 Veri üretimi

`lower_resolution.py`:

- HR görüntüyü `640×512` tek kanallı görüntü olarak okuyor;
- PIL bicubic ile `160×128` boyutuna indiriyor;
- çıktıyı JPEG kalite `95` ile kaydediyor.

Dolayısıyla mevcut eğitim bozulma modeli esas olarak:

```text
640×512 HR → bicubic küçültme → 160×128 LR → JPEG Q95
```

Gerçek kameradaki optik bulanıklık, sensör gürültüsü, non-uniformity, farklı
keskinleştirme, palette/tone mapping ve farklı sıkıştırma süreçleri bu modelde
temsil edilmiyor.

### 2.4 Dataset ve split yapısı

Projede kullanılabilecek veri miktarı:

| Bölüm | Görüntü | Video kimliği | HR boyutu | LR ×4 boyutu |
|---|---:|---:|---:|---:|
| Train | 12.505 | 123 | 640×512 | 160×128 |
| Validation | 1.563 | 19 | 640×512 | 160×128 |
| Test | 1.567 | 16 | 640×512 | 160×128 |
| **Toplam** | **15.635** | **158** |  |  |

Train, validation ve test video kimlikleri arasında çakışma bulunmadı. Bu iyi bir
özelliktir ve korunmalıdır. Yeni mozaikler de yalnız kendi split’i içindeki
görüntülerden oluşturulmalıdır.

### 2.5 Eğitim ve test sonucu

Mevcut loglarda:

- en iyi validation sonucu yaklaşık epoch `74`;
- validation PSNR: `29.769 dB`;
- validation SSIM: `0.762441`;
- tam test setinde bicubic: `29.9793 dB / 0.712734`;
- tam test setinde EDSR: `30.5971 dB / 0.729357`;
- EDSR farkı: `+0.6178 dB / +0.016623`;
- 1.567 test görüntüsünün yalnızca birinde EDSR PSNR’ı bicubic’ten düşük.

Sonuç: mevcut checkpoint bozuk veya tamamen başarısız değildir. Başarısı,
mevcut bicubic tabanlı eğitim/test dağılımına özeldir.

### 2.6 Native yüksek çözünürlüklü deneme hattı

`upscale_testfoto_x4.py`, native `640×512` görüntüyü doğrudan modele vererek
`2560×2048` çıktı üretir. Bellek için halo’lu tiled inference kullanır.
Modelin yaklaşık LR receptive-field yarıçapıyla uyumlu biçimde en az `36` piksel
halo zorunluluğu bulunur. Bu nedenle görülen temel kalite sorunu ilk bakışta
tile sınırı hatasından çok model/veri dağılımı sorunudur.

`testFoto` içindeki `_T.JPG` dosyaları RGB pseudo-color görünümündedir. Script
bunları `.convert("L")` ile tek kanala çevirir. Yani mevcut model:

1. renk paletli görseli luma kanalına indirger;
2. luma görüntüsünü büyütür;
3. gri çıktı üretir.

Eğer nihai ürün palette renklerini koruyacaksa bu hat ayrı tasarlanmalıdır.
Mevcut tek kanallı checkpoint için en güvenli ana rota, mümkünse monoton gri
termal katmanı büyütmek ve görsel renk haritasını SR işleminden sonra
uygulamaktır. Sadece RGB preview mevcutsa, üç kanallı model ayrı bir deney
koludur; mevcut modelin gövde ağırlıkları aktarılabilse de giriş/çıkış
katmanları doğrudan uyumlu değildir.

### 2.7 Ölçülen dağılım farkı

Projeden örneklenen yaklaşık `962` eğitim HR görüntüsü ile altı DJI test görüntüsü
karşılaştırıldığında:

| Ölçüm | Eğitim HR ortalaması | DJI görüntülerinin gri dönüşümü |
|---|---:|---:|
| Ortalama yoğunluk | 130.86 | 97.94 |
| Standart sapma | 55.23 | 67.30 |
| Ortalama yatay gradyan | 6.37 | 28.14 |
| Ortalama dikey gradyan | 6.78 | 22.89 |

DJI örneklerinin gradyan enerjisi eğitim görüntülerinden yaklaşık üç-dört kat
yüksektir. İncelenen bir örnekte EDSR, bicubic’e göre:

- gradyan büyüklüğünü daha fazla artırdı;
- Laplacian enerjisini yaklaşık `6.21`’den `15.24`’e çıkardı;
- doymuş/kırpılmış piksel oranını yaklaşık `%0.42`’den `%2.32`’ye çıkardı.

Bu, kullanıcının “belirgin ama güzel değil” gözlemini destekler: model ayrıntı
üretmekten çok mevcut yüksek frekansları ve ton uçlarını büyütüyor olabilir.

## 3. Problem tanımı

### 3.1 Eski ve yeni kullanım profilleri ayrılmalı

Projenin ilk kullanım profili:

```text
Hikvision fiziksel termal akış: 160×120 → 640×480
```

Yeni kullanım profili:

```text
Native görsel termal görüntü: 640×512 → 2560×2048
```

İki görev aynı `×4` ölçek faktörünü kullanır, fakat giriş dağılımları ve beklenen
çıktı karakteri aynı değildir. Tek checkpoint’i iki profile zorlamak yerine:

- `edsr_x4_sensor_lr.pth`: eski sensör-LR profili;
- `edsr_x4_native_visual.pth`: yeni native-görsel profili

olarak iki ayrı model sürümü tutulmalıdır.

### 3.2 Neden mevcut model yüksek piksel girdide kötü görünebilir?

Olası nedenler önem sırasıyla:

1. **Domain shift:** Kara/karayolu ağırlıklı gri termal eğitim verisi ile aerial
   pseudo-color DJI görüntüsü farklı dağılımlardır.
2. **Degradation gap:** Eğitim yalnız bicubic + JPEG Q95 görür; gerçek kamera
   görüntüsü farklı optik, ISP, keskinleştirme ve gürültü taşır.
3. **Aşırı kenar baskısı:** `0.1 × Sobel` kaybı native yüksek frekansları
   gereğinden fazla güçlendirebilir.
4. **Upsampler davranışı:** PixelShuffle sonrası ReLU ve öğrenilmiş faz
   farklılıkları düz alanlarda periyodik/ızgara artefaktına katkıda bulunabilir.
5. **Yetersiz bağlam:** LR eğitim yaması `48×48`, yaklaşık receptive-field
   çapından küçüktür. Bu kesin bir hata değildir; özgün EDSR çalışmalarında da
   küçük yamalar kullanılmıştır. Fakat bu proje için daha büyük yamalarla
   karşılaştırılmalıdır.
6. **Metrik-hedef uyumsuzluğu:** PSNR ve SSIM yükselse bile kullanıcı görsel
   kaliteyi düşük değerlendirebilir.

## 4. 16 görüntülü pseudo-HR üretim planı

### 4.1 Temel algoritma

Her örnek için:

1. Aynı split’ten 16 farklı `640×512` HR görüntü seç.
2. Mümkün olduğunda 16 farklı video kimliğinden örnekle.
3. Görüntüleri rastgele permütasyonla `4×4` ızgaraya yerleştir.
4. `2560×2048` pseudo-HR tuvali bellekte oluştur.
5. Tuvali anti-aliased bicubic ile `640×512` pseudo-LR’a indir.
6. Eğitime bütün görüntü yerine eşleşen LR/HR yamaları ver.
7. Mozaik dikiş koordinatlarını ve maskesini dataset ile birlikte döndür.

Koordinatlar:

```text
HR dikey dikişler:   x = 640, 1280, 1920
HR yatay dikişler:   y = 512, 1024, 1536

LR dikey dikişler:   x = 160, 320, 480
LR yatay dikişler:   y = 128, 256, 384
```

### 4.2 Sabit mozaik sayısı

Görüntüler yalnızca bir defa ve çakışmasız gruplanırsa:

| Split | Görüntü | Tam 16’lı mozaik | Artan görüntü |
|---|---:|---:|---:|
| Train | 12.505 | 781 | 9 |
| Validation | 1.563 | 97 | 11 |
| Test | 1.567 | 97 | 15 |

Train için sabit mozaik kaydetmek yerine on-the-fly rastgele gruplama önerilir.
Böylece aynı görüntüler farklı komşuluklarla görülebilir ve disk tüketimi
azalır. Validation ve test mozaikleri ise tekrarlanabilirlik için sabit manifest
ile oluşturulmalıdır.

### 4.3 Veri formatı

- Pseudo-HR hedef JPEG olarak tekrar kaydedilmemelidir.
- Cache gerekiyorsa kayıpsız PNG veya tensör tabanlı format kullanılmalıdır.
- Bicubic üretim fonksiyonu train/validation/test için aynı ve sürümlenmiş
  olmalıdır.
- Random seed ile mozaik manifestleri yeniden üretilebilir olmalıdır.
- Split’ler arasında görüntü veya video geçişi olmamalıdır.

### 4.4 Mozaik dikişlerinin yönetimi

Farklı sahnelerin fiziksel olarak ilgisiz sınırları gerçek bir termal sahnede
bulunmaz. Bicubic küçültme dikişin iki tarafını birbirine karıştırır ve model
olmayan bir kenarı öğrenebilir.

Üç deney kolu kurulmalıdır:

1. **Dikiş dışı:** LR yaması tamamen tek bir karonun içinde kalır.
2. **Dikiş maskeli:** Yama dikişe değebilir; fakat dikiş çevresindeki HR kayıp
   pikselleri hesap dışı bırakılır.
3. **Dikiş dahil:** Kontrol amacıyla dikişler normal piksel gibi kayba katılır.

Üçüncü kol ana yöntem olarak kullanılmamalıdır. Dikiş dahil modelde periyodik
çizgi/ızgara artarsa hipotez doğrulanmış olur.

### 4.5 Kritik sınırlama: mozaik tek başına yeni bilgi üretmez

Model yerel ve tam evrişimli olduğu için, dikişlerden uzakta:

```text
mozaikteki 640×512 bir karo
↓ bicubic ×4
160×128 LR karo
↓ EDSR
640×512 HR karo
```

ilişkisi mevcut veri hattındaki ilişkiyle aynıdır. Eğitim yine `48×48` LR
yamalarıyla yapılırsa, mozaik çoğunlukla yalnızca veri paketleme biçimini
değiştirir.

Bu nedenle 16’lı yaklaşımın araştırma değeri şu koşullardan gelir:

- `96×96`, `128×128` veya `160×128` gibi daha geniş LR bağlamı sunmak;
- tek batch örneğinde farklı sahne istatistiklerini birlikte göstermek;
- doğrudan `640×512` giriş boyutunda validation yapabilmek;
- dikiş artefaktlarını kontrollü bir değişken olarak ölçmek;
- ileride native yüksek çözünürlüklü gerçek hedefler geldiğinde aynı veri
  arayüzünü korumak.

## 5. Revize veri stratejisi

### 5.1 Ana veri kaynağı

Mevcut `15.635` görüntülük veri seti backbone bilgisini korumak için
kullanılacaktır. Büyük veri setini terk edip yalnız birkaç native test
görüntüsüne fine-tune etmek hızlı overfit ve catastrophic forgetting doğurur.

### 5.2 Eğitim karışımı

Başlangıç önerisi:

| Örnek türü | Başlangıç oranı | Amaç |
|---|---:|---|
| Mevcut tek-görüntü çiftleri | %50 | Eski yeteneği korumak |
| 16’lı pseudo-HR mozaik çiftleri | %30 | Büyük giriş ve geniş bağlam uyarlaması |
| Hedef kamera/sahne stili | %20 | Domain shift’i azaltmak |

Bu oran nihai gerçek değildir; validation sonucuna göre ayarlanacaktır. Hedef
kamera verisi azsa aynı görüntünün kopyalarını çoğaltmak yerine crop, flip,
rotation, hafif ton/gamma ve gerçekçi bozulma çeşitliliği kullanılmalıdır.

### 5.3 Hedef alan verisi

Yeni model hangi görüntülere uygulanacaksa hedef alan o olmalıdır:

- üretim Hikvision gri termal akışı ise DJI pseudo-color görüntüler ana hedef
  dağılım kabul edilmemeli;
- üretim DJI benzeri aerial görselse bu stile ait daha fazla native görüntü
  train ve bağımsız validation/test olarak toplanmalı;
- aynı çekimin ardışık kareleri farklı split’lere dağıtılmamalı;
- mümkünse kamera, palet, sahne ve gün/çekim koşulu metadata’sı tutulmalı.

### 5.4 Degradation modeli

İlk kontrollü deney yalnız bicubic kalmalıdır; böylece mozaik ve fine-tuning
etkisi izole edilir. İkinci aşamada hafif, hedefe uygun bozulma havuzu denenir:

- isotropic ve anisotropic Gaussian blur;
- küçük sensör gürültüsü;
- hafif JPEG kalite değişimi;
- bicubic/bilinear/area resize seçimi;
- hafif ringing/overshoot örnekleri;
- kamera zincirine uyuyorsa tone/gamma değişimi.

Real-ESRGAN veya BSRGAN’daki ağır bozulma aralıkları doğrudan kopyalanmamalıdır.
Girdi zaten temiz native görüntüyse aşırı bozulma modeli, modele olmayan
kusurları düzeltmeyi öğretip yeni artefakt ürettirebilir.

## 6. Model revizyonu ve deney matrisi

### 6.1 Kontroller

| Kod | Deney | Başlangıç | Veri | Amaç |
|---|---|---|---|---|
| B0 | Mevcut baseline | mevcut checkpoint | mevcut test | Referansı dondur |
| B1 | Native doğrudan inference | mevcut checkpoint | hedef native set | Mevcut şikâyeti ölç |
| M0 | Sadece mozaik paketleme | checkpoint | 16’lı, patch 48 | Mozaiğin tek başına etkisi |
| M1 | Büyük bağlam | checkpoint | 16’lı, patch 96/128 | Bağlam hipotezi |
| M2 | Mixed replay | checkpoint | eski + mozaik + hedef | Domain uyarlaması |
| M3 | Düşük Sobel | checkpoint | M2 verisi | Aşırı keskinlik hipotezi |
| M4 | Upsampler ablation | gövde checkpoint | M2 verisi | Izgara/clip hipotezi |
| S0 | Sıfırdan eğitim | random | M2’nin aynısı | Fine-tuning kontrolü |

Ana aday `M2 + M3` birleşimidir. `M4` ancak ilk üç deney yetersiz kalırsa
devreye alınmalıdır.

### 6.2 Patch boyutu

Mevcut LR patch `48×48`, HR patch `192×192`’dir. Modelin yaklaşık LR
receptive-field çapı `73` piksel civarındadır. Önerilen tarama:

| LR patch | HR patch | Kullanım |
|---:|---:|---|
| 48×48 | 192×192 | Mevcut kontrol |
| 96×96 | 384×384 | İlk önerilen fine-tune |
| 128×128 | 512×512 | Daha geniş bağlam |
| 160×128 | 640×512 | Tek karonun tamamı, bellek uygunsa |

Tam `640×512` LR mozaikten `2560×2048` HR’a geri yayılım, 64 kanallı ara
aktivasyonlar nedeniyle 12 GB sınıfı bir GPU’da büyük olasılıkla pratik
değildir. Patch tabanlı eğitim, AMP ve gradient accumulation kullanılmalıdır.

### 6.3 Fine-tuning başlangıç ayarı

İlk aday konfigürasyon:

```text
checkpoint yükleme: yalnız model ağırlıkları
optimizer: yeni Adam
başlangıç öğrenme oranı: 1e-5
LR patch: 96×96
batch: VRAM’e göre 1–4
gradient accumulation: etkin batch 8–16 olacak şekilde
AMP: açık
L1 ağırlığı: 1.0
Sobel ağırlığı: 0.01
epoch: 30–60 başlangıç aralığı
early stopping: hedef validation metriğine göre
```

Mevcut `--resume`, optimizer ve scheduler durumunu da yükler. Yeni görev için
bu davranış önerilmez. “Pretrained weights” ve “resume interrupted run” iki ayrı
seçenek olmalıdır.

### 6.4 Sobel ağırlığı ablation

Şu değerler aynı seed ve veri manifestiyle karşılaştırılmalıdır:

```text
0.00, 0.01, 0.03, 0.10
```

Seçim yalnız PSNR’a göre yapılmamalıdır. Kenar haleleri, clipping, düz alan
ızgarası ve kullanıcı görsel tercihi birlikte değerlendirilmelidir.

### 6.5 Upsampler ablation

Depodaki her PixelShuffle sonrasında ReLU bulunur. Resmî EDSR PyTorch
uygulamasındaki upsampler varsayılan olarak aktivasyon kullanmaz. Bu farkın
kalite üzerindeki etkisi test edilmelidir.

Önerilen sıra:

1. Mevcut upsampler korunarak veri/loss fine-tuning.
2. PixelShuffle sonrası ReLU kaldırılarak gövde ağırlıklarının aktarılması.
3. Gerekirse yeni upsampler için ICNR initialization.
4. Gerekirse resize-convolution veya bicubic residual çıkış kolu.

İkinci adımdan itibaren optimizer yeniden kurulmalı; yeni/değişen upsampler
katmanları gövdeden daha yüksek öğrenme oranı alabilir. ICNR, var olan öğrenilmiş
ağırlıkları otomatik düzeltmez; özellikle yeniden başlatılan upsampler için
anlamlıdır.

## 7. Değerlendirme protokolü

### 7.1 Üç ayrı test seti

1. **Mevcut eşleşen test:** 1.567 görüntü, eski yeteneğin korunmasını ölçer.
2. **Sabit pseudo-HR test:** 97 adet 16’lı mozaik, gerçek referanslı
   `640×512 → 2560×2048` değerlendirme sağlar.
3. **Native hedef test:** Kamera/sahne bazında ayrılmış gerçek `640×512`
   girdiler, görsel kullanım kalitesini ölçer; gerçek `2560×2048` GT yoksa
   referanssız değerlendirilir.

Pseudo-HR test, modelin gerçek sensör çözünürlüğünü geri kazandığını kanıtlamaz;
yalnız oluşturulan sentetik görevi ne kadar iyi çözdüğünü ölçer.

### 7.2 Referanslı metrikler

- PSNR;
- SSIM;
- mümkünse LPIPS;
- iç bölge ve dikiş çevresi için ayrı metrik;
- bicubic’e göre görüntü başına fark dağılımı;
- median, `%5` ve `%95` yüzdelikleri.

### 7.3 Artefakt metrikleri

Native hedef sette:

- `0` ve `255` yoğunluklarında clipping oranı;
- gradyan büyüklüğü oranı: `SR / bicubic`;
- Laplacian enerji oranı;
- dört PixelShuffle fazındaki ortalama/variance farkı;
- düz bölgelerde periyodik 2×2 veya 4×4 enerji;
- tile sınırı boyunca süreksizlik;
- girişe tekrar küçültüldüğünde cycle consistency.

### 7.4 Görsel değerlendirme

Her görüntü için aynı zoom ve tone mapping ile:

```text
Native giriş | Bicubic ×4 | Eski EDSR | Yeni aday
```

yan yana gösterilmelidir. Dosya adı ve yöntem etiketi gizlenerek en az üç
değerlendiriciyle tercih testi yapılması önerilir. Sorular ayrı tutulmalıdır:

- Kenarlar daha okunaklı mı?
- Gürültü veya halelenme arttı mı?
- Düz yüzeylerde ızgara oluştu mu?
- Görüntü doğal görünüyor mu?
- Sahte ayrıntı şüphesi var mı?

### 7.5 Başarı ölçütü

Yeni checkpoint ancak şu şartlarla kabul edilmelidir:

- mevcut test setinde ciddi gerileme olmaması;
- pseudo-HR testte bicubic ve B0’dan daha iyi sonuç;
- native hedef görüntülerde clipping/ızgara artmaması;
- kör görsel testte B0 ve bicubic’e karşı çoğunluk tercihi;
- tiled ve tek-parça inference arasında iç bölgede anlamlı fark olmaması.

## 8. Kodda gerekli değişiklikler

Bu belge uygulama tasarımıdır; değişiklikler ayrı commit’lerde yapılmalıdır.

### 8.1 `dataset.py`

- `MosaicThermalSRDataset` veya eşdeğer mod eklenmeli;
- video kimliğine göre split-local 16’lı örnekleme yapılmalı;
- mozaik ve crop aynı anda bellekte oluşturulmalı;
- `patch_size`, seam mode ve degradation config parametre olmalı;
- validation/test için sabit manifest desteği eklenmeli.

### 8.2 `train.py`

- `--pretrained_weights` ile `--resume` ayrılmalı;
- `--patch_size`, `--edge_weight`, `--mosaic_probability` eklenmeli;
- mixed sampler/replay desteği eklenmeli;
- AMP, gradient accumulation ve seed kontrolü eklenmeli;
- deney konfigürasyonu checkpoint yanında JSON/YAML olarak saklanmalı;
- her deney ayrı çıktı klasörüne yazılmalı.

### 8.3 `losses.py`

- Sobel ağırlığı CLI/config üzerinden yönetilmeli;
- isteğe bağlı seam mask kabul edilmeli;
- daha ileri perceptual loss, ilk deneylerde zorunlu tutulmamalı.

### 8.4 `model.py`

- mevcut mimari checkpoint uyumluluğu için değişmeden korunmalı;
- `post_shuffle_relu=True/False` parametresiyle ablation yapılmalı;
- yeni upsampler denenirse eski gövdeyi yükleyen kontrollü ağırlık aktarımı
  yazılmalı.

### 8.5 `evaluate.py`

- model konfigürasyonunu checkpoint’ten yüklemeli;
- mevcut test, pseudo-HR test ve native hedef test için ayrı modlar sunmalı;
- seam/interior, clipping ve faz artefaktlarını raporlamalı;
- sonuçları CSV/JSON ve sabit görsel grid olarak kaydetmeli;
- x16 cascade testi, yeni native `×4` hedefiyle karıştırılmamalı.

## 9. Uygulama sırası

### Faz 0 — Baseline’ı dondur

- Mevcut checkpoint hash’i ve config’i kaydet.
- Mevcut test sonuçlarını yeniden üret.
- Native hedef sette B0 ve bicubic çıktıları oluştur.
- Clipping, gradient, Laplacian ve faz artefaktlarını raporla.

### Faz 1 — 16’lı veri hattı

- On-the-fly `4×4` pseudo-HR üret.
- Sabit validation/test manifestlerini oluştur.
- Boyut, hizalama ve dikiş maskesi unit testlerini yaz.
- M0 ile mozaiğin tek başına etkisini ölç.

### Faz 2 — Ağırlıktan fine-tuning

- M1: patch `96` ve `128`.
- M2: mevcut veri + mozaik + hedef alan replay.
- M3: Sobel ağırlığı taraması.
- En iyi iki adayı native görsel testte karşılaştır.

### Faz 3 — Mimari ablation

- Gerekirse post-PixelShuffle ReLU’yu kaldır.
- Gerekirse ICNR veya resize-convolution dene.
- Aynı veri ve seed ile yalnız mimari farkını ölç.

### Faz 4 — Sıfırdan kontrol

- En iyi veri/loss ayarıyla S0 eğit.
- Fine-tuning’in yakınsama, kalite ve veri verimliliği avantajını raporla.

## 10. Riskler ve karşı önlemler

| Risk | Sonuç | Karşı önlem |
|---|---|---|
| Mozaik dikişlerini öğrenme | Yapay çizgi/ızgara | Seam mask ve ayrı metrik |
| Mozaik aynı yerel görevi tekrarlar | Kazanç oluşmaz | Büyük patch ve M0 kontrolü |
| Az hedef veriyle overfit | Genelleme düşer | Ana dataset replay |
| Domain shift sürer | Native sonuç kötü kalır | Hedef kamera/stil verisi |
| Sobel aşırı keskinleştirir | Halo ve clipping | Ağırlık ablation |
| PixelShuffle faz artefaktı | 2×2/4×4 pattern | Faz metriği ve upsampler ablation |
| Full-frame training OOM | Eğitim durur | AMP, patch, accumulation |
| PSNR yükselir ama görüntü kötüleşir | Yanlış model seçimi | Kör görsel test + artefakt metriği |
| RGB palette luma’ya dönüşür | Görsel bilgi kaybı | Gri-önce/renklendir-sonra veya RGB kolu |
| Eski profile zarar | 160×120 performansı düşer | Ayrı checkpoint profilleri |

## 11. Literatür değerlendirmesi

### 11.1 EDSR ve mevcut mimari

Lim ve arkadaşlarının EDSR çalışması, residual SR ağlarında gereksiz modülleri
çıkarıp residual öğrenmeyi ölçeklendirerek güçlü bir piksel-doğruluk baseline’ı
kurar. Depodaki yaklaşık `1.52M` parametreli model, resmî EDSR baseline ×4
boyutuna çok yakındır. Bu nedenle ilk tercih mimariyi tamamen atmak değil,
mevcut ağırlıkları kontrollü uyarlamaktır.

- [Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 2017](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.html)
- [Resmî EDSR-PyTorch uygulaması](https://github.com/sanghyun-son/EDSR-PyTorch)
- [Resmî upsampler kaynak kodu](https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/model/common.py)

### 11.2 Sentetik ve gerçek bozulma farkı

RealSR, BSRGAN ve Real-ESRGAN çizgisinin ortak bulgusu, yalnız bicubic ile
üretilen LR görüntülerde eğitilen modellerin gerçek kamera görüntülerinde
genelleme sorunu yaşamasıdır. Bu proje de aynı yapıyı gösteriyor: sentetik testte
başarı, native hedefte görsel başarıyı garanti etmiyor.

- [Toward Real-World Single Image Super-Resolution: RealSR, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Cai_Toward_Real-World_Single_Image_Super-Resolution_A_New_Benchmark_and_a_ICCV_2019_paper.html)
- [Designing a Practical Degradation Model for Deep Blind SR: BSRGAN, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Designing_a_Practical_Degradation_Model_for_Deep_Blind_Image_Super-Resolution_ICCV_2021_paper.pdf)
- [Real-ESRGAN, ICCVW 2021](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.pdf)

Bu literatür ağır bir Real-ESRGAN reçetesinin doğrudan alınmasını değil, gerçek
hedefe benzeyen bozulma aralığının ölçülerek tasarlanmasını destekler.

### 11.3 İç örnekler ve öz-denetimli uyarlama

ZSSR, tek bir görüntü içindeki ölçek tekrarlarının görüntüye özel uyarlamada
kullanılabileceğini gösterir. 16’lı pseudo-HR planı aynı fikirle akrabadır;
ancak farklı sahneleri yan yana koymak yeni fiziksel ayrıntı üretmez. Bu nedenle
mozaik “gerçek HR sensör hedefi” değil, kontrollü sentetik eğitim tuvalidir.

- [Zero-Shot Super-Resolution Using Deep Internal Learning, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Shocher_Zero-Shot_Super-Resolution_Using_CVPR_2018_paper.html)

### 11.4 SR’ye özgü veri artırma

CutBlur çalışması, restorasyon problemlerinde uzamsal ilişkiyi bozan genel amaçlı
augmentasyonların zarar verebildiğini ve SR’ye özgü artırmanın önemli olduğunu
gösterir. Bu, mozaik dikişlerini kontrolsüz biçimde normal veri gibi eğitime
katmama kararını destekler.

- [Rethinking Data Augmentation for Image Super-Resolution, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yoo_Rethinking_Data_Augmentation_for_Image_Super-resolution_A_Comprehensive_Analysis_and_CVPR_2020_paper.html)

### 11.5 PixelShuffle ve checkerboard

Sub-pixel convolution verimlidir; fakat başlatma ve öğrenilmiş alt-piksel
filtreleri fazlar arasında farklı davranırsa checkerboard oluşabilir. ICNR,
özellikle yeni başlatılan sub-pixel katmanlarında bu riski azaltmak için
önerilmiştir. Projede önce faz metriğiyle artefakt doğrulanmalı, sonra mimari
değiştirilmelidir.

- [Checkerboard Artifact Free Sub-Pixel Convolution / ICNR](https://arxiv.org/abs/1707.02937)

### 11.6 Algısal kalite ve distortion

Blau ve Michaeli, distortion ölçüleri ile algısal kalite arasında temel bir
ödünleşim olduğunu gösterir. Bu nedenle görsel iyileştirme hedefinde yalnız
PSNR/SSIM ile seçim yapmak yeterli değildir. Bu proje için clipping, halo,
checkerboard ve kör kullanıcı tercihi doğrudan seçim ölçütü olmalıdır.

- [The Perception-Distortion Tradeoff, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html)

### 11.7 Termal SR literatürü

Termal SR literatürü, görünür görüntü SR’den farklı olarak düşük kontrast,
sınırlı yüksek frekans, sensöre özgü gürültü ve fiziksel termal tutarlılık
sorunlarını vurgular. Real-InfraredSR çalışması gerçek optik zoom ile eşlenmiş
termal çiftler kurar; bu, gelecekte mümkünse gerçek paired LR/HR toplamanın
sentetik bicubic hedeflerden daha güçlü olduğunu gösterir. TISR challenge
çalışmaları da sentetik ve gerçek senaryoların ayrı raporlanması gereğini
destekler.

- [Real-InfraredSR, Optics Express 2023](https://doi.org/10.1364/OE.496484)
- [Infrared Image Super-Resolution: A Systematic Review and Future Trends, IEEE JSTARS 2025](https://doi.org/10.1109/JSTARS.2025.3614673)
- [Thermal Image Super-Resolution Challenge, CVPRW 2021](https://openaccess.thecvf.com/content/CVPR2021W/PBVS/html/Rivadeneira_Thermal_Image_Super-Resolution_Challenge_-_PBVS_2021_CVPRW_2021_paper.html)

## 12. Nihai teknik karar

Bu proje için önerilen yol:

1. 16 adet `640×512` görüntüyle `4×4` pseudo-HR veri hattını kur.
2. Mozaiği ana veri setinin yerine değil, mixed replay’in bir parçası olarak
   kullan.
3. Fine-tuning’e mevcut checkpoint’in yalnız model ağırlıklarından başla.
4. İlk adayda LR patch’i `96×96`, öğrenme oranını `1e-5`, Sobel ağırlığını
   `0.01` kullan.
5. `48/96/128` patch ve `0/0.01/0.03/0.1` Sobel ablation’larını çalıştır.
6. Hedef kamera/sahne görüntülerini bağımsız validation/test olarak ayır.
7. Eski sensör-LR profili ile yeni native-görsel profili için ayrı checkpoint
   üret.
8. Sorun devam ederse post-PixelShuffle ReLU ve upsampler mimarisini sınayan
   ablation’a geç.
9. Sıfırdan eğitimi yalnız kontrol olarak çalıştır.

Bu tasarım, kullanıcının 16 görüntülü yaklaşımını korur; fakat başarıyı sadece
canvas boyutuna bağlamaz. Projedeki mevcut büyük dataset, güçlü başlangıç
ağırlıkları ve hedef alana özgü veriler birlikte kullanılır.

## 13. Gelecek çalışma: rekabetçi fusion

Bu aşama tamamlandıktan ve tek görüntülü EDSR baseline’ı güvenilir biçimde
ölçüldükten sonra, iki yapay zekâ modelinin tamamlayıcı/rekabetçi çıktılarından
yararlanan bir fusion modeli araştırılacaktır. Bu konu ayrı bir faz olarak
belgelenecek; mevcut deneylerin kapsamına dahil edilmeyecektir.


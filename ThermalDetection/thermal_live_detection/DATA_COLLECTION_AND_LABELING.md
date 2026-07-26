# Kamera Verisi Toplama ve Etiketleme Planı

Bu aşamanın hedefi çok kare toplamak değil, mevcut modelin bilmediği kamera
dağılımını temsil eden ve birbirinin kopyası olmayan insan kontrollü örnekler
üretmektir.

## 1. Önce sabitlenecek kamera koşulları

Her veri oturumunda şu bilgiler `session.json` yanında ayrıca not edilmelidir:

- kamera/konum kimliği ve kanal (`201` veya `202`);
- çözünürlük, FPS, codec ve varsa bitrate;
- termal palet, AGC/kontrast modu ve sıcaklık aralığı;
- saat aralığı, hava, yağış/sis ve sahne yoğunluğu;
- kamera açısı/zoom değiştiyse bu değişiklik.

Otomatik kazanç veya termal palet değişirse görüntü dağılımı da değişir.
Mümkünse üretimde kullanılacak ayarlar veri toplarken sabit tutulmalıdır.

## 2. İlk toplama turu

İlk turda `manual` ile birkaç kısa kontrol oturumu, ardından `hybrid` ile gerçek
trafik oturumları önerilir:

```powershell
.\thermal_live_detection\run_live.ps1 `
    -CameraIp "KAMERA_IP" -DetectorInput edsr -CaptureMode manual

.\thermal_live_detection\run_live.ps1 `
    -CameraIp "KAMERA_IP" -DetectorInput edsr -CaptureMode hybrid
```

`hybrid` şu kareleri önceliklendirir:

- güveni `uncertainty-low` ile `uncertainty-high` arasında olan zor tespitler;
- yeterince yüksek güvenli olaylar;
- belirli aralıkta gelen ve son kayıttan görsel olarak farklı kareler.

Başlangıç ayarları:

```text
capture interval     10 s
event interval        2 s
event confidence     0.25
uncertainty band     0.20 - 0.45
novelty threshold    3.0 ortalama gri seviye
```

Bunlar kesin eşikler değildir. İlk oturumdan sonra tekrar oranı çok yüksekse
interval/event aralıkları ve novelty eşiği artırılmalıdır. Küçük veya kaçırılan
nesneler hiç kayda girmiyorsa yalnızca tespit-tetiklemeli kayıt yetmez; interval
ve manuel kayıt mutlaka korunmalıdır.

İlk çevrim için yüzlerce birbirinden farklı, insan tarafından gözden geçirilmiş
kare; binlerce ardışık benzer kareden daha değerlidir. Özellikle mevcut modelin
zayıf sınıfı olan `bike_motorcycle`, küçük/uzak nesne, kısmi örtülme, sahne
kenarı, sıcak arka plan ve boş/negatif kareler bilinçli olarak aranmalıdır.

## 3. Hangi görüntü etiketlenecek?

Kanonik eğitim görüntüsü `detector_inputs/*.png` olmalıdır. Bu görüntü
640x512'dir; 640x480 EDSR/bicubic içerik üst ve altta letterbox dolgusu taşır.
Kutular yalnızca bu görüntünün koordinat sisteminde tutulur. Aynı kimliğe sahip
şu dosyalar etiketçiye referans olarak gösterilebilir:

- `source_frames`: EDSR'nin üretmediği gerçek kamera bilgisini kontrol etmek;
- `native_frames`: modelin 160x120 gerçek girdisini görmek;
- `sr_frames`: büyütülmüş ayrıntıyı görmek;
- `previews`: mevcut model önerisini hızlı incelemek.

EDSR yeni fiziksel bilgi yaratamaz ve kenar/tekstür artefaktı üretebilir.
Etiketçi yalnızca SR görüntüsünde nesneye benzeyen, fakat kaynak/native karede
desteklenmeyen bir şekli gerçek nesne olarak etiketlememelidir.

## 4. İnsan etiketleme kuralları

Sınıf sözlüğü değiştirilmeden korunur:

```text
0 person
1 bike_motorcycle
2 car
```

Önerilen kurallar:

- kutu görünür nesneyi sıkı biçimde çevreler; aşırı arka plan içermez;
- görüntü dışına taşan nesnede kutu görüntü sınırında kesilir;
- kısmen örtülü ama sınıfı ayırt edilebilen nesne etiketlenir;
- ayırt edilemeyecek sıcak lekeler zorla bir sınıfa atanmaz;
- motosiklet/bisiklet üzerindeki kişi ayrıca `person`, araç ise
  `bike_motorcycle` olarak etiketlenir;
- boş sahne için aynı adlı, içi boş `.txt` dosyası korunur;
- modelin kutu önermemesi, nesnenin etiketsiz bırakılması için gerekçe değildir;
- modelin yanlış önerisi silinir, kutusu zayıfsa düzeltilir.

`pseudo_labels` sadece ön-etikettir. CVAT, Label Studio veya benzeri bir araçta
görüntü ile birlikte içe aktarılıp her kare insan tarafından onaylanmadan eğitim
verisine eklenmemelidir. Onaylı çıktılar ayrı `reviewed_labels` dizininde
tutulmalı; ham `pseudo_labels` üzerine yazılmamalıdır.

## 5. Kalite kontrol

Etiketleme tamamlanınca:

1. Tüm sınıflardan rastgele örnekler ikinci kez incelenir.
2. Çok küçük, çok büyük ve görüntü sınırındaki kutular ayrıca filtrelenip
   kontrol edilir.
3. Boş etiketli görüntülerde kaçırılmış nesne aranır.
4. Sınıf sayıları ve kutu boyutu dağılımları raporlanır.
5. `bike_motorcycle` örneklerinin tek bir kısa oturumdan gelmediği doğrulanır.

Önceki değerlendirmede test çeşitliliği sınırlıydı ve `bike_motorcycle`
örneklerinin oturum/video çeşitliliği özellikle kritikti. Yeni veriyi sırf
sayısal denge için aynı olayın ardışık kareleriyle doldurmak gerçek çeşitlilik
sağlamaz.

## 6. Train/validation/test ayrımı

Ardışık kareler rastgele bölünmemelidir. Aynı kişi/araç birkaç saniye boyunca
hem train hem validation'a düşerse ölçüm yapay biçimde yükselir.

Ayrımın birimi kare değil `kamera + gün + oturum/zaman bloğu` olmalıdır:

- train: oturumların yaklaşık %70-80'i;
- validation: tamamen ayrı oturumların yaklaşık %10-15'i;
- test: en sona kilitlenen, tamamen ayrı oturumların yaklaşık %10-15'i.

Yüzdelerden daha önemli olan aynı olayın iki bölüme sızmamasıdır. Test kümesi
geliştirme sırasında tekrar tekrar seçilmemeli ve etiketi eğitime
karıştırılmamalıdır.

## 7. EDSR faydasını ölçme

Kamerada gerçek 640x480 yüksek çözünürlüklü eş görüntü bulunmadığı için canlı
veride yalnızca PSNR/SSIM ile karar verilemez. Aynı kaydedilmiş `native_frames`
üzerinde üç ayrı türev oluşturulmalıdır:

1. EDSR 640x480,
2. bicubic 640x480,
3. mümkünse RTSP kaynak karesi.

Tek bir insan etiketli test listesiyle her yol için aynı YOLO ayarlarında:

- sınıf bazlı precision, recall, AP50 ve AP50-95;
- özellikle `bike_motorcycle` recall;
- görüntü başına yanlış pozitif;
- EDSR ve YOLO gecikmesi/FPS

karşılaştırılmalıdır. Sonuç EDSR lehine değilse EDSR görsel iyileştirme olarak
korunabilir, fakat dedektör girdisi `bicubic` veya `source` seçilebilir.

## 8. İyileştirme çevrimi

İlk onaylı kamera veri setinden sonra önerilen sıra:

1. Mevcut modeli yeni kilitli test oturumunda ölç.
2. Hataları `kaçırma`, `yanlış sınıf`, `yanlış pozitif`, `kötü kutu` olarak
   gruplandır.
3. En çok hata üreten koşullardan yeni örnek seç.
4. Eski eğitim verisi ile yeni kamera verisini birlikte fine-tune et; yalnızca
   yeni veriye aşırı uyumdan kaçın.
5. Aynı kilitli testte EDSR/bicubic/source yollarını yeniden karşılaştır.

Bu aktif öğrenme çevrimi, internette yeni termal veri seti bulunamadığı durumda
modeli gerçek hedef kameraya uyarlamanın en verimli yoludur.

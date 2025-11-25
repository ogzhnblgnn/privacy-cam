import cv2
import mediapipe as mp
import numpy as np
import face_recognition

class PrivacyCam:
    def __init__(self):
        # kamera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        
        #Mediapipe
        
        # aşama 1 için yüz tespiti: bir yüz bulduğundan %50 emin ise yüzü tespit eder
        
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5) 
        
        # aşama 2 için segmentasyon: pikselleri insan ve arkaplan ayırmak için model
        
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1) 
        #uzaktan çekim ve hız için model_selection-> 1

        #durum değişkenleri
        self.mode = 0  # başlangıç modu -> 0: görüntüde işlem yapılmıyor. 1,2,3 ise aşamaları temsil eder
        self.blur_amount = 21 # bulanıklık
        self.blur_method = "Gaussian" # default bulanıklaştırma yöntemi
        
        #kişi değişkenleri (aşama 3 için kaydedilecek kişi)
        self.target_face_encoding = None 
        self.target_known = False

        # arayüz
        cv2.namedWindow("detection")
        
        #trackbar
        cv2.createTrackbar("Mod (0-3)", "detection", 0, 3, self.set_mode)
        cv2.createTrackbar("Blur Gucu", "detection", 10, 50, self.set_blur)

    def set_mode(self, val):
        self.mode = val
        
    def set_blur(self, val):
        # kernel boyutu tek sayı olmalıdır ki, komşu piksellere göre bir atama yapabilelim
        if val % 2 == 0: val += 1
        self.blur_amount = max(3, val) #en az 3x3


        #seçilen yönteme göre tüm görüntünün bulanık bir kopyasını oluşturup daha sonra maskeleme yaparken arkaplan olarak kullanıyoruz.

    def apply_blur(self, image):
        ksize = self.blur_amount
        
        if self.blur_method == "Gaussian":
            # gaussianBlur Piksellerin ortalamasını alarak yumuşak bir geçiş sağlar
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
            
        elif self.blur_method == "Median":
            # medianBlur komşu piksellerin ortanca değerini alarak çalışır, gürültüyü yok edip kenarları korur
            return cv2.medianBlur(image, ksize)
            
        elif self.blur_method == "Pixelate":
            # pixelation: çözünürlüğü düşürüp tekrar yükseltir
            h, w = image.shape[:2]
            
            # mozaik boyutu, blur arttıkaç oran küçülsün ki mozaikler büyüsün.
            ratio = max(2, 105 - ksize * 2) 
            
            # resmi küçültüp tekrar eski boyutuna upscale ediyoruz. Inter_Nearest da en yakın komşularını baz alarak pikselleri yumuşatmadan kare kare büyütür, böylece mozaik görüntü elde edilir
            temp = cv2.resize(image, (w // ratio, h // ratio), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        
        elif self.blur_method == "Bilateral":
            # bilateralFilter kenarları koruyarak yumuşatma yapar, d ise komşuluk çapını belirler
            sigma = ksize * 2
            return cv2.bilateralFilter(image, d=15, sigmaColor=sigma, sigmaSpace=sigma)
            
        return image

    #aşama 1 için yüzleri bulup, maskede o bölgeyi 0 yaparak siyah bölgeleri net bırakıyoruz
    def face_preservation(self, frame, blurred_frame):

       #bgr->rgb dönüşümü 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_detection.process(rgb_frame)
        
        # bulanıklaştırılacak alan için beyaz bir maske oluşturuyoruz
        mask = np.ones(frame.shape[:2], dtype="uint8") * 255 
        
       #mediapipe koordinatlarını yükseklik ve genişlik değerlerine dönüştürme 
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # net gösterilecek alan için siyah bir maske çiziyoruz
                cv2.rectangle(mask, (x, y), (x + width, y + height), 0, -1)

        # tek kanallı maske için 3 kanala çeviriyoruz
        mask_3ch = cv2.merge([mask, mask, mask])
        
        # maske beyaz ise blurred_frame'den piksel alır, siyah ise frame'den piksel alır
        output = np.where(mask_3ch == 255, blurred_frame, frame)
        return output

    #vücut segmentasyonu
    def body_preservation(self, frame, blurred_frame):

        #rgb dönüşümü
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #segmentasyon
        results = self.segmentation.process(rgb_frame)
        
        # eğer sonuç dönmezse direkt bulanık resmi verir
        if results.segmentation_mask is None:
            return blurred_frame

        # mediapipe her piksel için insan olma ihtimalini verir. 0.5 den büyük ise insan, değil ise arkaplandır
        # maskeyi 3 kanallı hale getiriyoruz
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5
        
        # true ise frame, false ise blurred_frame
        output = np.where(condition, frame, blurred_frame)
        return output

        #kişi tanıma
    def stage_3_target_preservation(self, frame, blurred_frame):
        
        # Eğer henüz 'k' tuşuna basıp birini kaydetmediysek uyarı ver.
        if not self.target_known:
            cv2.putText(frame, "KAYITLI KISI YOK! 'k' tusuna basin.", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return blurred_frame 


        # face_recognition'ın hızlı çalışması için resmi 0.25 boyutunda kullanıyoruz
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # resimdeki yüzlerin yerini bulup vektörlerini alıyoruz
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # başlarken her yer beyaz maskeleniyor
        mask = np.ones(frame.shape[:2], dtype="uint8") * 255 

        # bulunan her yüz için görüntülenen yüz ile hedeflenen yüz karşılaştırılıyor. 0.5 tolerans değeri kullandım
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            
            matches = face_recognition.compare_faces([self.target_face_encoding], face_encoding, tolerance=0.5)

            #eşleşme bulunduğunda resmi orijinal boyutuna çeviriyoruz
            if True in matches:
                
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # tanınan kişi görüntüsü net yapılır
                cv2.rectangle(mask, (left, top), (right, bottom), 0, -1)

        mask_3ch = cv2.merge([mask, mask, mask])
        output = np.where(mask_3ch == 255, blurred_frame, frame)
        return output

        #ana program, kameradan sürekli görüntü alır ve seçilen modu çalıştırır
    def run(self):

        print("Program Başladı. Çıkış için 'q' tuşuna basın.")

        #açık oldukça kameradan frame okur, hata alırsa çalışmayı durdurur.
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            # ayna etkisi için flip kullanıyoruz
            frame = cv2.flip(frame, 1) 

            #bulanık görüntü hazır edilsin ki modlar arası geçişte bekleme yaşamayalım
            blurred_frame = self.apply_blur(frame)
            
            output = frame #default görüntü

            # modlar arasında geçişler
            if self.mode == 0:
                cv2.putText(output, "Mod 0: Normal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            elif self.mode == 1:
                output = self.face_preservation(frame, blurred_frame)
                cv2.putText(output, "Mod 1: Yuz Tanima", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            elif self.mode == 2:
                output = self.body_preservation(frame, blurred_frame)
                cv2.putText(output, "Mod 2: Vucut Tanima", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            elif self.mode == 3:
                output = self.stage_3_target_preservation(frame, blurred_frame)
                cv2.putText(output, "Mod 3: Kisi Tanima", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(output, f"Blur Yontemi: {self.blur_method}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("detection", output)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'): # 'q' tuşuna basılırsa çık
                break
                
            elif key == ord('b'): # 'b' tuşuna basılırsa blur yöntemini değiştir
                if self.blur_method == "Gaussian":
                    self.blur_method = "Median"
                elif self.blur_method == "Median":
                    self.blur_method = "Pixelate"
                elif self.blur_method == "Pixelate":
                    self.blur_method = "Bilateral"
                else:
                    self.blur_method = "Gaussian"
                print(f"Yöntem değiştirildi: {self.blur_method}")
                
            elif key == ord('k'): # 'k' tuşuna basılırsa yüz kaydet

                #face_recognition ile yüz tanımlama
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb)
                # yüzün vektör haritası
                encodings = face_recognition.face_encodings(rgb, boxes)
                
                if len(encodings) > 0:
                    self.target_face_encoding = encodings[0]
                    self.target_known = True
                    print("HEDEF KİŞİ KAYDEDİLDİ!")
                else:
                    print("Yüz bulunamadı, kayıt yapılamadı.")

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = PrivacyCam()
    app.run()
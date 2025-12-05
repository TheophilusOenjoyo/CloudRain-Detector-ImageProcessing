# =================================================================================
#   IMPLEMENTASI ANALISIS MORFOLOGI DAN INTENSITAS WARNA CITRA AWAN UNTUK DETEKSI 
#           POTENSI HUJAN MENGGUNAKAN ALGORITMA PENGOLAHAN CITRA DIGITAL
# =================================================================================
# Dibuat oleh: Theophilus
# Deskripsi  : Aplikasi desktop untuk menganalisis gambar awan dan memprediksi
#              potensi hujan menggunakan teknik Image Processing sederhana.
# Library    : PyQt6 (GUI), OpenCV (Computer Vision), NumPy (Numerical)
# =================================================================================
# =================================================================================
#   PENJELASAN LOGIKA & PEMILIHAN AMBANG BATAS (THRESHOLD)
# =================================================================================
# 1. Analisis Intensitas (Warna/Kecerahan):
#    - Saya mengubah gambar menjadi Grayscale (skala abu-abu).
#    - Saya menghitung nilai rata-rata (Mean) dari piksel.
#    - Logika: Langit yang cerah biasanya memiliki intensitas cahaya yang tinggi 
#      (terang). Sebaliknya, awan hujan (mendung) memblokir sinar matahari, 
#      sehingga gambar cenderung lebih gelap. Jika nilai rata-rata di bawah ambang 
#      batas (140), Saya menganggap kondisi pencahayaan mendukung potensi 
#      mendung/gelap.
# 
# 2. Analisis Frekuensi (Tekstur/Bentuk):
#    - Saya menggunakan filter Laplacian. Dalam pengolahan citra, Laplacian 
#      adalah turunan kedua yang sangat sensitif terhadap perubahan intensitas 
#      yang cepat (tepi/garis).
#    - Saya menghitung Variance (variansi) dari hasil Laplacian tersebut.
#    - Nilai Variance Tinggi: Gambar memiliki banyak detail, tepi tajam, 
#      atau kontras tinggi (contoh: awan Cumulus putih di langit biru, 
#      atau pemandangan kota yang jelas).
#    - Nilai Variance Rendah: Gambar memiliki tekstur yang rata, blur, 
#      atau gradasi halus. Awan hujan tipe Nimbostratus biasanya berupa 
#      hamparan abu-abu yang rata tanpa detail tepi yang jelas (low frequency). 
#      Oleh karena itu, variance rendah (< 40) mengindikasikan tekstur 
#      awan hujan yang "rata".
# 
# 3. Keputusan Akhir (Binary):
#    Hanya jika gambar itu Gelap (Mean < 140) DAN bertekstur Rata (Variance < 40), 
# sistem akan memvonis "BERPOTENSI HUJAN". Jika salah satu syarat tidak terpenuhi, 
# maka dianggap tidak berpotensi.
# =================================================================================

import sys
import cv2
import numpy as np
import base64
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QFrame, QMessageBox)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt

class WeatherDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplikasi Deteksi Hujan (Laplacian & Intensity)")
        self.resize(1000, 700)
        
        # Variabel untuk menyimpan data saat ini agar bisa disimpan ke HTML
        self.current_image_path = None
        self.processed_data = None # Akan berisi dict hasil analisis
        
        self.init_ui()

    def init_ui(self):
        # Layout Utama Vertical
        main_layout = QVBoxLayout()
        
        # --- Bagian 1: Tombol Upload ---
        self.btn_upload = QPushButton("Upload Gambar")
        self.btn_upload.setMinimumHeight(40)
        self.btn_upload.clicked.connect(self.upload_image)
        main_layout.addWidget(self.btn_upload)

        # --- Bagian 2: Panel Gambar (Horizontal) ---
        image_layout = QHBoxLayout()
        
        # Helper function untuk membuat frame gambar dengan label
        def create_image_panel(title):
            v_layout = QVBoxLayout()
            lbl_title = QLabel(title)
            lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            
            lbl_img = QLabel("Belum ada gambar")
            lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_img.setFrameShape(QFrame.Shape.Box)
            lbl_img.setMinimumSize(300, 250)
            lbl_img.setScaledContents(True)
            
            v_layout.addWidget(lbl_title)
            v_layout.addWidget(lbl_img)
            return v_layout, lbl_img

        # Panel 1: Asli
        layout_orig, self.lbl_orig = create_image_panel("1. Gambar Asli")
        image_layout.addLayout(layout_orig)
        
        # Panel 2: Grayscale (Bukti Warna)
        layout_gray, self.lbl_gray = create_image_panel("2. Grayscale (Intensitas)")
        image_layout.addLayout(layout_gray)
        
        # Panel 3: Laplacian (Bukti Tekstur)
        layout_lap, self.lbl_lap = create_image_panel("3. Visualisasi Laplacian")
        image_layout.addLayout(layout_lap)
        
        main_layout.addLayout(image_layout)

        # --- Bagian 3: Panel Data ---
        data_layout = QHBoxLayout()
        
        self.lbl_intensity = QLabel("Intensitas (Mean): -")
        self.lbl_intensity.setFont(QFont("Arial", 12))
        
        self.lbl_texture = QLabel("Skor Tekstur (Variance): -")
        self.lbl_texture.setFont(QFont("Arial", 12))
        
        data_layout.addWidget(self.lbl_intensity)
        data_layout.addWidget(self.lbl_texture)
        main_layout.addLayout(data_layout)

        # --- Bagian 4: Status Utama ---
        self.lbl_status = QLabel("MENUNGGU INPUT")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.lbl_status.setStyleSheet("background-color: #DDDDDD; color: black; padding: 10px;")
        main_layout.addWidget(self.lbl_status)

        # --- Bagian 5: Tombol Simpan Laporan ---
        self.btn_save = QPushButton("Simpan Laporan (HTML)")
        self.btn_save.setMinimumHeight(40)
        self.btn_save.clicked.connect(self.save_report)
        self.btn_save.setEnabled(False) # Disable jika belum ada hasil
        main_layout.addWidget(self.btn_save)

        self.setLayout(main_layout)

    def upload_image(self):
        file_filter = "Image Files (*.png *.jpg *.jpeg *.bmp)"
        fname, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar Langit", "", file_filter)
        
        if fname:
            self.current_image_path = fname
            self.analyze_weather(fname)

    def analyze_weather(self, image_path):
        try:
            # Baca Gambar
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise ValueError("Gambar tidak dapat dibaca.")

            # 1. Parameter Intensitas (Convert ke Grayscale)
            gray_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            mean_val = np.mean(gray_image)
            
            # Condition A: Apakah Gelap?
            is_dark = mean_val < 140

            # 2. Parameter Frekuensi (Laplacian)
            # Menggunakan CV_64F agar kalkulasi negatif tetap presisi sebelum dihitung variansinya
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            variance_val = laplacian.var()
            
            # Visualisasi Laplacian (Convert kembali ke uint8 agar bisa ditampilkan)
            # convertScaleAbs mengambil nilai absolut dan konversi ke 8-bit
            laplacian_vis = cv2.convertScaleAbs(laplacian)

            # Condition B: Apakah Tekstur Rata?
            is_flat = variance_val < 40

            # 3. Keputusan Akhir (Binary Logic)
            result_text = ""
            bg_color = ""
            
            if is_dark and is_flat:
                result_text = "BERPOTENSI HUJAN"
                bg_color = "background-color: #FF0000; color: white;" # Merah
            else:
                result_text = "TIDAK BERPOTENSI HUJAN"
                bg_color = "background-color: #00FF00; color: black;" # Hijau

            # --- Update UI ---
            
            # Update Gambar
            self.display_image(img_bgr, self.lbl_orig)
            self.display_image(gray_image, self.lbl_gray, is_gray=True)
            self.display_image(laplacian_vis, self.lbl_lap, is_gray=True)

            # Update Data Text
            self.lbl_intensity.setText(f"Intensitas (Mean): {mean_val:.2f} (Apakah Gelap? {is_dark})")
            self.lbl_texture.setText(f"Skor Tekstur (Variance): {variance_val:.2f} (Apakah Rata? {is_flat})")

            # Update Status Label
            self.lbl_status.setText(result_text)
            self.lbl_status.setStyleSheet(f"{bg_color} padding: 10px; border-radius: 5px;")

            # Enable tombol save
            self.btn_save.setEnabled(True)

            # Simpan data untuk report
            self.processed_data = {
                'img_orig': img_bgr,
                'img_gray': gray_image,
                'img_lap': laplacian_vis,
                'mean': mean_val,
                'variance': variance_val,
                'is_dark': is_dark,
                'is_flat': is_flat,
                'result': result_text
            }

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def display_image(self, cv_img, label, is_gray=False):
        """Helper untuk konversi OpenCV ke QPixmap dan set ke Label"""
        h, w = cv_img.shape[:2]
        
        if is_gray:
            # Gray ke RGB agar QImage bisa baca
            converted_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            # BGR ke RGB
            converted_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
        bytes_per_line = 3 * w
        q_img = QImage(converted_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Resize agar pas di label namun aspect ratio terjaga
        label.setPixmap(pixmap.scaled(label.width(), label.height(), 
                                      Qt.AspectRatioMode.KeepAspectRatio))

    def img_to_base64(self, cv_img):
        """Helper konversi OpenCV image ke Base64 string untuk HTML"""
        _, buffer = cv2.imencode('.jpg', cv_img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return img_str

    def save_report(self):
        if not self.processed_data:
            return

        file_filter = "HTML Files (*.html)"
        fname, _ = QFileDialog.getSaveFileName(self, "Simpan Laporan", "laporan_cuaca.html", file_filter)

        if fname:
            try:
                # Siapkan Base64 images
                b64_orig = self.img_to_base64(self.processed_data['img_orig'])
                b64_gray = self.img_to_base64(self.processed_data['img_gray'])
                b64_lap = self.img_to_base64(self.processed_data['img_lap'])
                
                mean_val = self.processed_data['mean']
                var_val = self.processed_data['variance']
                res_text = self.processed_data['result']
                
                # Warna teks laporan
                res_color = "red" if "BERPOTENSI" in res_text and "TIDAK" not in res_text else "green"

                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Laporan Deteksi Hujan</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; padding: 20px; }}
                        h1 {{ text-align: center; }}
                        .result-box {{ 
                            font-size: 24px; font-weight: bold; 
                            text-align: center; padding: 15px; 
                            background-color: #f0f0f0; border: 2px solid #ccc;
                            margin: 20px 0; color: {res_color};
                        }}
                        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        .img-container {{ display: flex; justify-content: space-around; margin-top: 20px; }}
                        .img-box {{ text-align: center; width: 30%; }}
                        img {{ width: 100%; border: 1px solid #ddd; }}
                    </style>
                </head>
                <body>
                    <h1>Laporan Analisis Cuaca</h1>
                    
                    <div class="result-box">
                        KESIMPULAN: {res_text}
                    </div>

                    <h3>Data Statistik:</h3>
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Nilai Terukur</th>
                            <th>Ambang Batas (Threshold)</th>
                            <th>Status Parameter</th>
                        </tr>
                        <tr>
                            <td>Intensitas (Mean Brightness)</td>
                            <td>{mean_val:.2f}</td>
                            <td>< 140</td>
                            <td>{'GELAP (Pass)' if self.processed_data['is_dark'] else 'TERANG (Fail)'}</td>
                        </tr>
                        <tr>
                            <td>Frekuensi (Laplacian Variance)</td>
                            <td>{var_val:.2f}</td>
                            <td>< 40</td>
                            <td>{'RATA/SMOOTH (Pass)' if self.processed_data['is_flat'] else 'BERTEKSTUR/KASAR (Fail)'}</td>
                        </tr>
                    </table>

                    <h3>Visualisasi Analisis:</h3>
                    <div class="img-container">
                        <div class="img-box">
                            <h4>Gambar Asli</h4>
                            <img src="data:image/jpeg;base64,{b64_orig}" alt="Original">
                        </div>
                        <div class="img-box">
                            <h4>Grayscale (Warna)</h4>
                            <img src="data:image/jpeg;base64,{b64_gray}" alt="Gray">
                        </div>
                        <div class="img-box">
                            <h4>Laplacian (Tekstur)</h4>
                            <img src="data:image/jpeg;base64,{b64_lap}" alt="Laplacian">
                        </div>
                    </div>
                    
                    <p style="margin-top:30px; font-size:12px; color:#555;">
                        *Logika: Hujan dideteksi jika gambar gelap (Mean < 140) DAN tekstur langit rata/blur (Variance < 40).
                    </p>
                </body>
                </html>
                """
                
                with open(fname, 'w') as f:
                    f.write(html_content)
                    
                QMessageBox.information(self, "Sukses", "Laporan HTML berhasil disimpan!")
                
            except Exception as e:
                QMessageBox.critical(self, "Error Saving", str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WeatherDetectorApp()
    window.show()
    sys.exit(app.exec())
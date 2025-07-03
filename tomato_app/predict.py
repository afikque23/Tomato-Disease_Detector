import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageStat, ImageFilter
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Disease information with comprehensive details
disease_info = {
    'Tomato___Bacterial_spot': {
        'description': 'Bercak bakteri (Bacterial Spot) adalah penyakit serius yang disebabkan oleh bakteri Xanthomonas vesicatoria, X. euvesicatoria, X. gardneri, dan X. perforans. Penyakit ini dapat menyerang semua bagian tanaman tomat di atas tanah dan sangat merugikan produksi.',
        'symptoms': [
            'Bercak kecil berwarna coklat gelap dengan halo kuning pada daun muda',
            'Bercak berkembang menjadi lubang-lubang kecil pada daun tua',
            'Pada buah: bercak coklat kasar dengan permukaan yang menonjol',
            'Defoliasi atau gugur daun yang parah pada infeksi berat',
            'Batang dan tangkai buah menunjukkan garis-garis coklat memanjang'
        ],
        'causes': [
            'Kelembaban tinggi (>85%) dalam waktu lama',
            'Temperatur hangat (24-30°C)',
            'Percikan air hujan atau irigasi',
            'Luka pada tanaman akibat angin atau serangga',
            'Benih atau bibit yang sudah terinfeksi'
        ],
        'prevention': [
            'Gunakan benih bersertifikat dan bebas penyakit',
            'Rotasi tanaman dengan tanaman non-solanaceae selama 2-3 tahun',
            'Hindari penyiraman dari atas, gunakan drip irrigation',
            'Jaga jarak tanam yang cukup untuk sirkulasi udara',
            'Aplikasi mulch untuk mencegah percikan tanah',
            'Sanitasi kebun: buang sisa tanaman setelah panen',
            'Hindari bekerja di kebun saat tanaman basah'
        ],
        'treatment': [
            'Aplikasi bakterisida tembaga (copper hydroxide) setiap 7-10 hari',
            'Streptomycin sulfate untuk infeksi awal (ikuti aturan pakai)',
            'Buang dan musnahkan bagian tanaman yang terinfeksi',
            'Tingkatkan sirkulasi udara dengan pemangkasan',
            'Aplikasi pupuk kalium untuk meningkatkan ketahanan',
            'Gunakan mulch reflektif untuk mengurangi kelembaban'
        ],
        'impact': 'Dapat mengurangi hasil panen hingga 50% pada serangan berat',
        'severity': 'Tinggi',
        'prevention_schedule': {
            'Mingguan': 'Inspeksi rutin tanaman, sanitasi alat kerja',
            'Bi-mingguan': 'Aplikasi fungisida preventif saat cuaca lembab',
            'Bulanan': 'Evaluasi sistem irigasi dan drainase'
        }
    },
    'Tomato___Early_blight': {
        'description': 'Hawar awal (Early Blight) disebabkan oleh jamur Alternaria solani. Penyakit ini umumnya menyerang tanaman tomat yang sudah tua atau stres, dimulai dari daun bagian bawah dan menyebar ke atas.',
        'symptoms': [
            'Bercak coklat bulat dengan pola lingkaran konsentris (target spot)',
            'Dimulai dari daun bagian bawah dan menyebar ke atas',
            'Daun menguning dan rontok secara bertahap',
            'Pada batang: bercak coklat memanjang dengan pola konsentris',
            'Pada buah: bercak coklat gelap di dekat tangkai buah',
            'Reduksi vigor tanaman dan produksi buah'
        ],
        'causes': [
            'Kelembaban tinggi (>80%) dan temperatur hangat (26-28°C)',
            'Tanaman yang stres karena kekurangan nutrisi',
            'Sirkulasi udara yang buruk',
            'Tanaman terlalu rapat',
            'Irigasi berlebihan atau drainase buruk'
        ],
        'prevention': [
            'Rotasi tanaman dengan famili non-solanaceae',
            'Pastikan nutrisi tanaman seimbang, terutama nitrogen',
            'Mulching untuk mencegah percikan spora dari tanah',
            'Pruning untuk meningkatkan sirkulasi udara',
            'Hindari overhead irrigation, gunakan drip system',
            'Jaga kebersihan kebun dari sisa tanaman',
            'Tanam varietas yang tahan early blight'
        ],
        'treatment': [
            'Aplikasi fungisida chlorothalonil atau mancozeb',
            'Fungisida sistemik seperti azoxystrobin untuk pencegahan',
            'Buang daun terinfeksi dan musnahkan',
            'Tingkatkan pemupukan kalium dan fosfor',
            'Aplikasi fungisida organik seperti baking soda spray',
            'Pastikan drainase yang baik di sekitar tanaman'
        ],
        'impact': 'Dapat mengurangi hasil hingga 30-40% jika tidak ditangani',
        'severity': 'Sedang',
        'prevention_schedule': {
            'Mingguan': 'Inspeksi daun bagian bawah, buang daun terinfeksi',
            'Bi-mingguan': 'Aplikasi fungisida preventif saat cuaca lembab',
            'Bulanan': 'Evaluasi nutrisi tanaman dan sistem drainase'
        }
    },
    'Tomato___Late_blight': {
        'description': 'Hawar akhir (Late Blight) yang disebabkan oleh Phytophthora infestans adalah penyakit paling merusak pada tomat. Dapat menghancurkan seluruh tanaman dalam waktu singkat jika kondisi lingkungan mendukung.',
        'symptoms': [
            'Bercak coklat kehitaman yang cepat membesar pada daun',
            'Tepian putih berbulu pada bagian bawah daun (spora jamur)',
            'Bau busuk yang khas pada bagian tanaman yang terinfeksi',
            'Batang dan cabang menjadi coklat dan layu',
            'Buah menunjukkan bercak coklat keras dan keriput',
            'Tanaman dapat mati dalam 3-5 hari pada kondisi optimal untuk penyakit'
        ],
        'causes': [
            'Kelembaban sangat tinggi (>95%) dan suhu sejuk (15-20°C)',
            'Cuaca dingin dan lembab secara berkepanjangan',
            'Kondisi berawan dan berkabut',
            'Penyebaran spora melalui angin dan percikan air',
            'Tanaman yang lembab dalam waktu lama'
        ],
        'prevention': [
            'Gunakan varietas tahan late blight',
            'Hindari penanaman saat musim hujan atau cuaca lembab',
            'Pastikan drainase sangat baik',
            'Hindari irigasi malam hari',
            'Gunakan greenhouse atau polytunnel jika memungkinkan',
            'Monitor prakiraan cuaca dan siapkan tindakan preventif',
            'Jaga jarak tanam yang lebar untuk sirkulasi udara'
        ],
        'treatment': [
            'Aplikasi fungisida sistemik copper-based secara rutin',
            'Metalaxyl atau mefenoxam untuk pencegahan dan pengobatan',
            'Buang segera seluruh tanaman yang terinfeksi',
            'Bakar atau kubur jauh dari kebun bagian tanaman terinfeksi',
            'Aplikasi fungisida preventif sebelum kondisi cuaca mendukung',
            'Tingkatkan ventilasi dan kurangi kelembaban'
        ],
        'impact': 'Dapat menghancurkan 100% tanaman dalam kondisi optimal untuk penyakit',
        'severity': 'Sangat Tinggi',
        'prevention_schedule': {
            'Harian': 'Monitor cuaca dan kondisi tanaman saat musim berisiko',
            'Mingguan': 'Aplikasi fungisida preventif saat prakiraan cuaca lembab',
            'Bulanan': 'Evaluasi sistem drainase dan ventilasi'
        }
    },
    'Tomato___Leaf_Mold': {
        'description': 'Jamur daun (Leaf Mold) disebabkan oleh Passalora fulva (Fulvia fulva). Penyakit ini umum terjadi di greenhouse atau kondisi kelembaban tinggi dengan sirkulasi udara buruk.',
        'symptoms': [
            'Bercak kuning pada permukaan atas daun',
            'Pertumbuhan jamur berbulu hijau-coklat pada bagian bawah daun',
            'Bercak berkembang menjadi coklat dan nekrosis',
            'Daun mengering dan gugur dari bawah ke atas',
            'Jarang menyerang buah, tapi dapat terjadi pada kondisi lembab ekstrem'
        ],
        'causes': [
            'Kelembaban sangat tinggi (>85%) dengan sirkulasi udara buruk',
            'Temperatur sedang (20-25°C)',
            'Kondisi greenhouse yang lembab dan pengap',
            'Penyiraman berlebihan',
            'Tanaman terlalu rapat'
        ],
        'prevention': [
            'Pastikan ventilasi yang baik di greenhouse',
            'Gunakan kipas untuk meningkatkan sirkulasi udara',
            'Hindari penyiraman dari atas',
            'Kurangi kelembaban dengan heating jika perlu',
            'Jaga jarak tanam yang cukup',
            'Pruning untuk meningkatkan aliran udara',
            'Gunakan varietas tahan leaf mold'
        ],
        'treatment': [
            'Tingkatkan ventilasi dan sirkulasi udara',
            'Aplikasi fungisida chlorothalonil atau copper-based',
            'Kurangi frekuensi penyiraman',
            'Buang daun terinfeksi bagian bawah',
            'Aplikasi fungisida biologis Bacillus subtilis',
            'Pastikan drainase yang baik'
        ],
        'impact': 'Mengurangi hasil 20-30% terutama di greenhouse',
        'severity': 'Sedang',
        'prevention_schedule': {
            'Harian': 'Monitor kelembaban dan ventilasi greenhouse',
            'Mingguan': 'Pruning dan buang daun bagian bawah',
            'Bi-mingguan': 'Aplikasi fungisida saat kelembaban tinggi'
        }
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'Bercak daun Septoria disebabkan oleh jamur Septoria lycopersici. Penyakit ini berkembang pesat pada kondisi hangat dan lembab, menyerang daun dari bawah ke atas.',
        'symptoms': [
            'Bercak kecil bulat (1-3mm) dengan pusat abu-abu dan tepi coklat',
            'Titik hitam kecil (pycnidia) di tengah bercak',
            'Dimulai dari daun bagian bawah',
            'Daun menguning dan gugur secara bertahap',
            'Defoliasi parah dapat mengurangi kualitas buah'
        ],
        'causes': [
            'Kelembaban tinggi dan temperatur hangat (20-25°C)',
            'Percikan air dari tanah ke daun',
            'Sirkulasi udara yang buruk',
            'Sisa tanaman yang terinfeksi di tanah',
            'Peralatan yang terkontaminasi'
        ],
        'prevention': [
            'Rotasi tanaman dengan non-solanaceae selama 3 tahun',
            'Mulching untuk mencegah percikan tanah',
            'Hindari overhead irrigation',
            'Bersihkan sisa tanaman setelah panen',
            'Sterilisasi alat kerja',
            'Jaga jarak tanam yang tepat',
            'Pruning bagian bawah tanaman'
        ],
        'treatment': [
            'Aplikasi fungisida chlorothalonil atau mancozeb',
            'Buang dan musnahkan daun terinfeksi',
            'Tingkatkan sirkulasi udara',
            'Aplikasi fungisida copper-based',
            'Pastikan drainase yang baik',
            'Hindari bekerja saat tanaman basah'
        ],
        'impact': 'Mengurangi hasil 15-25% jika tidak dikontrol',
        'severity': 'Sedang',
        'prevention_schedule': {
            'Mingguan': 'Inspeksi dan buang daun terinfeksi',
            'Bi-mingguan': 'Aplikasi fungisida preventif',
            'Bulanan': 'Evaluasi mulching dan sistem irigasi'
        }
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': 'Tungau laba-laba (Spider Mites) Tetranychus urticae adalah hama kecil yang menghisap cairan sel tanaman. Berkembang pesat pada kondisi panas dan kering.',
        'symptoms': [
            'Stippling (bintik-bintik kuning kecil) pada permukaan daun',
            'Jaring laba-laba halus pada daun dan tunas',
            'Daun berubah kuning kemudian coklat dan gugur',
            'Penurunan vigor tanaman secara keseluruhan',
            'Pada serangan berat: seluruh tanaman tertutup jaring'
        ],
        'causes': [
            'Cuaca panas dan kering (>27°C, kelembaban <50%)',
            'Kekurangan air atau stress kekeringan',
            'Penggunaan insektisida broad-spectrum berlebihan',
            'Kurangnya predator alami',
            'Kondisi berdebu'
        ],
        'prevention': [
            'Jaga kelembaban tanah dan udara yang cukup',
            'Penyiraman teratur, hindari stress air',
            'Lindungi dan perbanyak predator alami',
            'Hindari penggunaan insektisida broad-spectrum',
            'Semprot air untuk membersihkan debu',
            'Gunakan tanaman companion yang mengusir tungau',
            'Monitor rutin terutama saat cuaca panas'
        ],
        'treatment': [
            'Semprotan air kuat untuk mengurangi populasi',
            'Aplikasi mitisida spesifik (abamectin, spiromesifen)',
            'Lepas predator alami (Phytoseiulus persimilis)',
            'Aplikasi minyak hortikultura atau neem oil',
            'Tingkatkan kelembaban sekitar tanaman',
            'Rotasi mitisida untuk mencegah resistensi'
        ],
        'impact': 'Mengurangi hasil 20-40% pada serangan berat',
        'severity': 'Sedang',
        'prevention_schedule': {
            'Harian': 'Monitor kondisi kelembaban saat cuaca panas',
            'Mingguan': 'Inspeksi bagian bawah daun, semprotan air',
            'Bi-mingguan': 'Aplikasi mitisida jika diperlukan'
        }
    },
    'Tomato___Target_Spot': {
        'description': 'Target spot disebabkan oleh jamur Corynespora cassiicola. Dinamakan demikian karena bercaknya menyerupai target dengan lingkaran konsentris.',
        'symptoms': [
            'Bercak coklat dengan pola lingkaran konsentris seperti target',
            'Dimulai sebagai bercak kecil kemudian membesar hingga 1cm',
            'Halo kuning di sekitar bercak pada daun muda',
            'Dapat menyerang daun, batang, dan buah',
            'Defoliasi dimulai dari daun bagian bawah'
        ],
        'causes': [
            'Kelembaban tinggi dan temperatur hangat (24-32°C)',
            'Percikan air dari tanah atau irigasi overhead',
            'Sirkulasi udara yang buruk',
            'Tanaman yang stress atau lemah',
            'Sisa tanaman yang terinfeksi'
        ],
        'prevention': [
            'Rotasi tanaman dengan non-host selama 2-3 tahun',
            'Mulching untuk mencegah percikan tanah',
            'Drip irrigation atau irigasi bawah permukaan',
            'Pruning untuk meningkatkan sirkulasi udara',
            'Nutrisi seimbang untuk menjaga vigor tanaman',
            'Bersihkan kebun dari sisa tanaman'
        ],
        'treatment': [
            'Aplikasi fungisida azoxystrobin atau pyraclostrobin',
            'Buang daun terinfeksi dan musnahkan',
            'Tingkatkan drainase dan sirkulasi udara',
            'Aplikasi fungisida copper-based',
            'Kurangi kelembaban di sekitar tanaman',
            'Pastikan nutrisi tanaman optimal'
        ],
        'impact': 'Mengurangi hasil 15-30% tergantung keparahan',
        'severity': 'Sedang',
        'prevention_schedule': {
            'Mingguan': 'Inspeksi tanaman dan buang daun terinfeksi',
            'Bi-mingguan': 'Aplikasi fungisida preventif saat lembab',
            'Bulanan': 'Evaluasi sistem irigasi dan drainase'
        }
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'Virus keriting kuning daun tomat (TYLCV) ditularkan oleh kutu kebul (Bemisia tabaci). Virus ini sangat merusak dan sulit dikontrol setelah tanaman terinfeksi.',
        'symptoms': [
            'Daun menggulung ke atas dan menebal',
            'Warna daun berubah kuning dengan urat hijau',
            'Pertumbuhan tanaman terhambat (stunting)',
            'Bunga rontok, produksi buah sangat menurun',
            'Internoda memendek, tanaman tampak kerdil'
        ],
        'causes': [
            'Penularan melalui kutu kebul (Bemisia tabaci)',
            'Bibit atau transplant yang sudah terinfeksi',
            'Tanaman gulma yang menjadi reservoir virus',
            'Migrasi kutu kebul dari area terinfeksi',
            'Kondisi cuaca hangat yang mendukung kutu kebul'
        ],
        'prevention': [
            'Gunakan varietas tahan TYLCV',
            'Kontrol populasi kutu kebul dengan insektisida sistemik',
            'Gunakan jaring serangga pada nursery dan greenhouse',
            'Mulsa reflektif silver untuk mengusir kutu kebul',
            'Bersihkan gulma di sekitar pertanaman',
            'Isolasi tanaman baru sebelum tanam di lapang',
            'Monitor rutin kehadiran kutu kebul'
        ],
        'treatment': [
            'Tidak ada pengobatan langsung untuk virus',
            'Kontrol intensif kutu kebul vektor',
            'Buang dan musnahkan tanaman terinfeksi',
            'Aplikasi insektisida sistemik (imidacloprid)',
            'Gunakan sticky trap kuning untuk monitoring',
            'Semprot insektisida setiap 7-10 hari',
            'Tanam tanaman perangkap untuk kutu kebul'
        ],
        'impact': 'Dapat menyebabkan kehilangan hasil 100% pada serangan berat',
        'severity': 'Tinggi',
        'prevention_schedule': {
            'Harian': 'Monitor kehadiran kutu kebul',
            'Mingguan': 'Aplikasi insektisida dan inspeksi tanaman',
            'Bi-mingguan': 'Bersihkan gulma dan ganti sticky trap'
        }
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'Virus mosaik tomat (ToMV) adalah virus yang sangat stabil dan mudah menular melalui kontak mekanis. Dapat bertahan lama pada alat, pakaian, dan sisa tanaman.',
        'symptoms': [
            'Pola mosaik kuning-hijau pada daun',
            'Daun berkerut dan bentuk tidak normal',
            'Pertumbuhan tidak merata dan stunting ringan',
            'Buah berbintik dan kualitas menurun',
            'Malformasi daun dan tunas muda'
        ],
        'causes': [
            'Penularan melalui kontak mekanis (tangan, alat)',
            'Benih yang terinfeksi virus',
            'Sisa tanaman terinfeksi di tanah',
            'Pekerja yang merokok (virus tembakau serupa)',
            'Transplantasi atau grafting yang terkontaminasi'
        ],
        'prevention': [
            'Gunakan benih bersertifikat bebas virus',
            'Sterilisasi alat kerja dengan alkohol 70%',
            'Cuci tangan sebelum menangani tanaman',
            'Hindari merokok di area pertanaman',
            'Bersihkan sisa tanaman setelah panen',
            'Gunakan varietas tahan virus jika tersedia',
            'Isolasi tanaman baru selama periode karantina'
        ],
        'treatment': [
            'Tidak ada pengobatan untuk virus',
            'Buang dan musnahkan tanaman terinfeksi',
            'Sterilisasi semua alat dan peralatan',
            'Desinfeksi area tanam',
            'Ganti tanah atau sterilisasi tanah',
            'Hindari penanaman solanaceae di area sama',
            'Monitoring ketat tanaman baru'
        ],
        'impact': 'Mengurangi hasil 20-50% tergantung waktu infeksi',
        'severity': 'Tinggi',
        'prevention_schedule': {
            'Harian': 'Sterilisasi alat kerja dan cuci tangan',
            'Mingguan': 'Inspeksi gejala virus pada tanaman',
            'Bulanan': 'Evaluasi protokol sanitasi'
        }
    },
    'Tomato___healthy': {
        'description': 'Tanaman tomat sehat menunjukkan pertumbuhan vigor dengan daun hijau segar, batang kuat, dan produksi buah optimal. Kondisi ini dicapai melalui manajemen budidaya yang tepat.',
        'symptoms': [
            'Daun hijau tua segar tanpa bercak atau perubahan warna',
            'Pertumbuhan seragam dan vigor yang baik',
            'Batang kokoh dengan internoda normal',
            'Pembungaan dan pembuahan normal',
            'Tidak ada tanda-tanda stress atau penyakit'
        ],
        'maintenance': [
            'Pemupukan berimbang sesuai fase pertumbuhan',
            'Penyiraman teratur sesuai kebutuhan tanaman',
            'Pruning dan pemeliharaan rutin',
            'Monitoring hama dan penyakit secara berkala',
            'Penyiangan gulma secara teratur'
        ],
        'prevention': [
            'Rotasi tanaman untuk mencegah penumpukan patogen',
            'Sanitasi kebun dan alat kerja',
            'Pemilihan varietas yang sesuai dengan kondisi lokal',
            'Sistem irigasi yang efisien',
            'Pemupukan organik untuk meningkatkan kesehatan tanah',
            'Monitoring cuaca dan penyesuaian praktik budidaya',
            'Integrated Pest Management (IPM)'
        ],
        'optimal_conditions': [
            'Suhu optimal: 18-24°C (malam) dan 20-26°C (siang)',
            'Kelembaban relatif: 60-70%',
            'pH tanah: 6.0-6.8',
            'Drainase baik dengan kelembaban tanah konsisten',
            'Sinar matahari penuh (6-8 jam per hari)',
            'Sirkulasi udara yang baik'
        ],
        'impact': 'Produktivitas optimal dengan kualitas buah terbaik',
        'severity': 'Tidak ada',
        'maintenance_schedule': {
            'Harian': 'Monitoring visual kondisi tanaman',
            'Mingguan': 'Penyiraman, pemupukan, dan pruning sesuai kebutuhan',
            'Bulanan': 'Evaluasi nutrisi tanah dan sistem budidaya'
        }
    }
}

def load_model(path='best_model.pth'):
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded model from {path}")
    except FileNotFoundError:
        print(f"Model file {path} not found. Using randomly initialized model for demo purposes.")
        print("Note: Predictions will be random until you train and save a proper model.")
    
    model.eval()
    model.to(device)
    return model

def detect_non_tomato_features(image_path):
    """
    Deteksi sederhana untuk gambar non-tomat
    """
    try:
        image = Image.open(image_path).convert("RGB")
        stat = ImageStat.Stat(image)
        mean_colors = stat.mean
        
        # Analisis warna dasar
        red, green, blue = mean_colors
        
        # Scoring sederhana
        non_tomato_score = 0
        reasons = []
        
        # Check jika bukan hijau dominan
        if not (green > red and green > blue):
            non_tomato_score += 30
            reasons.append("Warna dominan bukan hijau")
            
        # Check jika terlalu gelap atau terang
        brightness = sum(mean_colors) / 3
        if brightness < 50 or brightness > 200:
            non_tomato_score += 20
            reasons.append("Pencahayaan tidak optimal")
        
        return non_tomato_score, reasons
        
    except Exception as e:
        print(f"Error in detection: {e}")
        return 0, []

def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    # Pre-analysis for non-tomato detection
    non_tomato_score, non_tomato_reasons = detect_non_tomato_features(image_path)
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)

    predicted_class = class_names[preds.item()]
    confidence_score = float(confidence.item()) * 100
    
    # Get all probabilities for top predictions
    probs = probabilities[0].cpu().numpy()
    top_3_indices = probs.argsort()[-3:][::-1]
    top_3_predictions = [(class_names[i], float(probs[i]) * 100) for i in top_3_indices]
    
    # Enhanced validation system
    is_likely_tomato = True
    warning_message = None
    validation_reasons = []
    
    # Check 1: Pre-analysis score
    if non_tomato_score >= 40:
        is_likely_tomato = False
        validation_reasons.extend(non_tomato_reasons[:2])
    elif non_tomato_score >= 25:
        validation_reasons.extend(non_tomato_reasons[:1])
    
    # Check 2: Model confidence
    if confidence_score < 20:
        is_likely_tomato = False
        validation_reasons.append("Tingkat kepercayaan model sangat rendah")
    elif confidence_score < 40:
        validation_reasons.append("Tingkat kepercayaan model kurang optimal")
    
    # Check 3: Distribution of probabilities
    top_3_scores = [prob for _, prob in top_3_predictions]
    score_variance = max(top_3_scores) - min(top_3_scores)
    
    if max(top_3_scores) < 30:
        is_likely_tomato = False
        validation_reasons.append("Semua prediksi memiliki probabilitas rendah")
    elif score_variance < 10:
        is_likely_tomato = False
        validation_reasons.append("Model tidak dapat membedakan kelas dengan jelas")
    
    # Check 4: Entropy analysis (uncertainty)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    if entropy > 2.2:
        is_likely_tomato = False
        validation_reasons.append("Model menunjukkan ketidakpastian sangat tinggi")
    elif entropy > 1.9:
        validation_reasons.append("Model menunjukkan ketidakpastian")
    
    # Generate warning message
    if not is_likely_tomato:
        warning_message = "PERINGATAN: Gambar kemungkinan bukan daun tomat. " + "; ".join(validation_reasons[:3])
    elif validation_reasons:
        warning_message = "Perhatian: " + "; ".join(validation_reasons[:2])
    
    # Get disease information
    disease_data = disease_info.get(predicted_class, {})
    
    result = {
        'prediction': predicted_class,
        'confidence': confidence_score,
        'top_3': top_3_predictions,
        'disease_info': disease_data,
        'is_likely_tomato': is_likely_tomato,
        'warning_message': warning_message,
        'debug_info': {
            'non_tomato_score': non_tomato_score,
            'entropy': float(entropy),
            'validation_reasons': validation_reasons,
            'non_tomato_reasons': non_tomato_reasons
        }
    }
    
    return result

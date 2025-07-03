from flask import Flask, render_template, request, send_file, jsonify
from predict import load_model, predict_image
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            return jsonify({'error': 'No file selected'})
        
        image = request.files["image"]
        if image.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if image and allowed_file(image.filename):
            # Generate unique filename
            filename = str(uuid.uuid4()) + '_' + image.filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            
            # Get prediction results
            result = predict_image(image_path, model)
            
            # Add image path to result
            result['image_path'] = image_path
            result['filename'] = filename
            result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return jsonify(result)
    
    return render_template("index.html")

@app.route("/download_report/<filename>")
def download_report(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(image_path):
        return "File not found", 404
    
    # Get prediction results again for PDF
    result = predict_image(image_path, model)
    
    # Generate PDF
    pdf_filename = f"tomato_disease_report_{filename.split('.')[0]}.pdf"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    
    generate_pdf_report(result, image_path, pdf_path)
    
    return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_pdf_report(result, image_path, pdf_path):
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.darkgreen,
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.darkblue,
        spaceAfter=12
    )
    
    subheader_style = ParagraphStyle(
        'CustomSubHeader',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.darkred,
        spaceAfter=8
    )
    
    # Title
    story.append(Paragraph("Laporan Deteksi Penyakit Daun Tomat", title_style))
    story.append(Paragraph("Sistem AI untuk Analisis Kesehatan Tanaman", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Timestamp
    story.append(Paragraph(f"Tanggal Analisis: {datetime.now().strftime('%d %B %Y, %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Image
    try:
        img = RLImage(image_path, width=4*inch, height=3*inch)
        story.append(img)
        story.append(Spacer(1, 20))
    except:
        story.append(Paragraph("Gambar tidak dapat dimuat", styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Prediction Results
    story.append(Paragraph("Hasil Deteksi", header_style))
    
    # Check if this is likely a tomato leaf
    is_likely_tomato = result.get('is_likely_tomato', True)
    warning_message = result.get('warning_message', '')
    
    if not is_likely_tomato:
        # Add warning for non-tomato images
        warning_text = f"<b>⚠️ PERINGATAN:</b> {warning_message}<br/><br/>"
        warning_text += "<b>Catatan:</b> Sistem ini dirancang khusus untuk mendeteksi penyakit pada daun tomat. "
        warning_text += "Hasil prediksi mungkin tidak akurat jika gambar bukan daun tomat.<br/><br/>"
        story.append(Paragraph(warning_text, ParagraphStyle('Warning', parent=styles['Normal'], 
                                                           textColor=colors.red, fontSize=12)))
        story.append(Spacer(1, 10))
    
    prediction_text = f"<b>Diagnosis:</b> {result['prediction'].replace('Tomato___', '').replace('_', ' ')}<br/>"
    prediction_text += f"<b>Tingkat Kepercayaan:</b> {result['confidence']:.2f}%<br/>"
    
    if not is_likely_tomato:
        prediction_text += f"<b>Status:</b> TIDAK VALID - Kemungkinan bukan daun tomat"
    elif result['confidence'] < 50:
        prediction_text += f"<b>Status:</b> KEPERCAYAAN RENDAH - Perlu verifikasi"
    else:
        prediction_text += f"<b>Status:</b> VALID"
        
    story.append(Paragraph(prediction_text, styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Top 3 Predictions
    story.append(Paragraph("Top 3 Prediksi:", subheader_style))
    top3_data = [['Peringkat', 'Penyakit', 'Probabilitas']]
    for i, (disease, prob) in enumerate(result['top_3'], 1):
        disease_name = disease.replace('Tomato___', '').replace('_', ' ')
        top3_data.append([str(i), disease_name, f"{prob:.2f}%"])
    
    top3_table = Table(top3_data)
    top3_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(top3_table)
    story.append(Spacer(1, 20))
    
    # Disease Information
    disease_info = result.get('disease_info', {})
    if disease_info:
        story.append(Paragraph("Informasi Lengkap Penyakit", header_style))
        
        # Description
        if disease_info.get('description'):
            story.append(Paragraph("Deskripsi:", subheader_style))
            story.append(Paragraph(disease_info['description'], styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Symptoms
        if disease_info.get('symptoms'):
            story.append(Paragraph("Gejala dan Tanda-tanda:", subheader_style))
            if isinstance(disease_info['symptoms'], list):
                for symptom in disease_info['symptoms']:
                    story.append(Paragraph(f"• {symptom}", styles['Normal']))
            else:
                story.append(Paragraph(disease_info['symptoms'], styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Causes
        if disease_info.get('causes'):
            story.append(Paragraph("Penyebab dan Kondisi Pemicu:", subheader_style))
            if isinstance(disease_info['causes'], list):
                for cause in disease_info['causes']:
                    story.append(Paragraph(f"• {cause}", styles['Normal']))
            else:
                story.append(Paragraph(disease_info['causes'], styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Prevention
        if disease_info.get('prevention'):
            story.append(Paragraph("Pencegahan:", subheader_style))
            if isinstance(disease_info['prevention'], list):
                for prevent in disease_info['prevention']:
                    story.append(Paragraph(f"• {prevent}", styles['Normal']))
            else:
                story.append(Paragraph(disease_info['prevention'], styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Treatment
        if disease_info.get('treatment'):
            story.append(Paragraph("Pengobatan dan Pengendalian:", subheader_style))
            if isinstance(disease_info['treatment'], list):
                for treat in disease_info['treatment']:
                    story.append(Paragraph(f"• {treat}", styles['Normal']))
            else:
                story.append(Paragraph(disease_info['treatment'], styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Impact
        if disease_info.get('impact'):
            story.append(Paragraph("Dampak dan Kerugian:", subheader_style))
            story.append(Paragraph(disease_info['impact'], styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Severity
        if disease_info.get('severity'):
            story.append(Paragraph("Tingkat Keparahan:", subheader_style))
            severity_text = f"<b>{disease_info['severity']}</b>"
            story.append(Paragraph(severity_text, styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Prevention Schedule
        schedule = disease_info.get('prevention_schedule') or disease_info.get('maintenance_schedule')
        if schedule:
            story.append(Paragraph("Jadwal Pengendalian:", subheader_style))
            for period, activity in schedule.items():
                story.append(Paragraph(f"<b>{period}:</b> {activity}", styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Special handling for healthy plants
        if 'healthy' in result['prediction'].lower():
            if disease_info.get('maintenance'):
                story.append(Paragraph("Perawatan Optimal:", subheader_style))
                if isinstance(disease_info['maintenance'], list):
                    for maintain in disease_info['maintenance']:
                        story.append(Paragraph(f"• {maintain}", styles['Normal']))
                story.append(Spacer(1, 10))
            
            if disease_info.get('optimal_conditions'):
                story.append(Paragraph("Kondisi Optimal:", subheader_style))
                if isinstance(disease_info['optimal_conditions'], list):
                    for condition in disease_info['optimal_conditions']:
                        story.append(Paragraph(f"• {condition}", styles['Normal']))
                story.append(Spacer(1, 10))
    
    # Recommendations based on severity
    story.append(Paragraph("Rekomendasi Tindakan:", header_style))
    severity = disease_info.get('severity', '')
    
    if not is_likely_tomato:
        story.append(Paragraph("<b>GAMBAR TIDAK VALID - BUKAN DAUN TOMAT</b>", 
                              ParagraphStyle('Invalid', parent=styles['Normal'], 
                                           textColor=colors.red, fontSize=14)))
        recommendations = [
            "Upload ulang dengan gambar daun tomat yang jelas",
            "Pastikan pencahayaan yang cukup saat mengambil foto",
            "Fokuskan kamera pada satu daun tomat",
            "Hindari background yang mengganggu",
            "Gunakan gambar dengan resolusi yang baik"
        ]
    elif severity == 'Sangat Tinggi':
        story.append(Paragraph("<b>TINDAKAN DARURAT DIPERLUKAN!</b>", 
                              ParagraphStyle('Emergency', parent=styles['Normal'], 
                                           textColor=colors.red, fontSize=12)))
        recommendations = [
            "Segera isolasi tanaman yang terinfeksi",
            "Lakukan pengobatan intensif dalam 24 jam",
            "Monitor penyebaran ke tanaman lain setiap hari",
            "Konsultasi dengan ahli pertanian segera"
        ]
    elif severity == 'Tinggi':
        story.append(Paragraph("<b>PERLU TINDAKAN CEPAT</b>", 
                              ParagraphStyle('High', parent=styles['Normal'], 
                                           textColor=colors.orange, fontSize=12)))
        recommendations = [
            "Lakukan pengobatan sesuai rekomendasi dalam 2-3 hari",
            "Tingkatkan monitoring tanaman",
            "Cegah penyebaran dengan isolasi",
            "Dokumentasi perkembangan penyakit"
        ]
    elif severity == 'Sedang':
        story.append(Paragraph("<b>MONITOR & TREATMENT</b>", 
                              ParagraphStyle('Medium', parent=styles['Normal'], 
                                           textColor=colors.yellow, fontSize=12)))
        recommendations = [
            "Lakukan pengobatan pencegahan",
            "Tingkatkan perawatan rutin",
            "Monitor setiap 3-5 hari",
            "Fokus pada tindakan pencegahan"
        ]
    else:
        story.append(Paragraph("<b>PERTAHANKAN KONDISI OPTIMAL</b>", 
                              ParagraphStyle('Healthy', parent=styles['Normal'], 
                                           textColor=colors.green, fontSize=12)))
        recommendations = [
            "Lanjutkan perawatan rutin",
            "Monitor mingguan sudah cukup",
            "Optimalisasi pertumbuhan tanaman",
            "Jaga kondisi lingkungan optimal"
        ]
    
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles['Normal']))
    
    story.append(Spacer(1, 20))
    
    # Footer
    story.append(Paragraph("---", styles['Normal']))
    story.append(Paragraph("Laporan ini dibuat secara otomatis oleh sistem AI untuk deteksi penyakit daun tomat.", 
                          styles['Italic']))
    story.append(Paragraph("Untuk diagnosis yang akurat dan pengobatan yang tepat, konsultasikan dengan ahli pertanian atau penyuluh pertanian setempat.", 
                          styles['Italic']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Sistem ini dikembangkan untuk membantu petani dalam deteksi dini penyakit tanaman.", 
                          styles['Italic']))
    
    doc.build(story)

if __name__ == "__main__":
    app.run(debug=True)

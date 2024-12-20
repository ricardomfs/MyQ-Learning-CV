import cv2
import pytesseract

# Configurar o caminho para o executável do Tesseract (se necessário)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path, roi_coords):
    # Carregar a imagem
    image = cv2.imread(image_path) 
    print(image.shape)
    # Coordenadas da ROI (xmin, ymin, xmax, ymax)
    xmin, ymin, xmax, ymax = roi_coords

    # Extrair a ROI
    roi = image[ymin:ymax, xmin:xmax]

    # Converter a ROI para escala de cinza
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Aplicar OCR
    text = pytesseract.image_to_string(gray_roi)

    return text

# Exemplo de uso
image_path = 'image.png'  # Caminho para a imagem
# roi_coords = (600, 925, 870, 1000)  # Coordenadas da ROI
roi_coords = (520, 820, 720, 900)  # Coordenadas da ROI
extracted_text = extract_text_from_image(image_path, roi_coords)
print("Texto extraído:", extracted_text)

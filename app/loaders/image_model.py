import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


def image_model(file_path):

    try:
        image = Image.open(file_path)

        # Preprocessing (IMPORTANT)
        image = image.convert("L")  # grayscale
        image = image.filter(ImageFilter.SHARPEN)

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)

        # OCR
        text = pytesseract.image_to_string(image)

        # Clean output
        text = text.strip()

        return text if text else "No readable text found in image."

    except Exception as e:
        return f"OCR Error: {str(e)}"
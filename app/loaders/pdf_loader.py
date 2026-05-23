from pypdf import PdfReader

def pdf_loader(file_path):

    reader = PdfReader(file_path)

    pages_text = []

    for i, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:
            pages_text.append(f"Page {i+1}\n{text.strip()}")

    return "\n\n".join(pages_text)
import sys
import os

pdf_path = r"C:\Users\user\Downloads\CCCQ.pdf"
output_path = r"d:\Download\trae-research-code\open-ts-search\cccq_content.txt"

def read_pypdf():
    try:
        from pypdf import PdfReader
        # print("Using pypdf")
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except ImportError:
        return None
    except Exception as e:
        return f"Error using pypdf: {e}"

content = read_pypdf()

if content is None:
    content = "No suitable PDF library found or error occurred."

try:
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Successfully wrote content to {output_path}")
except Exception as e:
    print(f"Failed to write output file: {e}")

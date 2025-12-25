import sys
import os

pdf_path = "/Users/coco/docker/ubuntu/home/jq/deep_potential_arbitrage/reference/Kim 等 - 2025 - Deep Mean-Reversion A Physics-Informed Contrastive Approach to Pairs Trading.pdf"

def try_pypdf():
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        print("Success with pypdf")
        print(text)
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"pypdf failed: {e}")
        return False

def try_pypdf2():
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        print("Success with PyPDF2")
        print(text)
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"PyPDF2 failed: {e}")
        return False

def try_pdfminer():
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(pdf_path)
        print("Success with pdfminer")
        print(text)
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"pdfminer failed: {e}")
        return False

if not try_pypdf():
    if not try_pypdf2():
        if not try_pdfminer():
             print("No suitable PDF library found (pypdf, PyPDF2, pdfminer).")
        else:
            with open("paper.txt", "w", encoding='utf-8') as f:
                 from pdfminer.high_level import extract_text
                 f.write(extract_text(pdf_path))
            print("Text saved to paper.txt")
    else:
        # PyPDF2 success
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        with open("paper.txt", "w", encoding='utf-8') as f:
            f.write(text)
        print("Text saved to paper.txt")
else:
    # pypdf success
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    with open("paper.txt", "w", encoding='utf-8') as f:
        f.write(text)
    print("Text saved to paper.txt")

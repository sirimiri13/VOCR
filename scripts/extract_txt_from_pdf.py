import PyPDF2

def pdf_to_txt(pdf_path, txt_path=None):
   
    if txt_path is None:
        txt_path = pdf_path.replace('.pdf', '.txt')
    
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            print(f"Đang xử lý {num_pages} trang...")
            
            text_content = ""
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text()
                text_content += f"\n\n--- Trang {page_num + 1} ---\n\n"
            
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text_content)
            
            print(f"✅ Chuyển đổi thành công! File lưu tại: {txt_path}")
            return text_content
            
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file {pdf_path}")
        return None
    except Exception as e:
        print(f"❌ Lỗi khi xử lý PDF: {str(e)}")
        return None
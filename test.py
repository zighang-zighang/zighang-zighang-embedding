from pypdf import PdfReader

from recruitment_db import RecruitmentsVector

if __name__ == "__main__":
    recruitment_vector = RecruitmentsVector()

    reader = PdfReader('자기소개서/최호_포폴.pdf')

    all_text = ""

    for page in reader.pages:
        all_text += page.extract_text()

    recruitment_vector.search(all_text, top_k=10)

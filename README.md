import os
import tkinter as tk
from tkinter import filedialog
from PyPDF2 import PdfReader
import docx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight, efficient model

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file format")

def select_file(title="Select a file"):
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title)

def get_match_score(resume_text, job_text):
    resume_embedding = model.encode([resume_text])[0]
    job_embedding = model.encode([job_text])[0]
    
    similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
    score = round(similarity * 100, 2)

    return f"Match Score: {score}/100\n\nThis score represents the semantic similarity between the resume and the job description."

def main():
    print("📄 Select a RESUME file:")
    resume_file = select_file("Select Resume")
    resume_text = extract_text(resume_file)

    print("📄 Select a JOB DESCRIPTION file:")
    jd_file = select_file("Select Job Description")
    job_text = extract_text(jd_file)

    print("🔍 Analyzing...")
    result = get_match_score(resume_text, job_text)
    print("\n📝 RESULT:\n", result)

if __name__ == "__main__":
    main()

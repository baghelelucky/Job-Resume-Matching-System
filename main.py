from flask import Flask, request, render_template
import os
import PyPDF2
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Helper functions
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:  # Handle cases where extract_text() returns None
                text += extracted_text
    return text.strip()

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path).strip()

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def extract_text(file_path):
    """Extract text based on file type"""
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.endswith(".txt"):
        return extract_text_from_txt(file_path)
    return ""

# Flask App
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def matchResume():
    return render_template('matchResume.html')

@app.route("/matcher", methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form.get('job_description')
        resume_files = request.files.getlist('resumes')

        if not resume_files or not job_description.strip():
            return render_template('matchResume.html', message="Please upload resumes and enter a job description.")

        resumes = []
        file_names = []

        for resume_file in resume_files:
            if resume_file.filename == "":
                continue  # Skip empty files

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(file_path)  # Save uploaded file

            extracted_text = extract_text(file_path)
            if extracted_text:
                file_names.append(resume_file.filename)
                resumes.append(extracted_text)
        
        if not resumes:  # If no valid text was extracted
            return render_template('matchResume.html', message="No valid text extracted from uploaded files.")

        # Compute similarity scores
        vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
        vectors = vectorizer.toarray()
        job_vector = vectors[0]  # Job description vector
        resume_vectors = vectors[1:]  # Resume vectors

        similarities = cosine_similarity([job_vector], resume_vectors)[0]  # Compute cosine similarity

        # Get top 3 matching resumes
        top_indices = similarities.argsort()[-3:][::-1]  # Get top 3 highest similarity scores
        top_resumes = [file_names[i] for i in top_indices]
        similarity_scores = [round(similarities[i], 2) for i in top_indices]

        print("\n--- Debugging Info ---")
        print("Job Description:\n", job_description)
        print("Resume Files:", file_names)
        print("Similarity Scores:", similarity_scores)

        return render_template(
            'matchResume.html',
            message="Top matching resumes:",
            Top_Resumes=list(zip(top_resumes, similarity_scores))  # Pass as a list of tuples
        )

    return render_template('matchResume.html')

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def calculate_similarity(resume_text, job_desc):
    documents = [resume_text, job_desc]
    tfidf = TfidfVectorizer().fit_transform(documents)
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(similarity * 100, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_desc = request.form["job_desc"]
        resume_file = request.files["resume"]
        resume_text = extract_text_from_pdf(resume_file)
        score = calculate_similarity(resume_text, job_desc)
        return render_template("index.html", score=score, job_desc=job_desc)
    return render_template("index.html", score=None)

if __name__ == "__main__":
    app.run(debug=True)

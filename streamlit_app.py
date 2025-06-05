import streamlit as st
import streamlit.components.v1 as components
import cohere
import PyPDF2
import base64
import os
import time

# === Secure API key from secrets ===
co = cohere.Client(st.secrets["cohere"]["api_key"])

# === Extract text from PDF ===
def extract_text_from_pdf(file_path):
    pdf_reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# === Chunk text for long inputs ===
def chunk_text(text, max_chunk_size=2500):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk_size
        period_pos = text.rfind('.', start, end)
        if period_pos != -1:
            end = period_pos + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks

# === Cohere-based summary ===
def cohere_chat_summary(text):
    try:
        response = co.chat(
            model="command-xlarge-nightly",
            message=f"Summarize this text clearly and concisely:\n\n{text}",
            temperature=0.4,
            max_tokens=300
        )
        return response.text.strip()
    except Exception as e:
        return f"❌ API Error: {str(e)}"

def summarize_text(text):
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        summary = cohere_chat_summary(chunk)
        summaries.append(summary)
        time.sleep(6)
    combined = " ".join(summaries)
    if len(combined) > 2000:
        return cohere_chat_summary(combined[:2000])
    return combined

# === Generate Q&A ===
def generate_auto_qa(text, num_questions=5):
    prompt = (
        f"Generate {num_questions} questions and answers from the following text:\n\n{text}\n\nFormat: Q1: ... A1: ... Q2: ... A2: ..."
    )
    try:
        response = co.chat(
            model="command-xlarge-nightly",
            message=prompt,
            temperature=0.5,
            max_tokens=600
        )
        return response.text.strip()
    except Exception as e:
        return f"❌ API Error: {str(e)}"

# === Custom answer from user question ===
def generate_answer(text, question):
    prompt = f"Answer the question based on this text:\n\n{text}\n\nQuestion: {question}"
    try:
        response = co.chat(
            model="command-xlarge-nightly",
            message=prompt,
            temperature=0.3,
            max_tokens=200
        )
        return response.text.strip()
    except Exception as e:
        return f"❌ API Error: {str(e)}"

# === Fix for Chrome PDF preview using PDF.js ===
def display_pdf_using_pdfjs(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    pdf_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            html, body {{ margin: 0; padding: 0; height: 100%; }}
            iframe {{ width: 100%; height: 100%; border: none; }}
        </style>
    </head>
    <body>
        <iframe src="https://mozilla.github.io/pdf.js/web/viewer.html?file=data:application/pdf;base64,{base64_pdf}"></iframe>
    </body>
    </html>
    """
    components.html(pdf_html, height=600, scrolling=False)

# === Convert result to CSV ===
def prepare_csv(content):
    lines = content.split("\n")
    rows = []
    question = ""
    answer = ""
    if "Q1:" in content or "A1:" in content:
        for line in lines:
            if line.strip().startswith("Q") and ":" in line:
                question = line.split(":", 1)[1].strip()
            elif line.strip().startswith("A") and ":" in line:
                answer = line.split(":", 1)[1].strip()
                rows.append(f'"{question}","{answer}"')
        csv_content = "\n".join(["Question,Answer"] + rows)
    else:
        csv_content = f"Text\n\"{content}\""
    return csv_content

# === Get file download data ===
def get_file_data(content, file_format):
    if file_format == "txt" or file_format == "doc":
        return content.encode()
    elif file_format == "csv":
        csv_content = prepare_csv(content)
        return csv_content.encode()
    return None

# === Streamlit App ===
st.set_page_config(layout="wide")
def main():
    st.title("📄 PDF Summarizer + Q&A Tool")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is None:
        st.info("Please upload a PDF to start.")
        return

    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("❌ File too large! Please upload a PDF under 10MB.")
        return

    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    full_text = extract_text_from_pdf(filepath)
    if not full_text.strip():
        st.warning("⚠ No readable text found in the PDF.")
        return

    if len(full_text) > 100_000:
        st.warning("⚠ PDF content too long. Using only first 100,000 characters.")
        full_text = full_text[:100_000]

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("### 📄 PDF Preview")
        display_pdf_using_pdfjs(filepath)

    with col2:
        st.markdown("### What would you like to do?")
        option = st.radio("Choose an option:", ["📄 Summarize", "❓ Q&A"])

        if option == "📄 Summarize":
            if st.button("Generate Summary"):
                with st.spinner("Summarizing..."):
                    summary = summarize_text(full_text)
                st.subheader("📝 Summary")
                st.write(summary)

                format = st.selectbox("Download as:", ["txt", "doc", "csv"])
                data = get_file_data(summary, format)
                st.download_button("Download Summary", data, f"summary.{format}")

        elif option == "❓ Q&A":
            qa_mode = st.radio("Q&A Type:", ["🧠 Auto-generate Q&A", "🗨 Ask Your Own Question"])
            if qa_mode == "🧠 Auto-generate Q&A":
                num_q = st.slider("Number of Questions", 1, 10, 3)
                if st.button("Generate Q&A"):
                    with st.spinner("Generating..."):
                        qa = generate_auto_qa(full_text, num_q)
                    st.subheader("📚 Q&A")
                    st.write(qa)

                    format = st.selectbox("Download as:", ["txt", "doc", "csv"])
                    data = get_file_data(qa, format)
                    st.download_button("Download Q&A", data, f"auto_qa.{format}")
            else:
                q = st.text_input("Your question:")
                if st.button("Get Answer"):
                    with st.spinner("Answering..."):
                        ans = generate_answer(full_text, q)
                    st.subheader("💬 Answer")
                    st.markdown(f"**Q: {q}**")
                    st.markdown(f"A: {ans}")

                    format = st.selectbox("Download as:", ["txt", "doc", "csv"])
                    data = get_file_data(f"Q: {q}\nA: {ans}", format)
                    st.download_button("Download Answer", data, f"custom_qa.{format}")

if __name__ == "__main__":
    main()

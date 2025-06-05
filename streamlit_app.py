import streamlit as st
import streamlit.components.v1 as components
import base64
import PyPDF2
import cohere
import time

# Initialize Cohere client from Streamlit secrets
co = cohere.Client(st.secrets["cohere"]["api_key"])

def display_pdf(file):
    """Show PDF preview in Streamlit using base64 data URI iframe."""
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'''
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="700"
            type="application/pdf"
        ></iframe>
    '''
    file.seek(0)  # reset file pointer after reading
    components.html(pdf_display, height=700)

def extract_text_from_pdf(file):
    """Extract all text from PDF file-like object."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    file.seek(0)  # reset pointer so file can be reused
    return text

def chunk_text(text, max_len=2500):
    """Split text into chunks close to max_len, ending at sentence."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_len
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        # Try to break at last period before max_len
        period_pos = text.rfind('.', start, end)
        if period_pos == -1 or period_pos <= start:
            period_pos = end  # fallback to max_len chunk
        chunks.append(text[start:period_pos+1].strip())
        start = period_pos + 1
    return chunks

def cohere_summarize(text_chunk):
    """Call Cohere chat model to summarize a text chunk."""
    try:
        response = co.chat(
            model="command-xlarge-nightly",
            message=f"Summarize this text clearly and concisely:\n\n{text_chunk}",
            temperature=0.4,
            max_tokens=300
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def summarize_text(text):
    """Split large text into chunks and summarize each, then combine."""
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        summary = cohere_summarize(chunk)
        summaries.append(summary)
        time.sleep(6)  # to avoid rate limits
    combined_summary = " ".join(summaries)
    # Optional: summarize combined if too long
    if len(combined_summary) > 2000:
        combined_summary = cohere_summarize(combined_summary[:2000])
    return combined_summary

def generate_questions_answers(text, num_questions=5):
    """Generate Q&A pairs from text using Cohere chat."""
    prompt = (
        f"Generate {num_questions} question and answer pairs based on the following text:\n\n{text}\n\n"
        "Format as:\nQ1: ...\nA1: ...\nQ2: ...\nA2: ... and so on."
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
        return f"Error: {str(e)}"

def answer_question(text, question):
    """Answer user question based on text using Cohere chat."""
    prompt = f"Based on the following text, answer the question:\n\n{text}\n\nQuestion: {question}"
    try:
        response = co.chat(
            model="command-xlarge-nightly",
            message=prompt,
            temperature=0.3,
            max_tokens=200
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def prepare_download_data(content, fmt):
    """Prepare downloadable content bytes for given format."""
    if fmt == "txt":
        return content.encode("utf-8")
    elif fmt == "doc":
        return content.encode("utf-8")  # simple doc is just text with .doc ext
    elif fmt == "csv":
        # For Q&A, convert to CSV; for summary just one column
        lines = content.split('\n')
        rows = []
        for line in lines:
            if line.strip().startswith('Q') and ':' in line:
                question = line.split(':', 1)[1].strip()
            elif line.strip().startswith('A') and ':' in line:
                answer = line.split(':', 1)[1].strip()
                rows.append(f'"{question}","{answer}"')
        csv_text = "Question,Answer\n" + "\n".join(rows) if rows else content
        return csv_text.encode("utf-8")
    else:
        return content.encode("utf-8")

# === Streamlit UI ===

st.set_page_config(page_title="PDF Summarizer + Q&A", layout="wide")

st.title("📄 PDF Summarizer + Q&A Tool")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Show PDF preview
    st.markdown("### PDF Preview")
    display_pdf(uploaded_file)

    # Extract text for processing
    try:
        full_text = extract_text_from_pdf(uploaded_file)
    except Exception as e:
        st.error(f"Failed to extract text from PDF: {e}")
        st.stop()

    if not full_text.strip():
        st.warning("⚠ No readable text found in this PDF.")
        st.stop()

    # Limit length to avoid API limits
    if len(full_text) > 100_000:
        st.warning("⚠ PDF text too large. Truncating to first 100,000 characters.")
        full_text = full_text[:100_000]

    action = st.radio("What would you like to do?", ["📄 Summarize", "❓ Q&A"])

    if action == "📄 Summarize":
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = summarize_text(full_text)
            st.markdown("### 📝 Summary")
            st.write(summary)
            download_format = st.selectbox("Download summary as", ["txt", "doc", "csv"])
            data = prepare_download_data(summary, download_format)
            st.download_button("Download Summary", data=data, file_name=f"summary.{download_format}")

    elif action == "❓ Q&A":
        qa_option = st.radio("Choose Q&A option", ["Generate Questions", "Ask a Question"])

        if qa_option == "Generate Questions":
            num_qs = st.slider("Number of questions to generate", 1, 10, 3)
            if st.button("Generate Q&A"):
                with st.spinner("Generating questions and answers..."):
                    qa_text = generate_questions_answers(full_text, num_qs)
                st.markdown("### Generated Questions & Answers")
                st.text(qa_text)
                download_format = st.selectbox("Download Q&A as", ["txt", "doc", "csv"])
                data = prepare_download_data(qa_text, download_format)
                st.download_button("Download Q&A", data=data, file_name=f"qa.{download_format}")

        else:  # Ask a Question
            question = st.text_input("Enter your question")
            if st.button("Get Answer") and question.strip():
                with st.spinner("Getting answer..."):
                    answer = answer_question(full_text, question)
                st.markdown("### Answer")
                st.write(f"**Q:** {question}")
                st.write(f"**A:** {answer}")
                download_format = st.selectbox("Download answer as", ["txt", "doc", "csv"])
                data = prepare_download_data(f"Q: {question}\nA: {answer}", download_format)
                st.download_button("Download Answer", data=data, file_name=f"answer.{download_format}")

else:
    st.info("Please upload a PDF file to begin.")


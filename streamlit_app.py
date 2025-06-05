import streamlit as st
import cohere
import PyPDF2
import base64
import os
import time

# === Secure API key from secrets.toml ===
co = cohere.Client(st.secrets["cohere"]["api_key"])

def extract_text_from_pdf(file_path):
    pdf_reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

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
        time.sleep(6)  # rate limit safety
    combined = " ".join(summaries)
    if len(combined) > 2000:
        return cohere_chat_summary(combined[:2000])
    return combined

def generate_auto_qa(text, num_questions=5):
    prompt = f"Generate {num_questions} questions and answers from the following text:\n\n{text}\n\nFormat: Q1: ... A1: ... Q2: ... A2: ..."
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

@st.cache_data(show_spinner=False)
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_view = f'''
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%" height="600"
        style="border:none;"
        sandbox="allow-scripts allow-same-origin allow-popups">
    </iframe>
    '''
    return pdf_view

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

def get_file_data(content, file_format):
    if file_format == "txt" or file_format == "doc":
        return content.encode()
    elif file_format == "csv":
        csv_content = prepare_csv(content)
        return csv_content.encode()
    return None

st.set_page_config(layout="wide")

def main():
    st.title("📄 PDF Summarizer + Q&A Tool")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_uploader")

    if uploaded_file is None:
        st.session_state.clear()
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
        pdf_html = display_pdf(filepath)
        st.markdown(pdf_html, unsafe_allow_html=True)

    with col2:
        st.markdown("### What would you like to do?")
        option = st.radio("Choose an option:", ["📄 Summarize", "❓ Q&A"])

        if option == "📄 Summarize":
            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    summary = summarize_text(full_text)
                st.subheader("📝 Summary")
                st.write(summary)

                file_format = st.selectbox("Select format to download", ["txt", "doc", "csv"])
                file_data = get_file_data(summary, file_format)
                st.download_button(
                    label="Download Summary",
                    data=file_data,
                    file_name=f"summary.{file_format}",
                    mime="text/plain" if file_format == "txt" else "application/msword" if file_format == "doc" else "text/csv"
                )

        elif option == "❓ Q&A":
            qa_mode = st.radio("Choose Q&A Type:", ["🧠 Generate Questions", "🗨 Ask Your Question"])

            if qa_mode == "🧠 Generate Questions":
                num_qs = st.slider("Number of Questions", 1, 10, 3)
                if st.button("Generate Q&A"):
                    with st.spinner("Generating Q&A..."):
                        result = generate_auto_qa(full_text, num_qs)
                    st.subheader("📚 Generated Q&A")
                    st.write(result)

                    file_format = st.selectbox("Select format to download", ["txt", "doc", "csv"])
                    file_data = get_file_data(result, file_format)
                    st.download_button(
                        label="Download Q&A",
                        data=file_data,
                        file_name=f"qa.{file_format}",
                        mime="text/plain" if file_format == "txt" else "application/msword" if file_format == "doc" else "text/csv"
                    )

            elif qa_mode == "🗨 Ask Your Question":
                user_question = st.text_input("Enter your question:")
                if st.button("Get Answer"):
                    with st.spinner("Finding the answer..."):
                        result = generate_answer(full_text, user_question)
                    st.subheader("💬 Answer")
                    st.markdown(f"**Q: {user_question}**")
                    st.markdown(f"A: {result}")

                    file_format = st.selectbox("Select format to download", ["txt", "doc", "csv"])
                    file_data = get_file_data(f"Q: {user_question}\nA: {result}", file_format)
                    st.download_button(
                        label="Download Answer",
                        data=file_data,
                        file_name=f"answer.{file_format}",
                        mime="text/plain" if file_format == "txt" else "application/msword" if file_format == "doc" else "text/csv"
                    )

if __name__ == "__main__":
    main()

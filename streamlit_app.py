import streamlit as st
import cohere
import PyPDF2
import base64
import os
import time
from io import BytesIO

# === Secure API key from secrets ===
co = cohere.Client(st.secrets["cohere"]["api_key"])

def extract_text_from_pdf(file_stream):
    pdf_reader = PyPDF2.PdfReader(file_stream)
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
        time.sleep(6)
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
def render_pdf_view(pdf_bytes):
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    return f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

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

    if uploaded_file is None and st.session_state.get("last_uploaded_file") is not None:
        for key in ["output", "output_type", "show_download_options", "selected_format", "last_option", "last_qa_mode", "last_uploaded_file"]:
            if key in st.session_state:
                del st.session_state[key]

    if uploaded_file:
        if st.session_state.get("last_uploaded_file") != uploaded_file.name:
            for key in ["output", "output_type", "show_download_options", "selected_format", "last_option", "last_qa_mode"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.last_uploaded_file = uploaded_file.name

        pdf_bytes = uploaded_file.read()

        full_text = extract_text_from_pdf(BytesIO(pdf_bytes))
        if not full_text.strip():
            st.warning("⚠ No readable text found in the PDF.")
            return

        if len(full_text) > 100_000:
            st.warning("⚠ PDF content too long. Using only first 100,000 characters.")
            full_text = full_text[:100_000]

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown("### 📄 PDF Preview")
            pdf_html = render_pdf_view(pdf_bytes)
            st.markdown(pdf_html, unsafe_allow_html=True)

        with col2:
            st.markdown("### What would you like to do?")
            option = st.radio("Choose an option:", ["📄 Summarize", "❓ Q&A"], key="main_option")

            if "output" not in st.session_state:
                st.session_state.output = ""
            if "output_type" not in st.session_state:
                st.session_state.output_type = ""
            if "show_download_options" not in st.session_state:
                st.session_state.show_download_options = False
            if "selected_format" not in st.session_state:
                st.session_state.selected_format = None

            def reset_download():
                st.session_state.show_download_options = False
                st.session_state.selected_format = None

            if "last_option" not in st.session_state or st.session_state.last_option != option:
                st.session_state.output = ""
                st.session_state.output_type = ""
                reset_download()
                st.session_state.last_option = option

            if option == "📄 Summarize":
                if st.button("Generate Summary", key="gen_summary"):
                    with st.spinner("Generating summary..."):
                        summary = summarize_text(full_text)
                    st.session_state.output = summary
                    st.session_state.output_type = "summary"
                    reset_download()

                if st.session_state.output_type == "summary" and st.session_state.output:
                    st.subheader("📝 Summary")
                    st.write(st.session_state.output)

                    if not st.session_state.show_download_options:
                        if st.button("Download Summary", key="download_summary_button"):
                            st.session_state.show_download_options = True

                    if st.session_state.show_download_options:
                        selected = st.selectbox("Select download format", ["txt", "doc", "csv"], key="download_format_summary")
                        st.session_state.selected_format = selected

                        if st.session_state.selected_format:
                            file_data = get_file_data(st.session_state.output, st.session_state.selected_format)
                            file_name = f"summary.{st.session_state.selected_format}"
                            st.download_button("Download", data=file_data, file_name=file_name,
                                               mime=("text/plain" if file_name.endswith(".txt") else
                                                     "application/msword" if file_name.endswith(".doc") else
                                                     "text/csv"), key="download_summary_file")

            elif option == "❓ Q&A":
                qa_mode = st.radio("Choose Q&A Type:", ["🧠 Generate Questions", "🗨 Ask Your Question"], key="qa_mode")

                if "last_qa_mode" not in st.session_state or st.session_state.last_qa_mode != qa_mode:
                    st.session_state.output = ""
                    st.session_state.output_type = ""
                    reset_download()
                    st.session_state.last_qa_mode = qa_mode

                if qa_mode == "🧠 Generate Questions":
                    num_qs = st.slider("Number of Questions", 1, 10, 3, key="num_questions")
                    if st.button("Generate Q&A", key="gen_auto_qa"):
                        with st.spinner("Generating questions and answers..."):
                            result = generate_auto_qa(full_text, num_qs)
                        st.session_state.output = result
                        st.session_state.output_type = "auto_qa"
                        reset_download()

                    if st.session_state.output_type == "auto_qa" and st.session_state.output:
                        st.subheader("📚 Generated Q&A")
                        st.write(st.session_state.output)

                        if not st.session_state.show_download_options:
                            if st.button("Download Q&A", key="download_auto_qa_button"):
                                st.session_state.show_download_options = True

                        if st.session_state.show_download_options:
                            selected = st.selectbox("Select download format", ["txt", "doc", "csv"], key="download_format_auto_qa")
                            st.session_state.selected_format = selected

                            if st.session_state.selected_format:
                                file_data = get_file_data(st.session_state.output, st.session_state.selected_format)
                                file_name = f"auto_qa.{st.session_state.selected_format}"
                                st.download_button("Download", data=file_data, file_name=file_name,
                                                   mime=("text/plain" if file_name.endswith(".txt") else
                                                         "application/msword" if file_name.endswith(".doc") else
                                                         "text/csv"), key="download_auto_qa_file")

                elif qa_mode == "🗨 Ask Your Question":
                    user_question = st.text_input("Enter your question:", key="user_question")
                    if st.button("Get Answer", key="get_answer"):
                        with st.spinner("Finding the answer..."):
                            result = generate_answer(full_text, user_question)
                        st.session_state.output = f"Q: {user_question}\nA: {result}"
                        st.session_state.output_type = "custom_qa"
                        reset_download()

                    if st.session_state.output_type == "custom_qa" and st.session_state.output:
                        st.subheader("💬 Answer")
                        try:
                            q, a = st.session_state.output.split('\n', 1)
                            st.markdown(f"{q}")
                            st.markdown(a)
                        except Exception:
                            st.markdown(st.session_state.output)

                        if not st.session_state.show_download_options:
                            if st.button("Download Answer", key="download_custom_qa_button"):
                                st.session_state.show_download_options = True

                        if st.session_state.show_download_options:
                            selected = st.selectbox("Select download format", ["txt", "doc", "csv"], key="download_format_custom_qa")
                            st.session_state.selected_format = selected

                            if st.session_state.selected_format:
                                file_data = get_file_data(st.session_state.output, st.session_state.selected_format)
                                file_name = f"custom_qa.{st.session_state.selected_format}"
                                st.download_button("Download", data=file_data, file_name=file_name,
                                                   mime=("text/plain" if file_name.endswith(".txt") else
                                                         "application/msword" if file_name.endswith(".doc") else
                                                         "text/csv"), key="download_custom_qa_file")
    else:
        st.info("Please upload a PDF to start.")

if __name__ == "__main__":
    main()

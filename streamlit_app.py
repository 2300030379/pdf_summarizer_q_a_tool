import streamlit as st
import cohere
import PyPDF2
import os
import time

# Initialize Cohere client with your API key stored in Streamlit secrets
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
        time.sleep(6)  # Rate limiting
    combined = " ".join(summaries)
    if len(combined) > 2000:
        return cohere_chat_summary(combined[:2000])
    return combined

def generate_auto_qa(text, num_questions=5):
    prompt = (
        f"Generate {num_questions} questions and answers from the following text:\n\n{text}\n\n"
        "Format: Q1: ... A1: ... Q2: ... A2: ..."
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

def main():
    st.title("📄 PDF Summarizer + Q&A Tool")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Save uploaded PDF temporarily
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.markdown("### PDF File Info")
        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**File size:** {uploaded_file.size / 1024:.2f} KB")

        # Provide download button for user to open PDF externally (avoids embed issues)
        st.download_button("📥 Download PDF", data=uploaded_file, file_name=uploaded_file.name)

        # Extract text from PDF
        full_text = extract_text_from_pdf(temp_path)

        if not full_text.strip():
            st.warning("⚠ No readable text found in the PDF.")
            return

        if len(full_text) > 100_000:
            st.warning("⚠ PDF content too long. Using only first 100,000 characters.")
            full_text = full_text[:100_000]

        st.markdown("---")
        st.markdown("### What would you like to do?")
        option = st.radio("", ["📄 Summarize", "❓ Q&A"])

        if option == "📄 Summarize":
            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    summary = summarize_text(full_text)
                st.markdown("### 📝 Summary")
                st.write(summary)

                st.download_button("Download Summary (txt)", data=summary, file_name="summary.txt")

        elif option == "❓ Q&A":
            qa_mode = st.radio("Choose Q&A Mode:", ["🧠 Generate Questions", "🗨 Ask Your Question"])

            if qa_mode == "🧠 Generate Questions":
                num_questions = st.slider("Number of questions", 1, 10, 3)
                if st.button("Generate Q&A"):
                    with st.spinner("Generating questions and answers..."):
                        qa_text = generate_auto_qa(full_text, num_questions)
                    st.markdown("### 📚 Generated Q&A")
                    st.text(qa_text)

                    st.download_button("Download Q&A (txt)", data=qa_text, file_name="generated_qa.txt")

            elif qa_mode == "🗨 Ask Your Question":
                user_question = st.text_input("Enter your question:")
                if st.button("Get Answer"):
                    if user_question.strip():
                        with st.spinner("Getting answer..."):
                            answer = generate_answer(full_text, user_question)
                        st.markdown("### 💬 Answer")
                        st.write(f"**Q:** {user_question}")
                        st.write(f"**A:** {answer}")
                    else:
                        st.warning("Please enter a question.")

if __name__ == "__main__":
    main()

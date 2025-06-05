import streamlit as st
import cohere
import PyPDF2
import time

# === Initialize Cohere client ===
co = cohere.Client(st.secrets["cohere"]["api_key"])

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text_by_page = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text_by_page.append((i + 1, text.strip()))
    return text_by_page

def chunk_text(text, max_chunk_size=2500):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        last_dot = text.rfind(".", start, end)
        if last_dot != -1:
            end = last_dot + 1
        chunks.append(text[start:end].strip())
        start = end
    return chunks

def summarize_text(full_text):
    chunks = chunk_text(full_text)
    summaries = []
    for chunk in chunks:
        response = co.chat(
            model="command-xlarge-nightly",
            message=f"Summarize this:\n\n{chunk}",
            temperature=0.4,
            max_tokens=300
        )
        summaries.append(response.text.strip())
        time.sleep(5)
    return " ".join(summaries)

def generate_auto_qa(text, num_questions=5):
    prompt = f"Generate {num_questions} questions and answers from the following text:\n\n{text}"
    response = co.chat(
        model="command-xlarge-nightly",
        message=prompt,
        temperature=0.5,
        max_tokens=600
    )
    return response.text.strip()

def answer_question(text, question):
    prompt = f"Answer the question based on this text:\n\n{text}\n\nQuestion: {question}"
    response = co.chat(
        model="command-xlarge-nightly",
        message=prompt,
        temperature=0.3,
        max_tokens=200
    )
    return response.text.strip()

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("📄 PDF Summarizer + Q&A Tool")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Extracting PDF..."):
        pages = extract_text_from_pdf(uploaded_file)
        combined_text = "\n".join(text for _, text in pages)

    st.markdown("## 📚 PDF Page Text Preview")
    for page_num, page_text in pages[:5]:  # Preview first 5 pages only
        with st.expander(f"Page {page_num}"):
            st.write(page_text if page_text else "⚠ No extractable text on this page.")

    st.markdown("---")
    choice = st.radio("Choose an action:", ["📄 Summarize", "❓ Generate Q&A", "💬 Ask a Question"])

    if choice == "📄 Summarize":
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = summarize_text(combined_text[:100000])
            st.subheader("📝 Summary")
            st.write(summary)

    elif choice == "❓ Generate Q&A":
        num_qs = st.slider("Number of Questions", 1, 10, 5)
        if st.button("Generate Q&A"):
            with st.spinner("Generating..."):
                qa = generate_auto_qa(combined_text[:100000], num_qs)
            st.subheader("📘 Q&A")
            st.text(qa)

    elif choice == "💬 Ask a Question":
        q = st.text_input("Your question:")
        if st.button("Get Answer"):
            with st.spinner("Answering..."):
                a = answer_question(combined_text[:100000], q)
            st.subheader("💬 Answer")
            st.markdown(f"*Q:* {q}")
            st.markdown(f"*A:* {a}")

else:
    st.info("Upload a PDF to begin.")

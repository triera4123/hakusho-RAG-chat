# -*- coding: utf-8 -*-
"""
白書RAGチャット（ポートフォリオ公開版）
- docs/ フォルダ内の複数PDFを読み込み、RAGで質問応答する
- 文書指定フィルタ（メタデータフィルタ）で複数文書の混線を防止
- LLMは環境変数 LLM_PROVIDER で Azure OpenAI / Google Gemini を切り替え可能
"""
import os
import glob

import streamlit as st

# Streamlit CloudのSecretsに設定した値を環境変数に反映する
# （ローカル実行時はSecrets未設定でもエラーにならないようにする）
try:
    for key in ("LLM_PROVIDER", "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT", "GOOGLE_API_KEY"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except FileNotFoundError:
    pass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI

# ===== 設定 =====
DOCS_DIR = "docs"          # 読み込むPDFの置き場所
MAX_QUESTIONS = 10         # 1セッションあたりの質問回数上限（APIコスト保護）
MAX_INPUT_CHARS = 200      # 質問の最大文字数（APIコスト保護）
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
RETRIEVE_K = 4             # 検索で取得するチャンク数

PROMPT_TEMPLATE = """以下の資料の抜粋だけを根拠に、質問に日本語で答えてください。
資料に記載がない内容は推測せず、「資料に記載がありません」と答えてください。

# 資料の抜粋
{context}

# 質問
{question}
"""


@st.cache_resource(show_spinner="文書を読み込んでいます（初回は数分かかります）...")
def build_vectorstore():
    """docs/ 内の全PDFを読み込み、文書名をメタデータに付与してベクトルDBを構築する"""
    pdf_paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
    if not pdf_paths:
        return None, []

    documents = []
    doc_names = []
    for path in pdf_paths:
        doc_name = os.path.splitext(os.path.basename(path))[0]
        doc_names.append(doc_name)
        pages = PyPDFLoader(path).load()
        for p in pages:
            # どの文書由来かをメタデータに記録する（混線対策の要）
            p.metadata["doc_name"] = doc_name
        documents.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore, doc_names


@st.cache_resource
def build_llm():
    """環境変数 LLM_PROVIDER に応じてLLMクライアントを初期化する"""
    provider = os.environ.get("LLM_PROVIDER", "gemini")
    if provider == "azure":
        return AzureChatOpenAI(
            azure_deployment="gpt-5-mini",
            api_version="2024-12-01-preview",
        ), "Azure OpenAI (gpt-5-mini)"
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash"), "Google Gemini (gemini-2.5-flash)"


# ===== 画面 =====
st.set_page_config(page_title="白書RAGチャット", page_icon="📄")
st.title("📄 白書RAGチャット")
st.caption("官公庁の白書PDFを読み込んだRAG（検索拡張生成）のデモです。"
           "回答は資料の記載のみを根拠とし、参照元ページを表示します。")
st.caption("Developed by **K.U.** ｜ "
           "[ソースコード（GitHub）](https://github.com/triera4123/hakusho-RAG-chat)")

vectorstore, doc_names = build_vectorstore()

if vectorstore is None:
    st.error(f"`{DOCS_DIR}/` フォルダにPDFが見つかりません。PDFを配置してから再起動してください。")
    st.stop()

llm, llm_label = build_llm()

# サイドバー：文書指定フィルタ（メタデータフィルタによる混線対策）
with st.sidebar:
    st.header("検索対象の文書")
    target = st.selectbox("文書を指定する", ["すべて"] + doc_names)
    st.caption("特定の文書を選ぶと、その文書のチャンクだけを検索対象にします"
               "（メタデータフィルタ）。複数文書の内容が混ざる「混線」を防ぎます。")
    st.divider()
    st.caption(f"LLM: {llm_label}")
    st.caption(f"読み込み文書数: {len(doc_names)}")
    st.divider()
    st.caption("製作: K.U.")
    st.caption("[GitHub リポジトリ](https://github.com/triera4123/hakusho-RAG-chat)")

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# 過去のやり取りを表示
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 質問回数の上限チェック（公開デモのためのAPIコスト保護）
if st.session_state.question_count >= MAX_QUESTIONS:
    st.info(f"デモのため、質問は1セッション{MAX_QUESTIONS}回までに制限しています。"
            "ページを再読み込みすると続けられます。")
    st.stop()

question = st.chat_input(f"質問を入力してください（{MAX_INPUT_CHARS}文字まで）")

if question:
    if len(question) > MAX_INPUT_CHARS:
        st.warning(f"質問は{MAX_INPUT_CHARS}文字以内でお願いします。")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 検索：文書が指定されていればメタデータフィルタを適用
    search_kwargs = {"k": RETRIEVE_K}
    if target != "すべて":
        search_kwargs["filter"] = {"doc_name": target}
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(question)

    context = "\n\n".join(
        f"【{d.metadata.get('doc_name', '不明')} p.{d.metadata.get('page', '?')}】\n{d.page_content}"
        for d in docs
    )

    with st.chat_message("assistant"):
        with st.spinner("回答を生成しています..."):
            response = llm.invoke(
                PROMPT_TEMPLATE.format(context=context, question=question)
            )
        st.markdown(response.content)

        # 参照元の表示（RAGの根拠の可視化）
        with st.expander("参照した箇所"):
            for d in docs:
                st.markdown(
                    f"- **{d.metadata.get('doc_name', '不明')}** "
                    f"p.{d.metadata.get('page', '?')}："
                    f"{d.page_content[:80]}..."
                )

    st.session_state.messages.append(
        {"role": "assistant", "content": response.content}
    )
    st.session_state.question_count += 1

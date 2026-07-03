# 白書RAGチャット

官公庁の白書PDFを知識源としたRAG（検索拡張生成）チャットアプリです。
生成AI／RAGの技術検証を目的に個人開発しました。

**🔗 デモ（ブラウザでそのまま試せる）:**
https://hakusho-rag-chat-developed-by-ku.streamlit.app/

読み込んでいる文書：

- 総務省「令和7年版 情報通信白書」
- 文部科学省「令和6年版 科学技術・イノベーション白書」（章別）

「日本企業の生成AI活用は海外と比べてどうですか？」のような質問を試してみてください。

## 構成

```
PDF読み込み（PyPDFLoader）
  → チャンク分割（RecursiveCharacterTextSplitter）
  → ベクトル化（HuggingFace multilingual-MiniLM ※ローカル実行）
  → ベクトルDB（Chroma）
  → 検索（メタデータフィルタ対応）
  → 回答生成（Azure OpenAI gpt-5-mini / Google Gemini 切り替え式）
  → UI（Streamlit）
```

## 工夫した点

### 1. 複数文書の「混線」対策（メタデータフィルタ）

複数のPDFを1つのベクトルDBに入れると、ベクトル検索は文書の境界を区別しない
ため、ある文書について質問しても別の文書のチャンクが根拠に混ざる「混線」が
起きます。本アプリでは各チャンクに文書名をメタデータとして付与し、サイドバー
で文書を指定すると検索対象をその文書に限定できるようにしました。

### 2. ハルシネーション対策

開発中、資料に記載のない事項をモデルが一般知識から補完して回答する現象を確認
したため、プロンプトで「資料に記載がない内容は推測せず『資料に記載がありません』
と答える」よう制約しています。

### 3. 回答根拠の可視化

各回答の下に、参照したチャンクの文書名・ページ番号・冒頭テキストを表示し、
回答の根拠を利用者が検証できるようにしています。

### 4. マルチクラウド対応

LLM呼び出し部を環境変数（LLM_PROVIDER）で Azure OpenAI / Google Gemini に
切り替えられる構成にしています。Embeddingはローカルモデルのため、生成部分
だけを独立して差し替えられます。

### 5. 公開デモとしての安全設計

- APIキーはコードに含めず、環境変数 / Streamlit Secrets で管理
- 1セッションあたりの質問回数・入力文字数を制限（APIコストの保護）
- 知識源には再配布可能な官公庁白書のみを使用（開発時の検証には非公開文書を
  使用していたため、公開版ではデータを差し替え）

## ローカルでの実行方法

```bash
pip install -r requirements.txt

# docs/ フォルダに任意のPDFを配置

# Azure OpenAI を使う場合
set LLM_PROVIDER=azure
set AZURE_OPENAI_API_KEY=<your-key>
set AZURE_OPENAI_ENDPOINT=<your-endpoint>

# （または Google Gemini を使う場合）
set GOOGLE_API_KEY=<your-key>

streamlit run hakusho_rag_chat.py
```

## 【開発者向け】ソースのpush

1. ローカルの「hakusho_rag_chat.py」ファイルを変更
2. コマンドプロンプトで以下を実行
```bash
cd <"hakusho-rag-chat" directory>
git add .
git commit -m "Add author info"
git push
```

## 使用技術

Python / LangChain / Chroma / sentence-transformers / Streamlit /
Azure OpenAI Service (gpt-5-mini) / Google Gemini API

## 出典

本アプリが読み込んでいる白書は、各府省のWebサイトで公開されているPDFです。

- 総務省「令和7年版 情報通信白書」
- 文部科学省「令和6年版 科学技術・イノベーション白書」

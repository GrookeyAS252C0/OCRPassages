import streamlit as st
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime
import io
import zipfile

# Streamlit Cloud環境でのNLTKデータダウンロード
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

from pdf_text_extractor import PDFTextExtractor

# ページ設定
st.set_page_config(
    page_title="PDF OCR Processor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSSスタイル
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.success-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    margin: 1rem 0;
}
.error-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    margin: 1rem 0;
}
.info-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# メインタイトル
st.markdown('<h1 class="main-header">📄 PDF OCR Processor</h1>', unsafe_allow_html=True)
st.markdown("---")

# サイドバー設定
st.sidebar.title("⚙️ 設定")
enhancement_level = st.sidebar.selectbox(
    "OCR処理レベル",
    ["light", "standard", "aggressive"],
    index=1,
    help="処理レベルが高いほど精度向上しますが、時間がかかります"
)

st.sidebar.markdown("### 処理レベル説明")
level_descriptions = {
    "light": "🟢 軽量処理 - 高速ですが基本的なOCR",
    "standard": "🟡 標準処理 - バランスの取れた精度と速度",
    "aggressive": "🔴 高精度処理 - 最高精度ですが時間がかかります"
}
st.sidebar.markdown(f"**{enhancement_level}**: {level_descriptions[enhancement_level]}")

# OpenAI API設定
st.sidebar.markdown("### OpenAI API設定")
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="OCR結果の校正に使用されます。設定しなくても基本的なOCRは実行されます。"
)

if api_key:
    os.environ['OPENAI_API_KEY'] = api_key
    st.sidebar.success("API Key設定完了")

# メインコンテンツ
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📁 PDFファイルアップロード")
    uploaded_files = st.file_uploader(
        "PDFファイルを選択してください",
        type=['pdf'],
        accept_multiple_files=True,
        help="複数のPDFファイルを同時にアップロードできます"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)}個のファイルが選択されました")
        for file in uploaded_files:
            st.write(f"📄 {file.name} ({file.size / 1024:.1f} KB)")

with col2:
    st.markdown("### 🔧 処理オプション")
    
    show_progress = st.checkbox("処理進捗を表示", value=True)
    show_word_list = st.checkbox("抽出単語リストを表示", value=True)
    show_passages = st.checkbox("英語文章を表示", value=True)
    include_stats = st.checkbox("詳細統計を含める", value=True)

def process_files(uploaded_files, enhancement_level, show_progress, show_word_list, show_passages, include_stats):
    """
    アップロードされたPDFファイルを処理
    """
    try:
        # PDFTextExtractorを初期化
        extractor = PDFTextExtractor()
        
        results = []
        
        # プログレスバー表示
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            if show_progress:
                status_text.text(f"処理中: {uploaded_file.name}")
                progress_bar.progress((i) / len(uploaded_files))
            
            # 一時ファイルに保存
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            
            try:
                # OCR処理実行
                result = extractor.process_pdf(temp_file_path, enhancement_level)
                
                # 結果を整理
                processed_result = {
                    'file_info': {
                        'source_file': uploaded_file.name,
                        'file_size': uploaded_file.size,
                        'processed_pages': result.get('pages_processed', 0),
                        'ocr_confidence': result.get('processing_stats', {}).get('average_confidence', 0.0),
                        'processing_level': enhancement_level,
                        'processing_timestamp': datetime.now().isoformat(),
                        'error': result.get('error', None)
                    },
                    'extraction_results': {
                        'total_words': len(result.get('extracted_words', [])),
                        'unique_words': len(set(result.get('extracted_words', []))),
                        'english_passages_count': len(result.get('pure_english_text', [])),
                        'ocr_attempts': result.get('processing_stats', {}).get('total_ocr_attempts', 0),
                        'successful_extractions': result.get('processing_stats', {}).get('successful_extractions', 0)
                    },
                    'content': {
                        'english_passages': result.get('pure_english_text', []),
                        'extracted_words': sorted(list(set(result.get('extracted_words', []))))
                    }
                }
                
                results.append(processed_result)
                
            except Exception as e:
                st.error(f"❌ {uploaded_file.name}の処理中にエラーが発生しました: {str(e)}")
                results.append({
                    'file_info': {
                        'source_file': uploaded_file.name,
                        'error': str(e)
                    },
                    'extraction_results': {},
                    'content': {}
                })
            
            finally:
                # 一時ファイルを削除
                os.unlink(temp_file_path)
        
        if show_progress:
            progress_bar.progress(1.0)
            status_text.text("処理完了！")
        
        # 結果をセッションステートに保存
        st.session_state.results = results
        
        # 成功メッセージ
        successful_files = [r for r in results if not r['file_info'].get('error')]
        if successful_files:
            st.markdown('<div class="success-box">✅ 処理が完了しました！</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f'<div class="error-box">❌ 処理中にエラーが発生しました: {str(e)}</div>', unsafe_allow_html=True)

def display_results(results, show_word_list, show_passages, include_stats):
    """
    処理結果を表示
    """
    st.markdown("---")
    st.markdown("## 📊 処理結果")
    
    # 全体統計
    total_words = sum(r['extraction_results'].get('total_words', 0) for r in results)
    total_unique_words = len(set(word for r in results for word in r['content'].get('extracted_words', [])))
    total_passages = sum(r['extraction_results'].get('english_passages_count', 0) for r in results)
    successful_files = [r for r in results if not r['file_info'].get('error')]
    
    # 統計表示
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("処理ファイル数", f"{len(successful_files)}/{len(results)}")
    with col2:
        st.metric("抽出単語総数", total_words)
    with col3:
        st.metric("ユニーク単語数", total_unique_words)
    with col4:
        st.metric("英語文章数", total_passages)
    
    # 各ファイルの詳細結果
    for i, result in enumerate(results):
        with st.expander(f"📄 {result['file_info']['source_file']}", expanded=True):
            
            # エラーチェック
            if result['file_info'].get('error'):
                st.error(f"エラー: {result['file_info']['error']}")
                continue
            
            # 基本情報
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("抽出単語数", result['extraction_results'].get('total_words', 0))
            with col2:
                st.metric("ユニーク単語数", result['extraction_results'].get('unique_words', 0))
            with col3:
                st.metric("OCR信頼度", f"{result['file_info'].get('ocr_confidence', 0):.3f}")
            
            # 詳細統計
            if include_stats:
                st.markdown("**処理詳細:**")
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    st.write(f"- 処理ページ数: {result['file_info'].get('processed_pages', 0)}")
                    st.write(f"- 処理レベル: {result['file_info'].get('processing_level', 'N/A')}")
                with stats_col2:
                    st.write(f"- OCR試行回数: {result['extraction_results'].get('ocr_attempts', 0)}")
                    st.write(f"- 成功抽出数: {result['extraction_results'].get('successful_extractions', 0)}")
            
            # 英語文章表示
            if show_passages and result['content'].get('english_passages'):
                st.markdown("**📝 抽出された英語文章:**")
                for j, passage in enumerate(result['content']['english_passages'][:3]):  # 最初の3文章
                    with st.container():
                        st.markdown(f"**文章 {j+1}:**")
                        st.text_area(f"passage_{i}_{j}", passage, height=100, key=f"passage_{i}_{j}")
                
                if len(result['content']['english_passages']) > 3:
                    st.info(f"他に{len(result['content']['english_passages']) - 3}文章があります")
            
            # 抽出単語リスト表示
            if show_word_list and result['content'].get('extracted_words'):
                st.markdown("**📚 抽出された単語:**")
                words = result['content']['extracted_words']
                
                # 単語を行ごとに表示（20語ずつ）
                for k in range(0, min(len(words), 100), 20):
                    word_group = words[k:k+20]
                    st.write(", ".join(word_group))
                
                if len(words) > 100:
                    st.info(f"他に{len(words) - 100}語があります")
            
            # ダウンロードボタン
            st.markdown("**💾 ダウンロード:**")
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON形式でダウンロード
                json_data = json.dumps(result, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📄 JSON形式でダウンロード",
                    data=json_data,
                    file_name=f"{Path(result['file_info']['source_file']).stem}.json",
                    mime="application/json"
                )
            
            with col2:
                # テキスト形式でダウンロード
                if result['content'].get('english_passages'):
                    text_data = "\n\n".join(result['content']['english_passages'])
                    st.download_button(
                        label="📝 テキスト形式でダウンロード",
                        data=text_data,
                        file_name=f"{Path(result['file_info']['source_file']).stem}.txt",
                        mime="text/plain"
                    )
    
    # 全体結果の一括ダウンロード
    if len(results) > 1:
        st.markdown("---")
        st.markdown("### 🎯 一括ダウンロード")
        
        # 全結果をZIPファイルで提供
        zip_buffer = create_zip_download(results)
        st.download_button(
            label="📦 全結果をZIPでダウンロード",
            data=zip_buffer,
            file_name=f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip"
        )

def create_zip_download(results):
    """
    処理結果をZIPファイルとして作成
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for result in results:
            if result['file_info'].get('error'):
                continue
            
            file_stem = Path(result['file_info']['source_file']).stem
            
            # JSON形式で保存
            json_data = json.dumps(result, ensure_ascii=False, indent=2)
            zip_file.writestr(f"{file_stem}.json", json_data)
            
            # テキスト形式で保存
            if result['content'].get('english_passages'):
                text_data = "\n\n".join(result['content']['english_passages'])
                zip_file.writestr(f"{file_stem}.txt", text_data)
            
            # 単語リストを保存
            if result['content'].get('extracted_words'):
                words_data = "\n".join(result['content']['extracted_words'])
                zip_file.writestr(f"{file_stem}_words.txt", words_data)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# 処理実行ボタン
if uploaded_files:
    if st.button("🚀 OCR処理を開始", type="primary", use_container_width=True):
        process_files(uploaded_files, enhancement_level, show_progress, show_word_list, show_passages, include_stats)

# 処理結果表示エリア
if 'results' in st.session_state:
    display_results(st.session_state.results, show_word_list, show_passages, include_stats)

# フッター
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🔧 PDF OCR Processor | 高精度OCR処理とLLM校正による英語テキスト抽出</p>
    <p>📚 Target単語帳との照合分析も可能</p>
</div>
""", unsafe_allow_html=True)
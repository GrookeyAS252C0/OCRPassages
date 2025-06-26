import streamlit as st
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime
import io
import zipfile
import time

# Streamlit Cloud環境でのNLTKデータダウンロード
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
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
    index=2,  # aggressive をデフォルトに
    help="処理レベルが高いほど精度向上しますが、時間がかかります"
)

st.sidebar.markdown("### 処理レベル説明")
level_descriptions = {
    "light": "🟢 軽量処理 - 高速ですが基本的なOCR",
    "standard": "🟡 標準処理 - バランスの取れた精度と速度",
    "aggressive": "🔴 高精度処理 - 最高精度ですが時間がかかります"
}
st.sidebar.markdown(f"**{enhancement_level}**: {level_descriptions[enhancement_level]}")

st.sidebar.markdown("### 🤖 AI校正設定")
st.sidebar.info("""
**モデル**: GPT-4o-mini  
**機能**: 
- OCR結果の自動校正
- 日本語コンテンツ除去
- 純粋英語テキスト抽出
- 文法・スペル修正
""")

st.sidebar.markdown("### 📋 処理仕様")
st.sidebar.markdown("""
- **解像度**: 300 DPI変換
- **前処理**: 6種類の画像強化
- **OCR**: Tesseract + AI校正
- **出力**: JSON/TXT/ZIP形式
""")

# OpenAI API設定チェック
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
    os.environ['OPENAI_API_KEY'] = api_key
    
    # APIキー接続テスト
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # 簡単なAPI接続テスト
        client.models.list()
        st.sidebar.success("✅ OpenAI API接続確認済み")
    except Exception as e:
        st.sidebar.error("❌ OpenAI API接続エラー")
        st.error(f"""
        🚨 **OpenAI API接続エラー**
        
        エラー内容: {str(e)}
        
        **対処方法:**
        1. Streamlit CloudのSecretsでAPIキーを確認
        2. APIキーが正しい形式か確認
        3. OpenAIアカウントの残高を確認
        """)
        st.stop()
else:
    st.sidebar.error("❌ OpenAI API Key未設定")
    st.error("""
    🚨 **OpenAI API Keyが設定されていません**
    
    **設定方法:**
    1. Streamlit Cloud → Settings → Secrets
    2. 以下を追加:
    ```
    OPENAI_API_KEY = "your-api-key-here"
    ```
    3. アプリを再起動
    
    詳細は `SECRETS_SETUP.md` を参照してください。
    """)
    st.stop()

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
    
    Args:
        uploaded_files: アップロードされたPDFファイルのリスト
        enhancement_level: OCR処理レベル
        show_progress: 進捗表示フラグ
        show_word_list: 単語リスト表示フラグ（結果表示で使用）
        show_passages: 英語文章表示フラグ（結果表示で使用）
        include_stats: 詳細統計表示フラグ（結果表示で使用）
    """
    # 表示オプションをセッションステートに保存（結果表示時に使用）
    st.session_state.display_options = {
        'show_word_list': show_word_list,
        'show_passages': show_passages,
        'include_stats': include_stats
    }
    
    # デバッグ情報表示
    st.write("🔍 **デバッグ情報**")
    debug_container = st.empty()
    error_container = st.empty()
    
    def log_debug(message):
        debug_container.text(f"[DEBUG] {message}")
    
    def log_error(message, error=None):
        error_msg = f"[ERROR] {message}"
        if error:
            error_msg += f" - {str(error)}"
        error_container.error(error_msg)
        print(error_msg)  # サーバーログにも出力
    try:
        log_debug("PDFTextExtractorを初期化中...")
        # PDFTextExtractorを初期化
        extractor = PDFTextExtractor()
        log_debug("PDFTextExtractor初期化完了")
        
        results = []
        log_debug(f"処理対象ファイル数: {len(uploaded_files)}")
        
        # 全体プログレスバー
        if show_progress:
            st.markdown("### 📊 処理進捗")
            overall_progress = st.progress(0)
            overall_status = st.empty()
            
            # 各ファイルの詳細進捗用のコンテナ
            progress_container = st.container()
        
        for i, uploaded_file in enumerate(uploaded_files):
            log_debug(f"ファイル {i+1}/{len(uploaded_files)} 処理開始: {uploaded_file.name}")
            
            if show_progress:
                # 全体進捗更新
                overall_progress.progress(i / len(uploaded_files))
                overall_status.text(f"処理中: {i+1}/{len(uploaded_files)} - {uploaded_file.name}")
                
                # 個別ファイル進捗表示
                with progress_container:
                    file_expander = st.expander(f"📄 {uploaded_file.name} - 処理中...", expanded=True)
                    with file_expander:
                        file_progress = st.progress(0)
                        file_status = st.empty()
                        step_status = st.empty()
                        
                        # ステップ表示
                        file_status.text("🔄 PDFファイル読み込み中...")
                        file_progress.progress(0.1)
            
            try:
                log_debug(f"一時ファイル作成: {uploaded_file.name}")
                # 一時ファイルに保存
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    file_data = uploaded_file.read()
                    log_debug(f"ファイルサイズ: {len(file_data)} bytes")
                    temp_file.write(file_data)
                    temp_file_path = temp_file.name
                log_debug(f"一時ファイル保存完了: {temp_file_path}")
                
                if show_progress:
                    file_status.text("💾 ファイル保存完了")
                    file_progress.progress(0.2)
            
                if show_progress:
                    file_status.text("🔍 OCR処理開始...")
                    file_progress.progress(0.3)
                    step_status.text(f"処理レベル: {enhancement_level}")
                
                log_debug(f"OCR処理開始: {enhancement_level}")
                # OCR処理実行
                result = extractor.process_pdf(temp_file_path, enhancement_level)
                log_debug(f"OCR処理完了: {uploaded_file.name}")
                
                if show_progress:
                    file_status.text("🤖 AI校正処理中...")
                    file_progress.progress(0.7)
                
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
                
                if show_progress:
                    file_status.text("✅ 処理完了!")
                    file_progress.progress(1.0)
                    # トークン使用量表示
                    token_usage = result.get('token_usage', {})
                    input_tokens = token_usage.get('total_input_tokens', 0)
                    output_tokens = token_usage.get('total_output_tokens', 0)
                    cost_usd = token_usage.get('total_cost_usd', 0.0)
                    
                    step_status.text(f"抽出語数: {processed_result['extraction_results']['total_words']}, "
                                   f"信頼度: {processed_result['file_info']['ocr_confidence']:.3f}")
                    
                    if input_tokens > 0 or output_tokens > 0:
                        step_status.text(f"💰 トークン: {input_tokens + output_tokens:,} (${cost_usd:.4f})")
                    
                    # expanderのタイトルを更新
                    file_expander.empty()
                    with progress_container:
                        completed_expander = st.expander(
                            f"✅ {uploaded_file.name} - 完了 "
                            f"({processed_result['extraction_results']['total_words']}語抽出)", 
                            expanded=False
                        )
                        with completed_expander:
                            st.success(f"📊 処理結果: {processed_result['extraction_results']['total_words']}語, "
                                     f"信頼度: {processed_result['file_info']['ocr_confidence']:.3f}")
                
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                log_error(f"{uploaded_file.name}の処理中にエラー発生: {error_type}", e)
                
                # 詳細なエラー情報を取得
                import traceback
                error_traceback = traceback.format_exc()
                log_debug(f"エラートレースバック:\n{error_traceback}")
                
                st.error(f"❌ {uploaded_file.name}の処理中にエラーが発生しました: {error_message}")
                
                if show_progress:
                    file_status.text("❌ エラー発生")
                    file_progress.progress(1.0)
                    step_status.text(f"エラー: {error_type}")
                    
                    # expanderのタイトルを更新
                    file_expander.empty()
                    with progress_container:
                        error_expander = st.expander(f"❌ {uploaded_file.name} - エラー", expanded=True)
                        with error_expander:
                            st.error(f"🚨 エラー内容: {error_message}")
                            st.code(f"エラータイプ: {error_type}")
                
                results.append({
                    'file_info': {
                        'source_file': uploaded_file.name,
                        'error': error_message,
                        'error_type': error_type
                    },
                    'extraction_results': {},
                    'content': {}
                })
            
            except Exception as inner_e:
                log_error(f"ファイル {uploaded_file.name} の内部処理エラー", inner_e)
                # 内部エラーもresultsに追加
                results.append({
                    'file_info': {
                        'source_file': uploaded_file.name,
                        'error': f"内部処理エラー: {str(inner_e)}",
                        'error_type': type(inner_e).__name__
                    },
                    'extraction_results': {},
                    'content': {}
                })
            finally:
                # 一時ファイルを削除
                try:
                    if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        log_debug(f"一時ファイル削除: {temp_file_path}")
                except Exception as cleanup_error:
                    log_error(f"一時ファイル削除エラー: {temp_file_path}", cleanup_error)
        
        if show_progress:
            overall_progress.progress(1.0)
            overall_status.text("🎉 全ての処理が完了しました！")
            
            # 完了サマリー
            st.markdown("### 📈 処理完了サマリー")
            successful_count = len([r for r in results if not r['file_info'].get('error')])
            total_words = sum(r['extraction_results'].get('total_words', 0) for r in results)
            
            # トークン使用量合計計算
            total_input_tokens = sum(r.get('token_usage', {}).get('total_input_tokens', 0) for r in results)
            total_output_tokens = sum(r.get('token_usage', {}).get('total_output_tokens', 0) for r in results)
            total_cost = sum(r.get('token_usage', {}).get('total_cost_usd', 0.0) for r in results)
            total_api_calls = sum(r.get('token_usage', {}).get('api_calls', 0) for r in results)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("成功ファイル数", f"{successful_count}/{len(results)}")
            with col2:
                st.metric("総抽出語数", total_words)
            with col3:
                avg_confidence = sum(r['file_info'].get('ocr_confidence', 0) 
                                   for r in results if r['file_info'].get('ocr_confidence', 0) > 0)
                avg_confidence = avg_confidence / max(successful_count, 1)
                st.metric("平均信頼度", f"{avg_confidence:.3f}")
            with col4:
                st.metric("API料金", f"${total_cost:.4f}")
            
            # トークン使用詳細
            if total_input_tokens > 0 or total_output_tokens > 0:
                st.markdown("### 💰 OpenAI API使用量詳細")
                token_col1, token_col2, token_col3 = st.columns(3)
                with token_col1:
                    st.metric("入力トークン", f"{total_input_tokens:,}")
                with token_col2:
                    st.metric("出力トークン", f"{total_output_tokens:,}")
                with token_col3:
                    st.metric("API呼び出し回数", total_api_calls)
        
        # 結果をセッションステートに保存
        st.session_state.results = results
        
        # 成功メッセージ
        successful_files = [r for r in results if not r['file_info'].get('error')]
        failed_files = [r for r in results if r['file_info'].get('error')]
        
        log_debug(f"処理完了: 成功 {len(successful_files)}, 失敗 {len(failed_files)}")
        
        if successful_files:
            st.markdown('<div class="success-box">✅ 処理が完了しました！</div>', unsafe_allow_html=True)
        
        if failed_files:
            st.warning(f"⚠️ {len(failed_files)}個のファイルでエラーが発生しました")
        
    except Exception as e:
        import traceback
        main_error_traceback = traceback.format_exc()
        log_error("メイン処理でエラー発生", e)
        log_debug(f"メインエラートレースバック:\n{main_error_traceback}")
        
        st.markdown(f'<div class="error-box">❌ 処理中にエラーが発生しました: {str(e)}</div>', unsafe_allow_html=True)
        st.code(f"エラータイプ: {type(e).__name__}")
        
        # エラー詳細を展開可能な形で表示
        with st.expander("🔍 エラー詳細情報", expanded=False):
            st.code(main_error_traceback)

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
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("抽出単語数", result['extraction_results'].get('total_words', 0))
            with col2:
                st.metric("ユニーク単語数", result['extraction_results'].get('unique_words', 0))
            with col3:
                st.metric("OCR信頼度", f"{result['file_info'].get('ocr_confidence', 0):.3f}")
            with col4:
                token_usage = result.get('token_usage', {})
                cost = token_usage.get('total_cost_usd', 0.0)
                st.metric("API料金", f"${cost:.4f}" if cost > 0 else "なし")
            
            # 詳細統計
            if include_stats:
                st.markdown("**処理詳細:**")
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    st.write(f"- 処理ページ数: {result['file_info'].get('processed_pages', 0)}")
                    st.write(f"- 処理レベル: {result['file_info'].get('processing_level', 'N/A')}")
                    # トークン情報
                    token_usage = result.get('token_usage', {})
                    if token_usage.get('total_input_tokens', 0) > 0:
                        st.write(f"- 入力トークン: {token_usage.get('total_input_tokens', 0):,}")
                        st.write(f"- 出力トークン: {token_usage.get('total_output_tokens', 0):,}")
                with stats_col2:
                    st.write(f"- OCR試行回数: {result['extraction_results'].get('ocr_attempts', 0)}")
                    st.write(f"- 成功抽出数: {result['extraction_results'].get('successful_extractions', 0)}")
                    # API情報
                    if token_usage.get('api_calls', 0) > 0:
                        st.write(f"- API呼び出し: {token_usage.get('api_calls', 0)}回")
                        st.write(f"- 料金: ${token_usage.get('total_cost_usd', 0.0):.4f}")
            
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
    # 表示オプションをセッションステートから取得、デフォルト値を設定
    display_options = st.session_state.get('display_options', {
        'show_word_list': True,
        'show_passages': True,
        'include_stats': True
    })
    display_results(
        st.session_state.results, 
        display_options['show_word_list'],
        display_options['show_passages'],
        display_options['include_stats']
    )

# フッター
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🔧 PDF OCR Processor | 高精度OCR処理とLLM校正による英語テキスト抽出</p>
    <p>📚 Target単語帳との照合分析も可能</p>
</div>
""", unsafe_allow_html=True)
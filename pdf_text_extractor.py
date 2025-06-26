import os
import re
from pathlib import Path
from typing import List, Dict
import logging

import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from langdetect import detect, DetectorFactory
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import openai
from dotenv import load_dotenv
from tqdm import tqdm

# 言語検出の結果を一定にする
DetectorFactory.seed = 0

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFTextExtractor:
    def __init__(self, openai_api_key: str = None):
        """
        PDFから英語テキストを抽出するクラス
        
        Args:
            openai_api_key: OpenAI APIキー（校正用）
        """
        # 環境変数から設定を読み込み
        load_dotenv()
        
        if openai_api_key:
            openai.api_key = openai_api_key
        elif os.getenv('OPENAI_API_KEY'):
            openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # NLTK データのダウンロード
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            nltk.download('omw-1.4')
        
        self.english_stopwords = set(stopwords.words('english'))
        
    def preprocess_image(self, image: Image.Image, enhancement_level: str = "standard") -> List[Image.Image]:
        """
        OCR前の高度な画像前処理（複数バリエーション生成）
        
        Args:
            image: PIL Image オブジェクト
            enhancement_level: 処理レベル（"light", "standard", "aggressive"）
            
        Returns:
            処理済み画像のリスト
        """
        processed_images = []
        
        # 元画像をグレースケールに
        if image.mode != 'L':
            base_image = image.convert('L')
        else:
            base_image = image.copy()
        
        # OpenCV形式に変換
        cv_image = cv2.cvtColor(np.array(base_image), cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 1. 標準処理
        processed_images.append(self._standard_preprocessing(gray))
        
        if enhancement_level in ["standard", "aggressive"]:
            # 2. コントラスト強化版
            processed_images.append(self._contrast_enhanced_preprocessing(gray))
            
            # 3. ノイズ除去強化版
            processed_images.append(self._noise_reduction_preprocessing(gray))
            
            # 4. 解像度向上版
            processed_images.append(self._resolution_enhanced_preprocessing(gray))
        
        if enhancement_level == "aggressive":
            # 5. 形態学的処理版
            processed_images.append(self._morphological_preprocessing(gray))
            
            # 6. アダプティブ二値化版
            processed_images.append(self._adaptive_threshold_preprocessing(gray))
        
        return processed_images
    
    def _standard_preprocessing(self, gray_image) -> Image.Image:
        """標準前処理"""
        # ガウシアンブラーでノイズ除去
        blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
        
        # OTSU二値化
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return Image.fromarray(binary)
    
    def _contrast_enhanced_preprocessing(self, gray_image) -> Image.Image:
        """コントラスト強化前処理"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray_image)
        
        # ガンマ補正
        gamma = 1.2
        lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(enhanced, lookup_table)
        
        # 二値化
        _, binary = cv2.threshold(gamma_corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return Image.fromarray(binary)
    
    def _noise_reduction_preprocessing(self, gray_image) -> Image.Image:
        """ノイズ除去強化前処理"""
        # バイラテラルフィルタでエッジ保持しながらノイズ除去
        denoised = cv2.bilateralFilter(gray_image, 9, 75, 75)
        
        # メディアンフィルタで追加ノイズ除去
        denoised = cv2.medianBlur(denoised, 3)
        
        # シャープ化カーネル
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # 二値化
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return Image.fromarray(binary)
    
    def _resolution_enhanced_preprocessing(self, gray_image) -> Image.Image:
        """解像度向上前処理"""
        # 2倍アップサンプリング
        height, width = gray_image.shape
        upscaled = cv2.resize(gray_image, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        
        # ガウシアンブラー
        blurred = cv2.GaussianBlur(upscaled, (3, 3), 0)
        
        # 二値化
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return Image.fromarray(binary)
    
    def _morphological_preprocessing(self, gray_image) -> Image.Image:
        """形態学的処理前処理"""
        # 二値化
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # モルフォロジー演算でノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # オープニング（ノイズ除去）
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # クロージング（文字の穴埋め）
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return Image.fromarray(closed)
    
    def _adaptive_threshold_preprocessing(self, gray_image) -> Image.Image:
        """アダプティブ二値化前処理"""
        # ガウシアンブラー
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # アダプティブ二値化
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return Image.fromarray(adaptive)
    
    def extract_text_from_image(self, image: Image.Image, enhancement_level: str = "standard") -> str:
        """
        複数OCRエンジンとアンサンブルを使ってテキストを抽出
        
        Args:
            image: PIL Image オブジェクト
            enhancement_level: 前処理レベル
            
        Returns:
            抽出されたテキスト
        """
        try:
            # 複数の前処理済み画像を生成
            processed_images = self.preprocess_image(image, enhancement_level)
            
            all_results = []
            
            # 各前処理画像に対して複数のOCR設定で実行
            for i, processed_image in enumerate(processed_images):
                # Tesseract設定のバリエーション
                ocr_configs = [
                    r'--oem 3 --psm 6 -l eng',  # 標準
                    r'--oem 3 --psm 8 -l eng',  # 単語レベル
                    r'--oem 3 --psm 13 -l eng', # 生テキスト行
                    r'--oem 1 --psm 6 -l eng',  # LSTM OCR
                ]
                
                for j, config in enumerate(ocr_configs):
                    try:
                        text = pytesseract.image_to_string(processed_image, config=config)
                        if text.strip():
                            all_results.append({
                                'text': text.strip(),
                                'preprocessing': i,
                                'config': j,
                                'confidence': self._estimate_ocr_confidence(text)
                            })
                    except Exception as e:
                        logger.debug(f"OCR設定 {j} 失敗: {e}")
                        continue
            
            if not all_results:
                return ""
            
            # 最高信頼度の結果を選択
            best_result = max(all_results, key=lambda x: x['confidence'])
            return best_result['text']
        
        except Exception as e:
            logger.error(f"OCR処理エラー: {e}")
            return ""
    
    def _estimate_ocr_confidence(self, text: str) -> float:
        """
        OCR結果の信頼度を推定
        
        Args:
            text: OCR結果テキスト
            
        Returns:
            信頼度スコア (0-1)
        """
        if not text.strip():
            return 0.0
        
        score = 0.0
        
        # 英語文字の割合
        english_chars = sum(1 for c in text if c.isalpha() and c.isascii())
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        if total_chars > 0:
            score += (english_chars / total_chars) * 0.4
        
        # 辞書単語の割合（簡易）
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        if words:
            # 基本的な英単語パターンをチェック
            valid_words = sum(1 for word in words if len(word) >= 2 and word.isalpha())
            score += (valid_words / len(words)) * 0.3
        
        # 文字列の長さ（長いほど信頼性が高い傾向）
        length_score = min(len(text) / 100, 1.0) * 0.2
        score += length_score
        
        # 特殊文字や数字の適度な存在
        special_ratio = len(re.findall(r'[.,!?;:]', text)) / max(len(text), 1)
        if 0.01 <= special_ratio <= 0.1:
            score += 0.1
        
        return min(score, 1.0)
    
    def detect_text_regions(self, image: Image.Image) -> List[tuple]:
        """
        画像内のテキスト領域を検出
        
        Args:
            image: PIL Image オブジェクト
            
        Returns:
            テキスト領域の座標リスト [(x, y, w, h), ...]
        """
        try:
            # OpenCV形式に変換
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # テキスト検出のための前処理
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 輪郭検出
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # サイズフィルタリング（小さすぎる・大きすぎる領域を除外）
                if 10 <= w <= image.width * 0.8 and 8 <= h <= image.height * 0.5:
                    # アスペクト比チェック（極端に細い・太い領域を除外）
                    aspect_ratio = w / h
                    if 0.5 <= aspect_ratio <= 20:
                        text_regions.append((x, y, w, h))
            
            # 領域を上から下、左から右の順にソート
            text_regions.sort(key=lambda region: (region[1], region[0]))
            
            return text_regions
            
        except Exception as e:
            logger.debug(f"テキスト領域検出エラー: {e}")
            return []
    
    def is_english_text(self, text: str) -> bool:
        """
        テキストが英語かどうかを判定（強化版）
        
        Args:
            text: 判定対象のテキスト
            
        Returns:
            英語の場合True
        """
        if not text.strip():
            return False
        
        # 日本語文字が含まれている場合は明確に除外
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return False
        
        # 数式や記号のみの場合は除外
        if re.match(r'^[\d\s\+\-\*\/\=\(\)\[\]]+$', text.strip()):
            return False
        
        try:
            # 英語文字の割合をチェック
            english_chars = sum(c.isalpha() and c.isascii() for c in text)
            total_alpha_chars = sum(c.isalpha() for c in text)
            
            if total_alpha_chars == 0:
                return False
            
            english_ratio = english_chars / total_alpha_chars
            
            # 英語文字の割合が低い場合は除外
            if english_ratio < 0.7:
                return False
            
            # 英語らしい単語が含まれているかチェック
            words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
            if not words:
                return False
            
            # 一般的な英語パターンのチェック
            english_patterns = [
                r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b',
                r'\b(is|are|was|were|be|been|being|have|has|had)\b',
                r'\b(can|could|should|would|will|shall|may|might|must)\b',
                r'\b(this|that|these|those|here|there|where|when|what|who|how|why)\b'
            ]
            
            pattern_matches = sum(1 for pattern in english_patterns if re.search(pattern, text.lower()))
            
            # langdetectによる言語判定
            try:
                detected_lang = detect(text)
                is_detected_english = detected_lang == 'en'
            except:
                is_detected_english = False
            
            # 総合判定
            return (english_ratio >= 0.8 and len(words) >= 1) or (english_ratio >= 0.7 and pattern_matches >= 1) or is_detected_english
            
        except Exception as e:
            logger.debug(f"言語判定エラー: {e}")
            return False
    
    def extract_english_words(self, text: str) -> List[str]:
        """
        テキストから英単語を抽出
        
        Args:
            text: 処理対象のテキスト
            
        Returns:
            英単語のリスト
        """
        if not self.is_english_text(text):
            return []
        
        # 単語トークン化
        words = word_tokenize(text.lower())
        
        # 英単語のみフィルタリング
        english_words = []
        for word in words:
            # アルファベットのみ、2文字以上、ストップワード除外
            if (re.match(r'^[a-z]+$', word) and 
                len(word) >= 2 and 
                word not in self.english_stopwords):
                english_words.append(word)
        
        return english_words
    
    def correct_ocr_with_llm(self, ocr_text: str) -> str:
        """
        LLMを使ってOCR結果を校正（強化版）
        
        Args:
            ocr_text: OCR結果のテキスト
            
        Returns:
            校正されたテキスト
        """
        if not ocr_text.strip():
            return ocr_text
        
        # OpenAI APIキーがない場合はOCRテキストをそのまま返す
        if not hasattr(openai, 'api_key') or not openai.api_key:
            return ocr_text
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai.api_key)
            
            # より具体的で詳細な指示を含むプロンプト
            prompt = f"""You are an expert English text corrector specializing in fixing OCR recognition errors in academic texts.

TASK: Fix OCR errors in the following English text from a university entrance exam. 

SPECIFIC INSTRUCTIONS:
1. Correct common OCR errors like:
   - 'cnough' → 'enough'
   - 'bady' → 'body' 
   - 'rn' → 'm'
   - '1' → 'l' or 'I'
   - '0' → 'o' or 'O'
   - Fragmented words
   - Missing spaces between words
   - Extra spaces within words

2. COMPLETELY IGNORE and REMOVE:
   - Japanese text (hiragana, katakana, kanji)
   - Mathematical formulas and equations
   - Numbers used as question numbers or labels
   - Formatting artifacts

3. OUTPUT REQUIREMENTS:
   - Return ONLY the corrected English text
   - Maintain original sentence structure where possible
   - Ensure proper spacing and punctuation
   - If no valid English content exists, return empty string

OCR TEXT:
{ocr_text}

CORRECTED ENGLISH TEXT:"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.05,  # Lower temperature for more consistent results
                top_p=0.9
            )
            
            corrected_text = response.choices[0].message.content.strip()
            
            # 基本的な後処理
            corrected_text = self._post_process_corrected_text(corrected_text)
            
            return corrected_text
            
        except Exception as e:
            logger.warning(f"LLM校正エラー: {e}")
            return ocr_text
    
    def _post_process_corrected_text(self, text: str) -> str:
        """
        LLM校正後のテキストを後処理
        
        Args:
            text: 校正されたテキスト
            
        Returns:
            後処理済みテキスト
        """
        # 多重スペースを単一スペースに
        text = re.sub(r'\s+', ' ', text)
        
        # 行頭行末の余分な空白を除去
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # 明らかに英語でない行を除去
        english_lines = []
        for line in lines:
            # 基本的な英語パターンチェック
            if re.search(r'[a-zA-Z]', line) and len(line) > 1:
                # 日本語文字が含まれていないかチェック
                if not re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', line):
                    english_lines.append(line)
        
        return '\n'.join(english_lines)
    
    def extract_pure_english_only(self, ocr_text: str) -> str:
        """
        純粋な英語のみを抽出（日本語・脚注・問題番号等完全除去）
        
        Args:
            ocr_text: OCR結果のテキスト
            
        Returns:
            純粋な英語テキストのみ
        """
        if not ocr_text.strip():
            return ""
        
        # OpenAI APIキーがない場合は空文字を返す
        if not hasattr(openai, 'api_key') or not openai.api_key:
            return ""
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai.api_key)
            
            prompt = f"""You are an expert text processor specialized in extracting PURE ENGLISH TEXT ONLY from Japanese university entrance exam materials.

CRITICAL TASK: Extract ONLY pure English text. COMPLETELY IGNORE ALL JAPANESE content.

WHAT TO EXTRACT (ENGLISH ONLY):
- English reading passages
- English dialogue
- English essay text
- English story content
- English article text

WHAT TO COMPLETELY IGNORE AND REMOVE:
- ALL Japanese text (hiragana: あいう, katakana: アイウ, kanji: 漢字)
- ALL Japanese footnotes and annotations
- Question numbers (1), (2), A), B), etc.
- Instructions in Japanese
- Page numbers and headers
- Copyright notices like "© Obunsha"
- Author names in Japanese
- Publication information
- OCR artifacts and garbled text
- Mathematical formulas
- Isolated random letters or symbols
- Translation notes or vocabulary lists
- Answer choice indicators

TEXT PROCESSING:
- Remove ALL line breaks within paragraphs
- Join sentences into flowing paragraphs
- Remove extra spaces
- Fix OCR errors: 'cnough'→'enough', 'bady'→'body', 'rn'→'m'
- Keep proper punctuation and capitalization
- Maintain natural paragraph breaks between different passages

OUTPUT REQUIREMENTS:
- Return ONLY continuous English prose
- NO Japanese characters whatsoever
- NO question numbers or markers
- NO footnotes or annotations
- Separate distinct passages with double line breaks
- If no pure English content exists, return empty string

OCR INPUT:
{ocr_text}

PURE ENGLISH OUTPUT:"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500,
                temperature=0.02  # Very low temperature for consistency
            )
            
            pure_english = response.choices[0].message.content.strip()
            
            # 厳格な後処理で日本語を完全除去
            pure_english = self._strict_english_filter(pure_english)
            
            return pure_english
            
        except Exception as e:
            logger.error(f"純粋英語抽出エラー: {e}")
            logger.error(f"OCRテキストの最初の100文字: {ocr_text[:100]}")
            return ""
    
    def _strict_english_filter(self, text: str) -> str:
        """
        厳格な英語フィルタで日本語を完全除去
        
        Args:
            text: 処理対象テキスト
            
        Returns:
            純粋英語テキスト
        """
        if not text:
            return ""
        
        # 日本語文字を含む行を完全除去
        lines = text.split('\n')
        english_only_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 日本語文字チェック（ひらがな、カタカナ、漢字）
            if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', line):
                continue
                
            # 問題番号や記号のみの行を除去
            if re.match(r'^[\d\s\(\)\[\]A-Z\.]+$', line) and len(line) < 10:
                continue
                
            # 意味のある英語文字が含まれているかチェック
            english_chars = re.findall(r'[a-zA-Z]', line)
            if len(english_chars) < 3:  # 英字が3文字未満は除去
                continue
                
            # 明らかなOCRエラーテキストを除去
            if len(line) > 5 and not re.search(r'\s', line):  # スペースなしの長い文字列
                continue
                
            english_only_lines.append(line)
        
        # 段落の再構成
        result = self._reconstruct_paragraphs(english_only_lines)
        
        return result.strip()
    
    def _reconstruct_paragraphs(self, lines: List[str]) -> str:
        """
        英語行から自然な段落を再構成
        """
        if not lines:
            return ""
        
        paragraphs = []
        current_paragraph = ""
        
        for line in lines:
            # 新しい段落の開始を判定
            if (current_paragraph and 
                (line[0].isupper() and current_paragraph.endswith('.')) or
                line.startswith('In ') or line.startswith('The ') or line.startswith('When ') or
                line.startswith('Last ') or line.startswith('After ') or line.startswith('My ')):
                
                if current_paragraph:
                    paragraphs.append(current_paragraph)
                current_paragraph = line
            else:
                # 現在の段落に追加
                if current_paragraph:
                    current_paragraph += " " + line
                else:
                    current_paragraph = line
        
        # 最後の段落を追加
        if current_paragraph:
            paragraphs.append(current_paragraph)
        
        # 段落間は二重改行で区切る
        result = "\n\n".join(paragraphs)
        
        # 最終クリーンアップ
        result = re.sub(r' +', ' ', result)  # 多重スペース除去
        result = re.sub(r'\n{3,}', '\n\n', result)  # 3つ以上の改行を2つに
        
        return result
    
    def _post_process_clean_text(self, text: str) -> str:
        """
        抽出されたクリーンテキストの後処理
        
        Args:
            text: クリーンテキスト
            
        Returns:
            最終処理済みテキスト
        """
        if not text:
            return ""
        
        # 複数の改行を単一の改行に
        text = re.sub(r'\n+', '\n', text)
        
        # 行頭行末の空白を除去
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # 段落内の改行を除去し、文章を連続させる
        cleaned_lines = []
        current_paragraph = ""
        
        for line in lines:
            # 新しい段落の開始を判定（大文字で始まる、または前の行がピリオドで終わる）
            if (current_paragraph and 
                (line[0].isupper() and current_paragraph.endswith('.')) or
                len(current_paragraph) == 0):
                if current_paragraph:
                    cleaned_lines.append(current_paragraph)
                current_paragraph = line
            else:
                # 現在の段落に追加
                if current_paragraph:
                    current_paragraph += " " + line
                else:
                    current_paragraph = line
        
        # 最後の段落を追加
        if current_paragraph:
            cleaned_lines.append(current_paragraph)
        
        # 段落間は二重改行で区切る
        result = "\n\n".join(cleaned_lines)
        
        # 多重スペースを単一スペースに
        result = re.sub(r' +', ' ', result)
        
        return result.strip()
    
    def process_pdf(self, pdf_path: str, enhancement_level: str = "standard") -> Dict[str, any]:
        """
        単一PDFファイルから英語テキストを抽出（アンサンブル手法）
        
        Args:
            pdf_path: PDFファイルのパス
            enhancement_level: 処理レベル ("light", "standard", "aggressive")
            
        Returns:
            抽出結果の辞書
        """
        logger.info(f"処理開始: {pdf_path} (レベル: {enhancement_level})")
        
        result = {
            'source_file': Path(pdf_path).name,
            'full_path': pdf_path,
            'pages_processed': 0,
            'pure_english_text': [],
            'extracted_words': [],
            'processing_stats': {
                'total_ocr_attempts': 0,
                'successful_extractions': 0,
                'average_confidence': 0.0,
                'enhancement_level': enhancement_level
            },
            'error': None
        }
        
        try:
            # PDFを画像に変換
            images = convert_from_path(pdf_path, dpi=300)
            result['pages_processed'] = len(images)
            
            all_extracted_words = []
            confidence_scores = []
            
            for i, image in enumerate(tqdm(images, desc=f"Processing {Path(pdf_path).name}")):
                logger.info(f"ページ {i+1}/{len(images)} 処理中...")
                
                # テキスト領域検出を試行
                text_regions = self.detect_text_regions(image)
                
                if text_regions and enhancement_level == "aggressive":
                    # 領域ごとに処理
                    for region in text_regions:
                        x, y, w, h = region
                        region_image = image.crop((x, y, x+w, y+h))
                        
                        # 複数手法でOCR実行
                        ocr_text = self.extract_text_from_image(region_image, enhancement_level)
                        result['processing_stats']['total_ocr_attempts'] += 1
                        
                        if ocr_text.strip():
                            # 純粋な英語のみを抽出
                            pure_english = self.extract_pure_english_only(ocr_text)
                            if pure_english:
                                result['pure_english_text'].append(pure_english)
                                
                                # 英単語を抽出
                                words = self._extract_words_from_text(pure_english)
                                all_extracted_words.extend(words)
                            
                            result['processing_stats']['successful_extractions'] += 1
                            
                            # 信頼度スコア記録
                            confidence = self._estimate_ocr_confidence(ocr_text)
                            confidence_scores.append(confidence)
                else:
                    # 通常の全体画像処理
                    ocr_text = self.extract_text_from_image(image, enhancement_level)
                    result['processing_stats']['total_ocr_attempts'] += 1
                    
                    if ocr_text.strip():
                        # 純粋な英語のみを抽出
                        pure_english = self.extract_pure_english_only(ocr_text)
                        if pure_english:
                            result['pure_english_text'].append(pure_english)
                            
                            # 英単語を抽出
                            words = self._extract_words_from_text(pure_english)
                            all_extracted_words.extend(words)
                        
                        result['processing_stats']['successful_extractions'] += 1
                        
                        # 信頼度スコア記録
                        confidence = self._estimate_ocr_confidence(ocr_text)
                        confidence_scores.append(confidence)
            
            # 重複単語を除去（頻度も考慮）
            result['extracted_words'] = self._deduplicate_words_with_frequency(all_extracted_words)
            
            # 統計情報更新
            if confidence_scores:
                result['processing_stats']['average_confidence'] = sum(confidence_scores) / len(confidence_scores)
        
        except Exception as e:
            logger.error(f"PDF処理エラー: {pdf_path} - {e}")
            result['error'] = str(e)
        
        logger.info(f"処理完了: {pdf_path} - {len(result['extracted_words'])}単語抽出 (平均信頼度: {result['processing_stats']['average_confidence']:.3f})")
        return result
    
    def _extract_words_from_text(self, text: str) -> List[str]:
        """
        テキストから英語単語を抽出（内部ヘルパー）
        """
        words = []
        for line in text.split('\n'):
            line = line.strip()
            if line and self.is_english_text(line):
                line_words = self.extract_english_words(line)
                words.extend(line_words)
        return words
    
    def _deduplicate_words_with_frequency(self, words: List[str]) -> List[str]:
        """
        頻度を考慮した重複除去
        """
        from collections import Counter
        
        # 単語の頻度をカウント
        word_counts = Counter(words)
        
        # 頻度の高い単語を優先（同じ単語の異なるバリエーションを統合）
        cleaned_words = set()
        for word, count in word_counts.most_common():
            # 短縮形や類似単語のチェック
            should_add = True
            for existing_word in list(cleaned_words):
                # レーベンシュタイン距離での類似チェック（簡易版）
                if self._are_similar_words(word, existing_word):
                    if count > word_counts.get(existing_word, 0):
                        cleaned_words.discard(existing_word)
                        cleaned_words.add(word)
                    should_add = False
                    break
            
            if should_add:
                cleaned_words.add(word)
        
        return list(cleaned_words)
    
    def _are_similar_words(self, word1: str, word2: str) -> bool:
        """
        単語の類似性を判定（簡易版）
        """
        if abs(len(word1) - len(word2)) > 2:
            return False
        
        # 完全一致
        if word1 == word2:
            return True
        
        # 文字の80%以上が一致
        common_chars = sum(1 for i, c in enumerate(word1) if i < len(word2) and c == word2[i])
        similarity = common_chars / max(len(word1), len(word2))
        
        return similarity > 0.8
    
    def process_pdf_folder(self, folder_path: str, output_file: str = None, enhancement_level: str = "standard") -> List[Dict]:
        """
        フォルダ内の全PDFファイルを処理（強化版）
        
        Args:
            folder_path: PDFフォルダのパス
            output_file: 結果保存ファイル（オプション）
            enhancement_level: 処理レベル ("light", "standard", "aggressive")
            
        Returns:
            全PDFの処理結果リスト
        """
        pdf_folder = Path(folder_path)
        if not pdf_folder.exists():
            raise FileNotFoundError(f"フォルダが見つかりません: {folder_path}")
        
        pdf_files = list(pdf_folder.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"PDFファイルが見つかりません: {folder_path}")
            return []
        
        logger.info(f"{len(pdf_files)}個のPDFファイルを処理開始（処理レベル: {enhancement_level}）")
        
        results = []
        total_words = 0
        total_confidence = 0.0
        
        for pdf_file in pdf_files:
            result = self.process_pdf(str(pdf_file), enhancement_level)
            results.append(result)
            
            total_words += len(result['extracted_words'])
            if result['processing_stats']['average_confidence'] > 0:
                total_confidence += result['processing_stats']['average_confidence']
        
        # 全体統計の計算
        avg_confidence = total_confidence / len([r for r in results if r['processing_stats']['average_confidence'] > 0])
        
        logger.info(f"全体処理完了: {total_words}単語抽出、平均信頼度: {avg_confidence:.3f}")
        
        # 結果をファイルに保存（純粋英語版のみ）
        if output_file:
            # 純粋英語抽出用のJSONファイル名
            pure_english_file = "extraction_results_pure_english.json"
            
            # 純粋英語抽出用の統計情報
            pure_english_results = {
                'extraction_summary': {
                    'total_source_files': len(pdf_files),
                    'total_words_extracted': total_words,
                    'average_ocr_confidence': avg_confidence,
                    'processing_level': enhancement_level,
                    'extraction_method': 'pure_english_only',
                    'japanese_content': 'completely_ignored'
                },
                'extracted_data': []
            }
            
            # 各PDFのデータを整理
            for result in results:
                if result['pure_english_text'] or result['extracted_words']:
                    pure_data = {
                        'source_file': result['source_file'],
                        'english_passages': result['pure_english_text'],
                        'word_count': len(result['extracted_words']),
                        'extracted_words': result['extracted_words'],
                        'ocr_confidence': result['processing_stats']['average_confidence'],
                        'pages_processed': result['pages_processed']
                    }
                    pure_english_results['extracted_data'].append(pure_data)
            
            self.save_results(pure_english_results, pure_english_file)
            logger.info(f"純粋英語データを保存: {pure_english_file}")
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        """
        処理結果をファイルに保存
        
        Args:
            results: 処理結果のリスト
            output_file: 出力ファイルパス
        """
        import json
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"結果を保存しました: {output_file}")
    
    def save_individual_results(self, results: List[Dict], output_dir: str = "individual_results"):
        """
        各PDFの処理結果を個別のJSONファイルとして保存
        
        Args:
            results: 処理結果のリスト
            output_dir: 出力ディレクトリ
        """
        import json
        import os
        from pathlib import Path
        
        # 出力ディレクトリを作成
        os.makedirs(output_dir, exist_ok=True)
        
        for result in results:
            if 'extracted_data' in result:
                # 統合結果の場合
                for pdf_data in result['extracted_data']:
                    filename = self._get_safe_filename(pdf_data['source_file'])
                    output_path = os.path.join(output_dir, f"{filename}.json")
                    
                    # 個別ファイル用のデータ構造
                    individual_data = {
                        "file_info": {
                            "source_file": pdf_data['source_file'],
                            "processed_pages": pdf_data.get('pages_processed', 0),
                            "ocr_confidence": pdf_data.get('ocr_confidence', 0.0),
                            "processing_timestamp": self._get_timestamp()
                        },
                        "extraction_results": {
                            "total_words": pdf_data.get('word_count', 0),
                            "unique_words": len(set(pdf_data.get('extracted_words', []))),
                            "english_passages_count": len(pdf_data.get('english_passages', []))
                        },
                        "content": {
                            "english_passages": pdf_data.get('english_passages', []),
                            "extracted_words": sorted(list(set(pdf_data.get('extracted_words', []))))
                        }
                    }
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(individual_data, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"個別結果を保存: {output_path}")
            else:
                # 単一結果の場合
                filename = self._get_safe_filename(result['source_file'])
                output_path = os.path.join(output_dir, f"{filename}.json")
                
                individual_data = {
                    "file_info": {
                        "source_file": result['source_file'],
                        "processed_pages": result.get('pages_processed', 0),
                        "ocr_confidence": result.get('processing_stats', {}).get('average_confidence', 0.0),
                        "processing_timestamp": self._get_timestamp()
                    },
                    "extraction_results": {
                        "total_words": len(result.get('extracted_words', [])),
                        "unique_words": len(set(result.get('extracted_words', []))),
                        "english_passages_count": len(result.get('pure_english_text', []))
                    },
                    "content": {
                        "english_passages": result.get('pure_english_text', []),
                        "extracted_words": sorted(list(set(result.get('extracted_words', []))))
                    }
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(individual_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"個別結果を保存: {output_path}")
    
    def _get_safe_filename(self, filename: str) -> str:
        """
        ファイル名を安全な形式に変換
        
        Args:
            filename: 元のファイル名
            
        Returns:
            安全なファイル名
        """
        import re
        
        # 拡張子を除去
        from pathlib import Path
        name = Path(filename).stem
        
        # 日本語文字をローマ字に変換（簡易版）
        name = re.sub(r'[^\w\-_\.]', '_', name)
        
        # 連続するアンダースコアを単一に
        name = re.sub(r'_+', '_', name)
        
        # 先頭・末尾のアンダースコアを除去
        name = name.strip('_')
        
        return name if name else "unnamed"
    
    def _get_timestamp(self) -> str:
        """
        現在のタイムスタンプを取得
        
        Returns:
            ISO形式のタイムスタンプ
        """
        from datetime import datetime
        return datetime.now().isoformat()

if __name__ == "__main__":
    import argparse
    
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='PDF英語テキスト抽出ツール')
    parser.add_argument('--pdf-folder', default='./PDF', help='PDFファイルが格納されているフォルダパス')
    parser.add_argument('--enhancement-level', choices=['light', 'standard', 'aggressive'], 
                       default='aggressive', help='OCR処理レベル')
    parser.add_argument('--output-format', choices=['combined', 'individual', 'both'], 
                       default='combined', help='出力形式: combined=統合ファイル, individual=個別ファイル, both=両方')
    parser.add_argument('--output-dir', default='individual_results', 
                       help='個別ファイル出力時のディレクトリ名')
    
    args = parser.parse_args()
    
    # 使用例（強化版）
    extractor = PDFTextExtractor()
    
    # PDFフォルダを処理（強化レベルを選択可能）
    pdf_folder = args.pdf_folder
    enhancement_level = args.enhancement_level
    
    print(f"OCR精度向上機能を使用してPDF処理を開始...")
    print(f"処理レベル: {enhancement_level}")
    print(f"出力形式: {args.output_format}")
    print(f"主な改善点:")
    print(f"- 6種類の高度な画像前処理")
    print(f"- 4つのOCR設定による複数実行")
    print(f"- 信頼度スコアによる最適結果選択")
    print(f"- 強化されたLLM校正プロンプト")
    print(f"- テキスト領域検出による局所処理")
    print(f"- 日本語完全除外設定")
    print()
    
    results = extractor.process_pdf_folder(pdf_folder, "pure_english_output", enhancement_level)
    
    # 全体の統計を表示（純粋英語抽出版）
    if results:
        total_words = sum(len(result['extracted_words']) for result in results)
        all_words = set()
        confidence_scores = []
        
        for result in results:
            all_words.update(result['extracted_words'])
            if result['processing_stats']['average_confidence'] > 0:
                confidence_scores.append(result['processing_stats']['average_confidence'])
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        print(f"\n=== 純粋英語抽出結果 ===")
        print(f"処理PDFファイル数: {len(results)}")
        print(f"抽出単語総数: {total_words}")
        print(f"ユニーク単語数: {len(all_words)}")
        print(f"平均OCR信頼度: {avg_confidence:.3f}")
        print(f"処理レベル: {enhancement_level}")
        print(f"日本語除去: 完全")
        
        # 純粋英語文章の統計
        total_passages = sum(len(result.get('pure_english_text', [])) for result in results)
        print(f"抽出された純粋英語文章数: {total_passages}")
        
        # 保存ファイル情報
        print(f"\n=== 出力ファイル ===")
        print(f"📄 extraction_results_pure_english.json")
        
        # 各PDFの詳細結果
        print(f"\n=== 各PDFの詳細 ===")
        for result in results:
            pdf_name = result['source_file']
            stats = result['processing_stats']
            passage_count = len(result.get('pure_english_text', []))
            print(f"{pdf_name}: {len(result['extracted_words'])}単語, {passage_count}文章, 信頼度: {stats['average_confidence']:.3f}")
        
        # 出力形式に応じてファイル保存
        if args.output_format in ['individual', 'both']:
            print(f"\n=== 個別ファイル出力 ===")
            extractor.save_individual_results(results, args.output_dir)
            print(f"個別JSONファイルを {args.output_dir}/ に保存しました")
        
        # 純粋英語文章のサンプル表示
        print(f"\n=== 純粋英語文章サンプル ===")
        for result in results:
            passages = result.get('pure_english_text', [])
            if passages:
                print(f"\n【{result['source_file']}】")
                for i, passage in enumerate(passages[:2]):  # 最初の2つの文章を表示
                    print(f"文章{i+1}: {passage[:300]}{'...' if len(passage) > 300 else ''}")
                    print()
    else:
        print("処理するPDFファイルが見つかりませんでした。")
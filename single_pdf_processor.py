#!/usr/bin/env python3
"""
単一PDFファイル処理スクリプト
指定されたPDFファイルをOCR処理し、同じフォルダに同じファイル名でJSONファイルを出力します。

使用方法:
python single_pdf_processor.py path/to/file.pdf
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any
import logging
from pdf_text_extractor import PDFTextExtractor

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_pdf(pdf_path: str, enhancement_level: str = "standard") -> Dict[str, Any]:
    """
    単一PDFファイルを処理してJSONファイルを出力
    
    Args:
        pdf_path: PDFファイルのパス
        enhancement_level: OCR処理レベル ("light", "standard", "aggressive")
    
    Returns:
        処理結果の辞書
    """
    # パスの検証
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDFファイルが見つかりません: {pdf_path}")
    
    if not pdf_file.suffix.lower() == '.pdf':
        raise ValueError(f"PDFファイルではありません: {pdf_path}")
    
    logger.info(f"PDFファイル処理開始: {pdf_file.name}")
    
    # PDFTextExtractorを初期化
    extractor = PDFTextExtractor()
    
    # PDFを処理
    result = extractor.process_pdf(str(pdf_file), enhancement_level)
    
    # 出力用データを整理
    output_data = {
        "file_info": {
            "source_file": pdf_file.name,
            "full_path": str(pdf_file.absolute()),
            "processed_pages": result.get('pages_processed', 0),
            "ocr_confidence": result.get('processing_stats', {}).get('average_confidence', 0.0),
            "processing_level": enhancement_level,
            "processing_timestamp": _get_timestamp(),
            "error": result.get('error', None)
        },
        "extraction_results": {
            "total_words": len(result.get('extracted_words', [])),
            "unique_words": len(set(result.get('extracted_words', []))),
            "english_passages_count": len(result.get('pure_english_text', [])),
            "ocr_attempts": result.get('processing_stats', {}).get('total_ocr_attempts', 0),
            "successful_extractions": result.get('processing_stats', {}).get('successful_extractions', 0)
        },
        "content": {
            "english_passages": result.get('pure_english_text', []),
            "extracted_words": sorted(list(set(result.get('extracted_words', []))))
        }
    }
    
    # JSONファイルのパスを生成（PDFと同じフォルダ、同じファイル名）
    json_file = pdf_file.parent / f"{pdf_file.stem}.json"
    
    # JSONファイルに保存
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"処理結果を保存: {json_file}")
    logger.info(f"抽出結果: {output_data['extraction_results']['total_words']}単語, "
               f"{output_data['extraction_results']['english_passages_count']}文章, "
               f"信頼度: {output_data['file_info']['ocr_confidence']:.3f}")
    
    return output_data

def _get_timestamp() -> str:
    """
    現在のタイムスタンプを取得
    
    Returns:
        ISO形式のタイムスタンプ
    """
    from datetime import datetime
    return datetime.now().isoformat()

def main():
    """
    メイン関数
    """
    if len(sys.argv) < 2:
        print("使用方法: python single_pdf_processor.py <PDFファイルパス> [処理レベル]")
        print("処理レベル: light, standard, aggressive (デフォルト: standard)")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    enhancement_level = sys.argv[2] if len(sys.argv) > 2 else "standard"
    
    # 処理レベルの検証
    if enhancement_level not in ["light", "standard", "aggressive"]:
        print(f"無効な処理レベル: {enhancement_level}")
        print("有効な処理レベル: light, standard, aggressive")
        sys.exit(1)
    
    try:
        # PDFファイルを処理
        result = process_single_pdf(pdf_path, enhancement_level)
        
        print(f"\n=== 処理完了 ===")
        print(f"ファイル: {result['file_info']['source_file']}")
        print(f"処理ページ数: {result['file_info']['processed_pages']}")
        print(f"抽出単語数: {result['extraction_results']['total_words']}")
        print(f"ユニーク単語数: {result['extraction_results']['unique_words']}")
        print(f"英語文章数: {result['extraction_results']['english_passages_count']}")
        print(f"OCR信頼度: {result['file_info']['ocr_confidence']:.3f}")
        print(f"処理レベル: {result['file_info']['processing_level']}")
        
        # 出力ファイルのパス
        pdf_file = Path(pdf_path)
        json_file = pdf_file.parent / f"{pdf_file.stem}.json"
        print(f"出力ファイル: {json_file}")
        
        # エラーがあった場合は表示
        if result['file_info']['error']:
            print(f"エラー: {result['file_info']['error']}")
        
        # 抽出された英語文章のサンプル表示
        if result['content']['english_passages']:
            print(f"\n=== 抽出された英語文章サンプル ===")
            for i, passage in enumerate(result['content']['english_passages'][:2]):
                print(f"文章{i+1}: {passage[:200]}{'...' if len(passage) > 200 else ''}")
                print()
        
        # 抽出された単語のサンプル表示
        if result['content']['extracted_words']:
            print(f"=== 抽出された単語サンプル ===")
            sample_words = result['content']['extracted_words'][:20]
            print(f"単語例: {', '.join(sample_words)}")
            if len(result['content']['extracted_words']) > 20:
                print(f"... 他{len(result['content']['extracted_words']) - 20}語")
        
    except Exception as e:
        logger.error(f"処理エラー: {e}")
        print(f"エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
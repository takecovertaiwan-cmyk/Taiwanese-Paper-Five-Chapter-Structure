# ====================================================================
# [A] 台灣學術論文寫作 AI 服務 (API Backend)
# 作者: Gemini
# --------------------------------------------------------------------
# [A1] 核心功能:
# 1. 提供 /api/write-paper 路由，根據章節和提示詞生成論文內容。
# 2. 整合 Gemini API (gemini-2.5-flash-preview-09-2025) 進行生成。
# 3. 啟用 Google Search Grounding 以增強事實依據。
# --------------------------------------------------------------------
# [A2] 環境設定:
# - 使用 Flask 框架。
# - 支援 CORS 跨域請求。
# - 必須設定 GEMINI_API_KEY 環境變數。
# ====================================================================

# === B1. 套件匯入 (Imports) ===
import os
import json
import logging
import requests # 用於呼叫 Gemini API
import time # 用於指數退避
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, Any, List

# === B2. 讀取環境變數與 API 配置 ===
# 必須讀取 GEMINI_API_KEY 作為認證
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") 
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

# === B3. 系統初始化與配置 ===
# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
# 允許所有來源的跨域請求 (用於開發環境)
CORS(app) 

# === C. 核心工具與輔助函式 ===

# === C1. 呼叫 AI 函式 (call_gemini_api) ===
def call_gemini_api(system_instruction: str, user_prompt: str) -> str:
    """
    呼叫 Gemini API 進行內容生成，包含指數退避和引用來源提取。
    """
    if not GEMINI_API_KEY:
        logging.error("API Key 缺失。請設定 GEMINI_API_KEY 環境變數。")
        # 返回 JSON 格式的錯誤以便前端處理
        return json.dumps({"error": "API Key 缺失。請檢查後端配置。"})

    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        # 啟用 Google 搜尋工具
        "tools": [{"google_search": {}}], 
    }

    try:
        # C1-1. 指數退避重試邏輯
        max_retries = 3
        delay = 1  
        
        for attempt in range(max_retries):
            response = requests.post(
                API_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                result = response.json()
                candidate = result.get("candidates", [])[0]
                text = candidate.get("content", {}).get("parts", [])[0].get("text", "")
                
                # C1-2. 引用來源提取
                sources_info: List[str] = []
                grounding_metadata = candidate.get("groundingMetadata")
                if grounding_metadata and grounding_metadata.get("groundingAttributions"):
                    for i, attr in enumerate(grounding_metadata["groundingAttributions"]):
                        uri = attr.get("web", {}).get("uri", "")
                        title = attr.get("web", {}).get("title", f"來源 {i+1}")
                        if uri:
                            sources_info.append(f"[{title}]({uri})")
                
                if sources_info:
                    sources_text = "\n\n---\n**引用來源 (Grounding Sources):**\n" + "\n".join(sources_info)
                    text += sources_text

                return text

            elif response.status_code == 429:
                logging.warning(f"API 限制，嘗試第 {attempt + 1} 次重試，延遲 {delay} 秒...")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2  # 指數退避
                else:
                    raise Exception("達到最大重試次數，API 呼叫失敗。")
            else:
                logging.error(f"API 呼叫失敗，狀態碼: {response.status_code}, 回應: {response.text}")
                raise Exception(f"API 呼叫失敗，請檢查 API Key 或服務狀態。")

        return json.dumps({"error": "發生未知錯誤，無法獲取模型回應。"})

    except Exception as e:
        logging.error(f"AI 服務發生錯誤: {e}")
        return json.dumps({"error": f"AI 服務錯誤: {str(e)}"})

# === D. API 服務路由 ===

# === D1. 論文寫作路由 (/api/write-paper) ===
@app.route('/api/write-paper', methods=['POST'])
def write_paper():
    try:
        data = request.get_json()
        chapter = data.get('chapter')
        prompt_text = data.get('prompt_text')

        # D1-1. 輸入驗證
        if not chapter or not prompt_text:
            return jsonify({"error": "缺少 'chapter' 或 'prompt_text' 參數"}), 400

        # D1-2. 根據章節定義提示詞和指導
        chapter_mapping = {
            'introduction': ("緒論 (Introduction)", "目標是寫出包含研究背景、動機、目的和論文結構的段落。請使用台灣學術論文慣用的中文語氣和詞彙，並注意邏輯連貫性。"),
            'literature_review': ("文獻回顧 (Literature Review)", "目標是總結與研究主題相關的國內外重要文獻，並指出文獻的不足或研究缺口。請以嚴謹的學術風格撰寫。"),
            'methodology': ("研究方法 (Methodology)", "目標是清楚描述您的研究設計、對象、工具和資料分析方法，確保可重複性。使用清晰、客觀的語言。"),
            'results': ("結果與討論 (Results and Discussion)", "目標是呈現主要研究結果並解釋其意義，將結果與文獻回顧中的理論進行比較討論。請勿包含圖表，僅描述文字分析部分。"),
            'conclusion': ("結論與建議 (Conclusion and Suggestions)", "目標是簡要總結研究發現，重申貢獻，並提出未來研究方向或實務建議。保持簡潔有力。")
        }
        
        chapter_name, guidance = chapter_mapping.get(chapter, (chapter, "請根據台灣學術論文的規範，撰寫這部分內容。"))

        # D1-3. 組合系統指令 (System Instruction)
        system_prompt = (
            f"你是一位精通台灣學術論文寫作規範的 AI 助理。你的任務是根據用戶提供的關鍵資訊，"
            f"撰寫符合學術要求和中文慣例的 {chapter_name} 章節草稿。請以繁體中文和正式的學術語氣回覆，"
            f"並確保內容符合以下指導原則: {guidance}"
        )

        # D1-4. 組合用戶查詢 (User Query)
        user_query = f"請根據以下資訊和要求，撰寫 {chapter_name} 的內容：\n\n關鍵資訊：{prompt_text}"

        logging.info(f"開始為章節 {chapter_name} 呼叫 AI 服務...")

        # D1-5. 呼叫 AI 服務
        generated_text = call_gemini_api(system_prompt, user_query)

        # D1-6. 檢查並返回結果
        try:
            # 嘗試解析是否為錯誤 JSON (來自 call_gemini_api)
            error_check = json.loads(generated_text)
            if 'error' in error_check:
                return jsonify(error_check), 500
        except json.JSONDecodeError:
            pass # 正常情況，generated_text 是純文本

        return jsonify({"generated_text": generated_text})

    # D1-7. 路由級錯誤處理
    except Exception as e:
        logging.error(f"後端服務發生未預期的錯誤: {e}")
        return jsonify({"error": f"後端服務發生錯誤: {str(e)}"}), 500

# === E. 服務啟動 ===
if __name__ == '__main__':
    # E1. 本地啟動配置 (僅用於開發/本地測試)
    # ⚠️ 部署到 Render 時會使用 Gunicorn 等 WSGI 伺服器
    logging.info("Flask 服務啟動中 (本地測試)...")
    app.run(debug=True, port=5000)
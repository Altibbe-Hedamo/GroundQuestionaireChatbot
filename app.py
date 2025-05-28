from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import psycopg2
from psycopg2 import sql
import json
import os
import base64
import mimetypes
import tempfile
from docx import Document
from io import BytesIO
import speech_recognition as sr
from pydub import AudioSegment
from PIL import Image
import pytesseract
import fitz
import io

app = Flask(__name__)
CORS(app)

from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

API_KEY = os.getenv("OCR_SPACE_API_KEY")
API_URL = "https://api.ocr.space/parse/image"

DB_URL=os.getenv("DB_URL")
NEON_DB_URL=os.getenv("NEON_DB_URL")
NEON_DB_URL_FOR_FINAL_REPORT=os.getenv("NEON_DB_URL_FOR_FINAL_REPORT")

def get_db_connection():
    return psycopg2.connect(DB_URL)

def get_db_connection1():
    return psycopg2.connect(NEON_DB_URL)

def get_db_connection2():
    return psycopg2.connect(NEON_DB_URL_FOR_FINAL_REPORT)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            history TEXT,
            answers TEXT,
            completed BOOLEAN,
            attempts TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()

def get_session(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT history, answers, completed, attempts FROM user_sessions WHERE session_id = %s", (session_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        history = json.loads(row[0])
        answers = json.loads(row[1])
        completed = row[2]
        attempts = json.loads(row[3]) if row[3] else {}
    else:
        history = []
        answers = []
        completed = False
        attempts = {}
        save_session(session_id, history, answers, completed, attempts)

    return history, answers, completed, attempts


def save_session(session_id, history, answers, completed, attempts):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_sessions (session_id, history, answers, completed, attempts)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (session_id)
        DO UPDATE SET 
            history = EXCLUDED.history, 
            answers = EXCLUDED.answers, 
            completed = EXCLUDED.completed,
            attempts = EXCLUDED.attempts
    ''', (
        session_id,
        json.dumps(history),
        json.dumps(answers),
        completed,
        json.dumps(attempts)
    ))
    conn.commit()
    conn.close()


def query_data(conn, table, subcategory):
    cursor = conn.cursor()
    try:
        cursor.execute(
            sql.SQL("SELECT content FROM {} WHERE subcategory_name = %s;")
            .format(sql.Identifier(table)),
            (subcategory,)
        )
        results = cursor.fetchall()
        if results:
            return results
        else:
            return []
    except Exception as e:
         return jsonify({
             "error":f"Error while getting data{e}"
         })
    finally:
        cursor.close()

def insert_product(product_name, content):
    conn = get_db_connection2()
    if conn is None:
        return "Database connection error."

    try:
        cursor = conn.cursor()
        query = """
        INSERT INTO product_info (product_name, content)
        VALUES (%s, %s)
        ON CONFLICT (product_name) DO UPDATE
        SET content = product_info.content || '\n' || EXCLUDED.content
        """
        cursor.execute(query, (product_name, content))
        conn.commit()

        cursor.close()
        conn.close()

        return "Product content updated successfully."
    except Exception as e:
        return f"Error: {e}"

def get_product_content(product_name):
    conn = get_db_connection2()
    if conn is None:
        return "Database connection error."

    try:
        cursor = conn.cursor()
        query = "SELECT content FROM product_info WHERE product_name = %s"
        cursor.execute(query, (product_name,))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        return result[0] if result else "Product not found."
    except Exception as e:
        return f"Error: {e}"

def get_mime_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"

def ocr_space_file(image_bytes, language="eng", filetype="JPG"):
    """
    Sends an image file (in bytes) to the OCR API and returns the extracted text.
    """
    payload = {
        "apikey": API_KEY,
        "language": language,
        "isOverlayRequired": False,
        "filetype": filetype
    }
    try:
        response = requests.post(
            API_URL,
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            data=payload,
        )
        result = response.json()

        print("OCR API Response:", result)  

        if result.get("IsErroredOnProcessing", False):
            return {"error": result.get("ErrorMessage", "Unknown error")}

        if "ParsedResults" not in result or not result["ParsedResults"]:
            return {"error": "No text extracted or invalid response format"}

        return {"text": result["ParsedResults"][0].get("ParsedText", "")}

    except Exception as e:
        return {"error": str(e)}


def extract_text(file_path,language,use_openocr=True):
    """
    Extracts text from a given file:
    - PDF: Converts pages to images using PyMuPDF, then uses OpenOCR or Tesseract.
    - DOCX: Uses `python-docx` to extract text.
    - Images: Uses OpenOCR or Tesseract OCR.
    - TXT: Reads the plain text.
    """
    text = ""
    try:
        if file_path.endswith(".pdf"):
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                pix = page.get_pixmap()
                img_byte_arr = io.BytesIO()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img.save(img_byte_arr, format="JPEG", quality=80)
                img_bytes = img_byte_arr.getvalue()

                extracted_result = (
                    ocr_space_file(img_bytes, language, filetype="JPG") if use_openocr
                    else {"text": pytesseract.image_to_string(img)}
                )

                if "error" in extracted_result:
                    return extracted_result 

                extracted_text = extracted_result["text"]
                print(extracted_text)
                if extracted_text:
                    text += f"Page {page_num + 1}:\n{extracted_text}\n\n"

        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])

        elif file_path.endswith((".png", ".jpg", ".jpeg")):
            with open(file_path, "rb") as img_file:
                img_bytes = img_file.read()
                extracted_result = (
                    ocr_space_file(img_bytes,language,filetype="JPG") if use_openocr
                    else {"text": pytesseract.image_to_string(Image.open(file_path))}
                )

                if "error" in extracted_result:
                    return extracted_result  

                text = extracted_result["text"]

        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

    except Exception as e:
        return {"error": str(e)}

    return text.strip()


@app.route('/chats', methods=['POST'])
def chat():
    contents = []
    session_id = None
    user_response = None
    temp_file_path = None
    report=""
    category=""
    sub_categories=[]
    product_name=""

    aspects = [
        "Supplier and traceability documentation",
        "GMOs and contamination risk analysis",
        "In-depth farming or livestock management practices",
        "Soil, pest, biodiversity, animal welfare checks",
        "Harvesting practices and post-harvest handling",
        "Review of manufacturing practices",
        "Cross-contamination prevention methods",
        "Review of all additives and processing aids",
        "Full packaging and label compliance",
        "Worker welfare review",
        "Environmental impact and sustainability assessment"
    ]
    
    try:
        if request.content_type.startswith('multipart/form-data'):
            session_id = request.form.get("session_id")
            user_response = request.form.get("user_response")
            file = request.files.get("file")
            report=request.form.get("report","")
            category=request.form.get("category","")
            sub_categories_str = request.form.get("sub_categories", "[]")
            
            try:
               sub_categories = json.loads(sub_categories_str)
               if not isinstance(sub_categories, list):
                   sub_categories = []
            except json.JSONDecodeError:
               sub_categories = []
               
            if file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as temp_file:
                    file.save(temp_file.name)
                    temp_file_path = temp_file.name

                mime = get_mime_type(temp_file_path)

                if mime in ['application/pdf', 'image/jpeg', 'image/png', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                    with open(temp_file_path, "rb") as f:
                        file_bytes = f.read()

                    extracted_result = extract_text(temp_file_path, language="eng", use_openocr=True)
                    print("Extracted text from File:")
                    print(extracted_result)

                    if isinstance(extracted_result, dict) and "error" in extracted_result:
                        os.unlink(temp_file_path)
                        return jsonify({"error": f"File processing error: {extracted_result['error']}"}), 400

                    if extracted_result.strip():
                        user_response = extracted_result.strip()

                    contents.append({
                        "inline_data": {
                            "mime_type": mime,
                            "data": base64.b64encode(file_bytes).decode('utf-8')
                        }
                    })

                elif mime.startswith("audio/"):
                    try:
                        audio = AudioSegment.from_file(temp_file_path)
                        wav_path = temp_file_path + ".wav"
                        audio.export(wav_path, format="wav")

                        recognizer = sr.Recognizer()
                        with sr.AudioFile(wav_path) as source:
                            audio_data = recognizer.record(source)

                        user_response = recognizer.recognize_google(audio_data)
                        print("Transcribed text:", user_response)

                        os.unlink(wav_path)
                    except sr.UnknownValueError:
                        os.unlink(temp_file_path)
                        return jsonify({"error": "Could not understand audio."}), 400
                    except sr.RequestError as e:
                        os.unlink(temp_file_path)
                        return jsonify({"error": f"Speech recognition service error: {e}"}), 500
                    except Exception as e:
                        os.unlink(temp_file_path)
                        return jsonify({"error": f"Audio transcription failed: {str(e)}"}), 400
                else:
                    os.unlink(temp_file_path)
                    return jsonify({"error": f"Unsupported MIME type: {mime}. Supported types: PDF, JPEG, PNG, DOCX, TXT, AUDIO."}), 400

        else:
            data = request.get_json()
            session_id = data.get("session_id")
            user_response = data.get("user_response")
            report = data.get("report","")
            category = data.get("category","")
            sub_categories = data.get("sub_categories", [])
            product_name= data.get("product_name","")

        print("product_name::->",product_name)
        print("report:",report)
        print("category",category)
        print("sub_categories",sub_categories)

        history, answers, completed, attempts = get_session(session_id)
                
        if attempts is None:
            attempts = {}

        current_aspect_index = attempts.get("current_aspect_index", 0)
        current_question_index = attempts.get("current_question_index", 0)

        if not history:
            if product_name:
                history.append(product_name)
            
            if report:
               history.append(f"[Initial Report]\n{report}")

            if category:
               history.append(f"[Category Chosen]\n{category}")

            if sub_categories:
               conn = get_db_connection1()
               subcat_content = ""
               for subcat in sub_categories:
                  subcat_contents = query_data(conn, category, subcat)
                  for content in subcat_contents:
                      subcat_content += content[0]  
               history.append(f"[Subcategory Content: {subcat_content}]\n")
            initial_question = f'Aspect: "{aspects[0]}"\n\nPlease provide the information regarding "{aspects[0]}"'
            
            history.append(initial_question)
            save_session(session_id, history, answers, completed, attempts)
            return jsonify({
                "message": initial_question,
                "completed": False,
                "question_number": 1
            })
        
        if user_response or contents:
            last_question = history[-1]
            question_number = len(answers) + 1
            question_key = f"question_{question_number}"

            history_text = '\n'.join(history)

            validation_prompt = (
                f"You are an expert AI assistant to validate the User Responses. A user was asked the following question:\n"
                f"Q: {last_question}\n\n"
                f"They responded with:\nA: {user_response or '[File Uploaded]'}\n\n"
                f"Conversation history so far:\n{history_text}\n\n"
                f"Your task is to validate the user's response based on the following criteria:\n"
                f"1. The answer must be correct, complete, and factually accurate.\n"
                f"2. The answer must not contradict any previous responses in the conversation.\n"
                f"3. Do not mark responses as incorrect for generic or straightforward answers like product name, company name, brand, or place of production—unless they clearly contradict earlier responses or are obviously wrong.\n\n"
                f"Output 'Correct' if the response is accurate and meets all the above criteria.\n"
                f"Otherwise, output 'Incorrect' followed by a brief explanation of why the response is incorrect.\n"
                f"Do not include any explanation if the response is correct."
            )

            validation_result = model.generate_content(validation_prompt).text.strip().lower()

            if validation_result == "correct":
                answers.append(user_response or "[File Uploaded]")
                contents.insert(0, {"text": f"Q: {last_question}\nA: {user_response}"})
                history.append(f"A: {user_response}")
                attempts.pop(question_key, None)

                if current_question_index >= 4:
                    current_aspect_index += 1
                    current_question_index = 0
                else:
                    current_question_index += 1

                attempts["current_aspect_index"] = current_aspect_index
                attempts["current_question_index"] = current_question_index

                save_session(session_id, history, answers, completed, attempts)

            else:
                attempts[question_key] = attempts.get(question_key, 0) + 1
                if attempts[question_key] >= 3:
                    answers.append(user_response or "[File Uploaded]")
                    contents.insert(0, {"text": f"Q: {last_question}\nA: {user_response}"})
                    history.append(f"A: {user_response}")

                    if current_question_index >= 4:
                        current_aspect_index += 1
                        current_question_index = 0
                    else:
                        current_question_index += 1

                    attempts["current_aspect_index"] = current_aspect_index
                    attempts["current_question_index"] = current_question_index

                    save_session(session_id, history, answers, completed, attempts)

                else:
                    save_session(session_id, history, answers, completed, attempts)
                    return jsonify({
                        "message": f"{validation_result}. Attempt {attempts[question_key]} of 3. Please try again.\n\n{last_question}",
                        "completed": False,
                        "question_number": len(history)
                    })

        if current_aspect_index >= len(aspects):
            completed = True
            report_prompt = "Generate a report summarizing the following questions and answers and instructions:\n"
            with open('prompt.txt', 'r', encoding='utf-8') as file:
                content = file.read()
            report_prompt += content
            
            content_to_append_for_final_report=""
            product_name=history[0]
            
            for i, a in enumerate(history): 
                report_prompt += f"{i+1}:{a}\n"
                content_to_append_for_final_report+= f"{i+1} : {a}"

            insert_product(product_name, content_to_append_for_final_report)
            
            contents.insert(0, {"text": report_prompt})
            response = model.generate_content(contents)
            report = response.text.strip()

            insert_product(product_name,report)

            content_from_db=get_product_content(product_name)
            print("Content Saved to DB for final report::::->>>>",content_from_db)

            save_session(session_id, history, answers, completed, attempts)
            return jsonify({
                "message": "Thank you! All questions answered.",
                "completed": True,
                "answers": answers,
                "report": report
            })

        current_aspect = aspects[current_aspect_index]
        qa_pairs = []
        for i in range(len(history) - 2, -1, -1):
            if f'Aspect: "{current_aspect}"' in history[i]:
                question = history[i].split("\n", 1)[-1]
                answer = history[i+1][3:] if i+1 < len(history) else ""
                qa_pairs.insert(0, f"Q: {question}\nA: {answer}")
                if len(qa_pairs) >= current_question_index:
                    break
                 
        history_block = "\n\n".join(qa_pairs)
        if len(history) > 1:
           history_block += history[1]
        if len(history) > 2:
           history_block += history[2]
        if len(history) > 3:
           history_block += history[3]


        prompt = (
            f'You are an AI interviewer assessing the aspect: "{current_aspect}".\n\n'
            f'So far, the following Q&A have occurred:\n\n'
            f'{history_block if history_block else "No previous questions yet."}\n\n'
            f'Generate the next relevant and detailed question for this aspect (question {current_question_index + 1} of 5).'
        )

        print("prompt:",prompt)

        contents.insert(0, {"text": prompt})
        model_response = model.generate_content(contents)
        next_question = f'Aspect: "{current_aspect}"\n\n{model_response.text.strip()}'

        history.append(next_question)

        save_session(session_id, history, answers, completed, attempts)

        return jsonify({
            "message": next_question,
            "completed": False,
            "question_number": len(history)
        })

    except Exception as e:
        print("Error in /chats:", e)
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Failed to delete temp file: {e}")

if __name__ == '__main__':
    init_db()
    app.run(debug=True)

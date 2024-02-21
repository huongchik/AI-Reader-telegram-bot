import telebot
import json
import io
import docx
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import requests
import html2text
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import openai
import re
import docx
from langchain.schema import HumanMessage, AIMessage
import pandas as pd
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import re
import edge_tts
import io
import tempfile
import os

import os
import soundfile as sf
import numpy as np

model = "gpt-3.5-turbo-16k"
user_contexts = {}
user_chat_histories = {}
user_markdowns = {}
user_first_stages = {}
send_voice_messages = {}

with open("config.json") as json_file:
    data = json.load(json_file)
    bot_token = data["bot_token"]
    api_key = data["api_key"]
bot = telebot.TeleBot(bot_token)
openai.api_key = api_key


def initialize(user_id):
    user_contexts[user_id] = None
    user_chat_histories[user_id] = []
    user_first_stages[user_id] = True
    user_markdowns[user_id] = ""
    send_voice_messages[user_id] = False


@bot.message_handler(commands=["start"])
def start(message):
    user_id = message.from_user.id
    initialize(user_id)

    bot.send_message(
        user_id,
        "Привет, я ваш помощник! Я умею анализировать документы и читать отправленные мне ссылки",
        reply_markup=menu(),
    )
    bot.send_message(
        user_id,
        "Чтобы начать, отправьте мне, пожалуйста, ссылки или пропустите этот этап, нажав на кнопку или сразу приступив к загрузке документов 😊",
        timeout=30,
        reply_markup=skip(),
    )


@bot.message_handler(func=lambda message: message.text == "Новый диалог")
def new_dialog(message):
    start(message)


def menu():
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    button1 = telebot.types.KeyboardButton("Новый диалог")
    button2 = telebot.types.KeyboardButton(
        "Отвечать голосовыми сообщениями\n(вкл/выкл)"
    )
    markup.add(button1, button2, row_width=1)
    return markup


@bot.message_handler(
    func=lambda message: message.text == "Отвечать голосовыми сообщениями\n(вкл/выкл)"
)
def toggle_voice_mode(message):
    user_id = message.from_user.id
    if send_voice_messages[user_id] == True:
        bot.send_message(user_id, "Теперь я не буду присылать голосовые 🙊")
        send_voice_messages[user_id] = False
    else:
        bot.send_message(user_id, "Отлично! Теперь я буду присылать голосовые 👅")
        send_voice_messages[user_id] = True


def skip():
    keyboard = telebot.types.InlineKeyboardMarkup()
    button = telebot.types.InlineKeyboardButton("Пропустить этап", callback_data="skip")
    keyboard.add(button)
    return keyboard


@bot.message_handler(content_types=["voice"])
def handle_voice_message(message):
    user_id = message.from_user.id
    if user_id in user_first_stages:
        if user_first_stages[user_id] == False:
            voice = message.voice
            file_id = voice.file_id
            file_info = bot.get_file(file_id)
            file_path = file_info.file_path
            downloaded_file = bot.download_file(file_path)
            output_file = f"{user_id}.ogg"
            with open(output_file, "wb") as file:
                file.write(downloaded_file)
            audio_file = open(output_file, "rb")
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            audio_file.close()
            text = transcript.get("text")
            handle_message(message, user_query=text, output_file=output_file)

            if output_file is not None:
                os.remove(output_file)

    else:
        bot.send_message(user_id, "Пожалуйста, начните диалог с помощью команды /start")


def extract_from_xlsx(path):
    df = pd.read_excel(path)
    markdown = df.to_markdown(index=False)
    return markdown


def extract_from_docx(path):
    doc = docx.Document(path)
    all_data = []
    for element in doc.element.body:
        if (
            element.tag
            == "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tbl"
        ):
            table_data = []
            for row in element.iter(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr"
            ):
                row_data = []
                for cell in row.iter(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc"
                ):
                    cell_text = "".join(
                        t.text
                        for t in cell.iter(
                            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
                        )
                        if t.text
                    )
                    row_data.append(cell_text)
                table_data.append(row_data)
            markdown_table = (
                "|".join(table_data[0])
                + "\n"
                + "|".join(["----"] * len(table_data[0]))
                + "\n"
                + "\n".join(["|".join(row) for row in table_data[1:]])
            )
            all_data.append(markdown_table)
        elif (
            element.tag
            == "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
        ):
            paragraph_text = "".join(
                t.text
                for t in element.iter(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
                )
                if t.text
            )
            all_data.append(paragraph_text)
    text = "\n".join(all_data)
    return text


@bot.callback_query_handler(func=lambda call: call.data == "skip")
def skip_callback(call):
    user_id = call.from_user.id
    if user_id in user_first_stages and user_first_stages[user_id]:
        user_first_stages[user_id] = False
        bot.send_message(
            user_id, "Отправьте Ваши документы (.docx, .txt, .pdf) в сообщении"
        )
    else:
        bot.send_message(user_id, "Готов отвечать на ваши вопросы!")


def extract_text_from_file(message, document, downloaded_file):
    file_name = document.file_name
    # Обрабатываем файл
    if file_name.endswith(".txt"):
        return downloaded_file.decode("utf-8")

    elif file_name.endswith(".docx"):
        return extract_from_docx(io.BytesIO(downloaded_file))

    elif file_name.endswith(".xlsx"):
        return extract_from_xlsx(io.BytesIO(downloaded_file))

    elif file_name.endswith(".pdf"):
        pdf_reader = PdfReader(io.BytesIO(downloaded_file))
        content = ""
        for page in pdf_reader.pages:
            content += page.extract_text()
        return content
    else:
        user_id = message.from_user.id
        bot.send_message(
            user_id,
            f"Невозможно прочитать содержимое файла {file_name}. Поддерживаются только текстовые файлы .txt, .docx и .pdf.",
        )
        return None


def create_index_from_text(markdown):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text=markdown)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_texts(chunks, embedding=embeddings)
    return db


def get_context(index):
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model=model,
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    template = """
    CONTEXT: {context}
    You are a helpful assistant, above is some context, 
    Please answer the question, and make sure you follow ALL of the rules below:
    1. Answer the questions only based on context provided, do not make things up
    2. Answer questions in a helpful manner that straight to the point, with clear structure & all relevant information that might help users answer the question
    3. Anwser should be formatted in Markdown
    4. If there are relevant images, video, links, they are very important reference data, please include them as part of the answer

    QUESTION: {question}
    ANSWER (formatted in Markdown):
    """
    prompt = ChatPromptTemplate.from_template(template)
    context = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=index.as_retriever(),
        memory=memory,
        # condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs=dict(prompt=prompt),
    )

    return context


import asyncio

import edge_tts


def detect_language(text):
    if any("а" <= char <= "я" or "А" <= char <= "Я" for char in text):
        return "ru"  # Текст на кириллице
    else:
        return "en"


async def generate_speech(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data


async def generate_voice(text, user_id) -> None:
    if send_voice_messages[user_id] == True:
        bot.send_chat_action(user_id, "record_voice")
        output_file = f"{user_id}.mp3"
        # voices = {"en": "en-US-RogerNeural", "ru": "ru-RU-DmitryNeural"}
        voices = {"en": "en-US-AriaNeural", "ru": "ru-RU-SvetlanaNeural"}

        words_and_symbols = re.findall(
            r"[a-zA-Z]+|[а-яА-ЯёЁ]+|[^a-zA-Zа-яА-ЯёЁ\s]", text
        )
        current_language = None
        segments = []

        for item in words_and_symbols:
            if re.match(r"^[a-zA-Z]+$", item):
                language = "en"
            elif re.match(r"^[а-яА-ЯёЁ]+$", item):
                language = "ru"
            else:
                language = current_language if current_language is not None else "ru"

            if language != current_language:
                current_language = language
                segments.append((current_language, []))

            segments[-1][1].append(item)

        tasks = [
            generate_speech(" ".join(segment[1]), voices[segment[0]])
            for segment in segments
        ]
        audio_data_list = await asyncio.gather(*tasks)

        combined_audio = None
        for audio_data in audio_data_list:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_name = temp_file.name
            audio_segment, sample_rate = sf.read(temp_file_name)
            if combined_audio is None:
                combined_audio = audio_segment
            else:
                combined_audio = np.concatenate((combined_audio, audio_segment), axis=0)
            os.remove(temp_file_name)  # Удаляем временный файл после его использования

        sf.write(output_file, combined_audio, sample_rate)
        bot.send_voice(user_id, voice=open(output_file, "rb"))
        os.remove(output_file)



def dialog_questions(query, message, context, chat_history):
    user_id = message.from_user.id
    bot.send_chat_action(user_id, action="typing")

    response = context({"question": query})
    bot.send_chat_action(user_id, action="typing")
    ai_response = response["chat_history"][-1].content
    chat_history.append(HumanMessage(content=query))

    bot.send_message(user_id, ai_response, parse_mode="Markdown")
    asyncio.run(generate_voice(ai_response, user_id))

    chat_history.append(AIMessage(content=ai_response))


@bot.message_handler(content_types=["document"])
def handle_document(message):
    print("Handle docs...")

    # Получаем информацию о документе
    document = message.document
    file_id = document.file_id
    user_id = message.from_user.id
    bot.send_chat_action(user_id, action="upload_document")
    # Скачиваем файл
    file_info = bot.get_file(file_id)
    file_path = file_info.file_path
    downloaded_file = bot.download_file(file_path)

    # Если у пользователя еще нет базы знаний, создаем новую
    if user_id not in user_markdowns:
        user_markdowns[user_id] = ""
    bot.send_chat_action(user_id, action="upload_document")
    # Добавляем содержимое файла в базу знаний пользователя
    user_markdowns[user_id] += extract_text_from_file(
        message, document, downloaded_file
    )
    bot.send_chat_action(user_id, action="upload_document")
    user_first_stages[user_id] = False
    bot.send_message(user_id, "Готов отвечать на ваши вопросы!")


def extract_links(text):
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    return re.findall(url_pattern, text)


def scrape_website(url: str, message):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        html_string = soup.prettify()
        return html_string
    else:
        bot.send_message(
            message.from_user.id,
            f"HTTP request failed with status code {response.status_code}",
        )
        start(message)


def get_base_url(url):
    parsed_url = urlparse(url)

    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url


def convert_to_absolute_url(html, base_url):
    soup = BeautifulSoup(html, "html.parser")

    for img_tag in soup.find_all("img"):
        if img_tag.get("src"):
            src = img_tag.get("src")
            if src is not None and src.startswith(("http://", "https://")):
                continue
            absolute_url = urljoin(base_url, src)
            img_tag["src"] = absolute_url
        elif img_tag.get("data-src"):
            src = img_tag.get("data-src")
            if src is not None and src.startswith(("http://", "https://")):
                continue
            absolute_url = urljoin(base_url, src)
            img_tag["data-src"] = absolute_url

    for link_tag in soup.find_all("a"):
        href = link_tag.get("href")
        if href is not None and href.startswith(("http://", "https://")):
            continue
        absolute_url = urljoin(base_url, href)
        link_tag["href"] = absolute_url

    updated_html = str(soup)

    return updated_html


def convert_html_to_markdown(html):
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    markdown = converter.handle(html)
    return markdown


def get_markdown_from_url(urls, message):
    markdown = ""
    for url in urls:
        base_url = get_base_url(url)
        html = scrape_website(url, message)
        updated_html = convert_to_absolute_url(html, base_url)
        markdown += convert_html_to_markdown(updated_html)

    return markdown


@bot.message_handler(
    func=lambda message: user_first_stages.get(message.from_user.id, True)
)
def handle_urls(message):
    print("Extracting urls...")
    user_id = message.from_user.id

    if user_id not in user_markdowns:
        user_markdowns[user_id] = ""

    urls = extract_links(message.text)
    user_markdowns[user_id] += get_markdown_from_url(urls, message)

    user_first_stages[user_id] = False

    bot.send_message(
        user_id,
        "Отправьте Ваши документы (.docx, .txt, .pdf, .xlsx) в сообщении",
        reply_markup=skip(),
    )


@bot.message_handler(func=lambda message: True)
def handle_message(message, user_query=None, output_file=None):
    user_id = message.from_user.id
    if user_query is None:
        user_query = message.text
    if user_query is None:
        return

    if user_id not in user_first_stages or not user_first_stages[user_id]:
        if user_id not in user_contexts or user_contexts[user_id] is None:
            print("Creating context for user", user_id)
            bot.send_chat_action(user_id, action="upload_document")
            db = create_index_from_text(user_markdowns[user_id])
            user_contexts[user_id] = get_context(db)
            user_chat_histories[user_id] = []

        if user_contexts[user_id] is not None:
            dialog_questions(
                user_query,
                message,
                user_contexts[user_id],
                user_chat_histories[user_id],
            )


bot.polling()

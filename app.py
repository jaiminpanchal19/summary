# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 19:56:25 2023

@author: Dell
"""


# Import required libraries
import streamlit as st
from PyPDF2 import PdfReader
import pyttsx3
from gtts import gTTS
from googletrans import Translator
import fitz
from transformers import BartTokenizer, BartForConditionalGeneration
import asyncio

translator = Translator()

# Function for extracting text from a PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# Function for generating a summary
def generate_summary(text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=40000, min_length=1000, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function for translating text
def translate_text(text, dest_language):
    translation = translator.translate(text, dest=dest_language)
    return translation.text

# Function for converting text to speech
def convert_text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)

# Main function using Streamlit
def main():
    st.title("Summarizing and Speech Converter")
    st.write("Upload a PDF file and click the download button to download MP3")

    # File upload
    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

    if pdf_file:
        st.write("Processing... Please wait.")
        # Use asyncio to execute text extraction and summary generation asynchronously
        async def process_pdf():
            pdf_text = extract_text_from_pdf(pdf_file)
            summary = generate_summary(pdf_text)

            # Display summary
            st.subheader("Summary")
            st.text_area("Text", summary, height=200)

            # Translation and audio conversion
            status = st.selectbox("Select your language", ['English', 'Hindi', 'Gujarati'])
            if status == 'English':
                final_text = summary
                output_file = 'English_audio.mp3'
            else:
                dest_language = 'hi' if status == 'Hindi' else 'gu'
                translation = await loop.run_in_executor(None, translate_text, summary, dest_language)
                final_text = translation
                output_file = f'{status}_audio.mp3'

            if st.button('Download Audio'):
                st.spinner(text='Downloading...')
                await loop.run_in_executor(None, convert_text_to_speech, final_text, output_file)
                st.success("Downloaded")
                st.balloons()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(process_pdf())

if __name__ == "__main__":
    main()


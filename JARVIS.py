#téléchrage Ollama --> ollama.com
#pip install Ollama
#run Ollama:
#   ollama pull llama3.1 && ollama run llama3.1


import time
from itertools import chain
from langchain.chains.qa_with_sources.map_reduce_prompt import question_prompt_template
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from pydantic.v1.schema import model_schema
import os
from tensorflow.python.framework.test_ops import int_input
import subprocess
import pyaudio
import wave
import speech_recognition as sr
from gtts import gTTS
import pygame


fichier = 'test.txt'
programme = '+ou-.py'


template = """
Répond a la question plus bas.

Ici il'y a le context de la question. {context}

La question est ici : {question}

Répond:

"""

template1 = """
Répond a la question plus bas.

Ici il'y a le context de la question. {context}

La question est ici :j'aimerais que tu me gènere un script python qui:  {question}

Répond en ecrivant que du code:

"""

template3 = """
Répond a la question plus bas.

Ici il'y a le context de la question. {context}

La question est ici :j'aimerais que tu me gènere un script python qui:  {question}

Répond en ecrivant que du code:

"""

def handle_conversation():
    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model="llama3.1")
    chain = prompt | model

    context = ""
    question = ''
    print("Salut dans mon Jarvis ! Pour quitter écriver 'exit'.")
    with open(fichier, 'a',buffering=1) as item:
        while True:
            if question == "exit":
                break
            question = str(input("MOI : "))
            result = chain.invoke({"context": context, "question": question})
            print("Jarvis : ", result)
            context += f"\nUser: {question}\nAI: {result}\n"
            item.write(f"\nUser: {question}\nAI: {result}\n")


def create_python_script():
    prompt = ChatPromptTemplate.from_template(template1)
    model = OllamaLLM(model="llama3.1")
    chain = prompt | model
    context = ""
    print()
    print("Bienvenue dans mon Jarvis ! Pour quitter écriver 'exit'.\n")
    code_context = int(input("Script a rajouter\n1) OUI 2) NON : "))
    programme = str(input("\nnom du fichier python : ")) + ".py"
    print(programme)

    # rajouter du code en contexte
    if code_context == 1:
        loca = str(input("nom fichier a extraire: "))
        with open(loca, 'r',) as file:
            contenu = file.read()
        context += contenu

    #question = 'je veux que tu me fasse un juste prix entre 1 et 100 dans une boucle infini je veux aussi que le script me demande un nombre je lui donne et il me dit si le nombre est pkus petit ou plu grand'
    #question ='j'aimerais que tu me génère un script python qui me permette de générer 1000 nombre premier avec le crible deratjostène'
    question = str(input("Moi: "))
    with open(fichier, 'a',buffering=1) as item:
        while True:
            if question == "exit":
                break
            result = chain.invoke({"context": context, "question": question})
            #print("Jarvis : ", result)
            item.write(f"\nUser: {question}\nAI: {result}\n")
            # Trouver la position des 2 symboles pour avoir le script python
            if "```" in result:
                script = result.split('```')[1]
                print()
                print("--------------------------------")
                print("script python en excution:")
                print()
                python_script = script.replace("python", "")
                context = f'script : {python_script}\n'
                with open(programme, 'w') as prgm:
                    prgm.write(python_script)

                try:
                    execution = subprocess.run(['python3', programme], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if execution.returncode != 0:
                        erreur_py = execution.stderr.decode('utf-8')
                        context += f'Erreur{erreur_py}'
                        print(context)
                except Exception as e:
                    f"Erreur interne : {e}"


def email_marketing():
    fichier = 'email_marketing.txt'

    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model="llama3.1")
    chain = prompt | model
    question = str(input("Qu'est ce que fais l'entreprise : "))

    jours = ['Jour 1 : Encourager les rêves (faire rêver le lecteur sur les résultats possible)',
             'Jour 2 : Excuser les échecs passés (faire comprendre au lecteur que ce n’est pas de sa fautes)',
             'Jour 3 : Réduire les peurs (faire comprendre au lecteur qu’il n’a rien à perdre)',
             'Jour 4 : Confirmer les doutes, c est la faute de quelqu un (faire comprendre au lecteur qu on complot contre lui est en cours, contre l atteinte de ses objectifs)',
             'Jour 5 : Jeter des pierres aux ennemis (cibler un méchant et lui faire porter le chapeau.)',
             'Jour 6 : Closing (rajouter de l’urgence, dans 48h c’est finis)',
             'Jour 7 : Closing (rajouter de l’urgence, dans 24h c’est finis et le prix fait x2 )']

    for jour in jours:
        context = f'''
            À partir de ton knowledge, rédige-moi 1 e-mails dont le sujet est le suivant :{question}
            L'objectif est d'inciter le lecteur à cliquer sur le lien cliquable dans lequel je vends mon savoir faire sur {question}

            Reproduis le style de l'auteur.
            Fais du bon marketing avec des accroches.
            Ce sont des e-mails promotionnels dissimulés.
            Fait en sorte que les objets des mails soient percutants.
            Il est essentiel que le lecteur apprenne des choses en lisant l'e-mail.
            Et qu'il ai envie de relire le lendemain.

            Lorsque tu vas rédiger les e-mails, suis la structure suivante :

            {jour}

            Rappel : J'aimerais que les email apparaisse en html et en texte Chaque e-mail doit contenir un lien cliquable (un CTA clair) vers mon site, où je partage une vidéo de vente.
            Tu mettras le lien cliquable en gras
            Et le lien cliquable doit être vers la fin du mail.
            Au minimum après le 300e mots.

            Chaque e-mail doit avoir une longueur comprise entre 400 et 500 mots.

            '''
        print(context)


        with open(fichier, 'a',buffering=1) as item:
            result = chain.invoke({"context": context, "question": question})
            print("Jarvis : ", result)
            item.write(f"\nUser: {question}\nAI: {result}\n")
            print('\n----------------------------------------------------------------------------------\n')

# Fonction pour enregistrer l'audio
def record_audio(filename, duration=5):
    chunk = 1024  # Nombre de frames par buffer
    format = pyaudio.paInt16  # Format audio
    channels = 1  # Mono
    rate = 44100  # Taux d'échantillonnage

    p = pyaudio.PyAudio()

    # Ouvrir le flux audio
    stream = p.open(format=format, channels=channels,
                    rate=rate, input=True,
                    frames_per_buffer=chunk)

    print("Enregistrement...")

    frames = []

    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Enregistrement terminé.")

    # Arrêter et fermer le flux
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Sauvegarder l'audio dans un fichier WAV
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

# Fonction pour convertir l'audio en texte
def audio_to_text(filename):
    recognizer = sr.Recognizer()

    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)  # Lire l'audio
        try:
            text = recognizer.recognize_google(audio_data, language='fr-FR')  # Utiliser Google Web Speech API
            return text
        except sr.UnknownValueError:
            return "Google Speech Recognition n'a pas pu comprendre l'audio"
        except sr.RequestError as e:
            return f"Erreur de service Google Speech Recognition; {e}"

def conversation():
    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model="llama3.1")
    chain = prompt | model
    audio_filename = "enregistrement.wav"
    context = ""
    print("Salut dans mon Jarvis ! Pour enregistrer écriver 'r'\n")
    with open(fichier, 'a', buffering=1) as item:
        while True:
            enregistrer = str(input(""))
            if enregistrer == 'r':

                record_audio(audio_filename, duration=5)  # Enregistrer pendant 5 secondes

                # Convertir l'audio en texte
                question = audio_to_text(audio_filename)
                print("Texte reconnu :", str(question))

                result = chain.invoke({"context": context, "question": question})
                result = result.replace('**', '')
                result = result.replace(',', '')
                result = result.replace('.', '')
                result = result.replace('-', '')
                print("Jarvis : ", result)
                context += f"\nUser: {question}\nAI: {result}\n"
                item.write(f"\nUser: {question}\nAI: {result}\n")
                output = gTTS(text=result, lang='fr', slow=False)
                output.save('reponse.mp3')
                # Initialiser pygame
                pygame.mixer.init()

                # Charger le fichier audio
                pygame.mixer.music.load('reponse.mp3')

                # Jouer le fichier audio
                pygame.mixer.music.play()

                # Attendre que la musique finisse de jouer
                while pygame.mixer.music.get_busy():
                    time.sleep(1)


if __name__ == "__main__":
    print("Quel type d'IA veux tu? \n"
          "1) IA textuelle \n"
          "2) IA génératrice de code python\n"
          "3) IA Email Marketing\n"
          "4) IA Conversationnel")

    choix = int(input("Chossissez entre ses options:"))
    if choix == 1:
        handle_conversation()

    if choix == 2:
        create_python_script()

    if choix == 3:
        email_marketing()

    if choix == 4:
        conversation()

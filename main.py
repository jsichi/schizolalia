# schizolalia main

import configparser
import os
import subprocess
import sys
import struct
import wave
import threading
import queue
import requests
import base64
import time
from datetime import datetime

import pvcobra
import pvporcupine
import pveagle

from pvrecorder import PvRecorder
from pveagle import EagleProfile
from datetime import datetime
from random import randrange

import openai

import tkinter as tk

from PIL import ImageTk, Image

config = configparser.ConfigParser()
config.read('config.ini')

client = openai.OpenAI(
    api_key="...",
    base_url=config['openai-client']['Url'])

import whisper

inputFilename = "input"
extraInputFilename = "extra"
stopCollecting = threading.Event()
stopConvo = threading.Event()
messages = []

subtitleQueue = queue.SimpleQueue()
imageQueue = queue.SimpleQueue()

whisperModel = whisper.load_model(config['whisper']['Model'], device="cuda")

global character
character = "bubbles"

global tkRoot
global stillTalking

def call_subprocess(script_path, *args):
    subprocess.run(["python", script_path, *args], capture_output=True, text=True, check=True)

def constructFilename(dir, name, seq, suffix):
    return f"{dir}/{name}-{seq}{suffix}"

def constructWavFilename(dir, name, seq):
    return constructFilename(dir, name, seq, ".wav")

def constructImgFilename(dir, name, seq):
    return constructFilename(dir, name, seq, ".png")

def speakText(dir, text, seq):
    print("Saying:  " + text)
    seed = "10"
    if (character == "buttercup"):
        seed = "60"
    call_subprocess(
        "fish-speech/tools/api_client.py",
        "--url", config['tts-client']['Url'],
        "-t", text,
        "--reference_id", character,
        "--output", constructFilename(dir, "output", seq, ""),
        "--latency", "balanced",
        "--use_memory_cache", "on",
        "--seed", seed,
        "--streaming", "True")

def loadSpeaker(filename):
    with open(filename, 'rb') as speakerFile:
        return EagleProfile.from_bytes(speakerFile.read())
    
def createWav(dir, filename, seq):
    wavFile = wave.open(constructWavFilename(dir, filename, seq), "w")
    wavFile.setnchannels(1)
    wavFile.setsampwidth(2)
    wavFile.setframerate(16000)
    return wavFile

def collectInputChunk(recorder, cobra, wavFile):
    global stillTalking

    silenceCount = 0
    while (silenceCount < 40) and not stopCollecting.is_set():
        pcm = recorder.read()
        voiceProb = cobra.process(pcm)
        wavFile.writeframes(struct.pack("h" * len(pcm), *pcm))
        if voiceProb > 0.5:
            stillTalking = True
            silenceCount = 0
        else:
            silenceCount += 1
    wavFile.close()

def loadImages(name):
    images = dict()
    for state in ['sleeping', 'listening', 'talking']:
        images[state] = ImageTk.PhotoImage(file=f"images/{name}/{state}.jpg")
    return images

def listenForWakeword(recorder, porcupine):
    global character
    awake = False
    while not awake and not stopConvo.is_set():
        pcm = recorder.read()
        result = porcupine.process(pcm)
        if result >= 0:
            if (result == 1):
                character = "buttercup"
            else:
                character = "bubbles"
            enqueueEvent("<<WakewordHeard>>")
            awake = True

def waitForSilence(recorder, cobra):
    silenceCount = 0
    while not stopConvo.is_set() and (silenceCount < 50):
        pcm = recorder.read()
        voiceProb = cobra.process(pcm)
        if (voiceProb > 0.5):
            silenceCount = 0
        else:
            silenceCount += 1

def collectInitialInput(recorder, cobra, eagle, dir, speakerNames, seq):
    eagle.reset()
    wavFile = createWav(dir, inputFilename, seq)
    collecting = True
    heardVoice = False
    silenceCount = 0
    while collecting and not stopConvo.is_set():
        pcm = recorder.read()
        voiceProb = cobra.process(pcm)
        wavFile.writeframes(struct.pack("h" * len(pcm), *pcm))
        if voiceProb > 0.5:
            silenceCount = 0
            heardVoice = True
            scores = eagle.process(pcm)
            maxScore = max(scores)
            if maxScore > 0.3:
                maxIndex = scores.index(maxScore)
                print("Speaker:  " + speakerNames[maxIndex])
                print(maxScore)
                print()
        else:
            silenceCount += 1
            if heardVoice:
                if silenceCount > 30:
                    collecting = False
            else:
                if silenceCount > 500:
                    collecting = False
    wavFile.close()
    return heardVoice
                    

def recordInput(target, input):
    target.append({"role": "user", "content": input}),

def recordOutput(target, output):
    target.append({"role": "assistant", "content": output}),

def processRestOfInput(recorder, cobra, dir, seq):
    global stillTalking

    while not stopConvo.is_set():
        stillTalking = False
        stopCollecting.clear()
        wavFile = createWav(dir, extraInputFilename, seq)
        collectThread = threading.Thread(
            target=collectInputChunk, args=(recorder, cobra, wavFile))
        collectThread.start()
        whisperResult = whisperModel.transcribe(
            constructWavFilename(dir, inputFilename, seq), fp16=False, language='English')['text']
        print("Heard:  " + whisperResult)
        subtitleQueue.put(whisperResult)
        enqueueEvent("<<InputHeard>>")
        (tentativeResponse, toolCalled) = sendChat(dir, whisperResult, seq)
        print("Tentative response:  " + tentativeResponse)
        stopCollecting.set()
        collectThread.join()
        if stillTalking and not toolCalled:
            print('Concatenating')
            infiles = [inputFilename, extraInputFilename]
            data = []
            for infile in infiles:
                with wave.open(constructWavFilename(dir, infile, seq), 'rb') as w:
                    data.append([w.getparams(), w.readframes(w.getnframes())])
            with wave.open(constructWavFilename(dir, inputFilename, seq), 'wb') as output:
                output.setparams(data[0][0])
                output.writeframes(data[0][1])
                output.writeframes(data[1][1])
        else:
            recordInput(messages, whisperResult)
            recordOutput(messages, tentativeResponse)
            with open(dir + "/transcript.txt", 'a') as transcript:
                print(f"[human:{seq}] {whisperResult}", file=transcript)
                print(f"[Bubbles:{seq}] {tentativeResponse}", file=transcript)
            return tentativeResponse

def processInput(recorder, cobra, eagle, dir, speakerNames, seq):
    if collectInitialInput(recorder, cobra, eagle, dir, speakerNames, seq):
        return processRestOfInput(recorder, cobra, dir, seq)
    else:
        return ""

def sendChat(dir, input, seq):
    tentative = messages.copy()
    recordInput(tentative, input)
    defaultTool = {
        'type': 'function',
        'function': {
            'name': 'defaultTool',
            'description': (
                'This is the default tool;'
                'call this tool when none of the other tools seem to fit the request from the user.'
            ),
            'parameters': {
            },
        },
    }
    snoozeTool = {
        'type': 'function',
        'function': {
            'name': 'snoozeTool',
            'strict': 'true',
            'description': 'Call this tool whenever the user says to go to sleep.',
            'parameters': {
            },
        },
    }
    drawTool = {
        'type': 'function',
        'function': {
            'name': 'drawTool',
            'strict': 'true',
            'description': 'Call this tool whenever the user says to draw something.',
            'parameters': {
                'type': 'object',
                'required': ['prompt'],
                'properties': {
                    'prompt': {
                        'type': 'string',
                        'description': 'The prompt to send to stable diffusion for rendering.',
                    },
                },
            },
        }
    }
    tools = [defaultTool, snoozeTool, drawTool]
    chatResponse = client.chat.completions.create(
        model = character,
        messages = tentative,
        tools = tools)
    print(chatResponse)
    response = chatResponse.choices[0]
    if response.message.tool_calls:
        for tool in response.message.tool_calls:
            if tool.function.name == "snoozeTool":
                return ("", True)
            if tool.function.name == "drawTool":
                imgPrompt = tool.function.arguments
                stylePrompt = "a kindergartener's crayon drawing, (cute:2), (pretty:2)"
                if (character == "buttercup"):
                    stylePrompt = "a child's crayon drawing, (brown:1.5), (black:1.5), "
                    "(green:1.5), (emo), (scribbling:2), (sloppy:2)"
                print("Image prompt: " +  imgPrompt)
                payload = {
                    "prompt": f"{imgPrompt}, {stylePrompt}",
                    "steps": 25,
                    "width": 512,
                    "height": 512,
                    "seed": 10
                }
                sdUrl = config['stable-diffusion-client']['Url']
                response = requests.post(
                    url=f"{sdUrl}/txt2img",
                    json=payload)
                r = response.json()
                imgFile = constructImgFilename(dir, "output", seq)
                with open(imgFile, 'wb') as f:
                    f.write(base64.b64decode(r['images'][0]))
                imageQueue.put(imgFile)
                enqueueEvent("<<ImageGenerated>>")
                time.sleep(5)
                return ("Tada!", True)
        chatResponse = client.chat.completions.create(
            model=character,
            messages=tentative)
        print(chatResponse)
        response = chatResponse.choices[0]
    responseText = response.message.content
    return (responseText, False)

def enqueueEvent(event):
    if not stopConvo.is_set():
        tkRoot.event_generate(event)

def convoLoopThread():
    try:
        convoLoop()
    finally:
        enqueueEvent("<<AllDone>>")

def convoLoop():
    picovoiceKey = config['picovoice']['AccessKey']

    porcupineKeywordPaths = [
        'wakewords/Hi-Bubbles_en_linux_v3_0_0.ppn',
        'wakewords/Hey-Buttercup_en_linux_v3_0_0.ppn',
    ]
    try:
        porcupine = pvporcupine.create(
            access_key=picovoiceKey,
            keyword_paths=porcupineKeywordPaths,
            sensitivities = [0.25] * len(porcupineKeywordPaths))
    except pvporcupine.PorcupineError as e:
        print("Failed to initialize Porcupine")
        raise e

    try:
        cobra = pvcobra.create(access_key=picovoiceKey)
    except pvcobra.CobraError as e:
        print("Failed to initialize Cobra")
        raise e

    speakerNames = [
        "speakers/pascal.eagle", "speakers/mia.eagle", "speakers/john.eagle",
        "speakers/steve.eagle"
    ]
    speakerProfiles = list(map(loadSpeaker, speakerNames))
    try:
        eagle = pveagle.create_recognizer(picovoiceKey, speakerProfiles)
    except pveagle.EagleError as e:
        print("Failed to initialize Eagle")
        raise e
    
    recorder = PvRecorder(
        frame_length=porcupine.frame_length)
    recorder.start()

    dirUnique = datetime.now().strftime("%Y%m%d-%H%M%S")
    dirSeq = 0

    try:
        while not stopConvo.is_set():
            dirSeq += 1
            messages.clear()
            listenForWakeword(recorder, porcupine)

            conversing = True
            dir = f"convos-{dirUnique}/c{dirSeq}-{character}"
            os.makedirs(dir)
            fileSeq = 0

            while conversing:
                fileSeq += 1
                output = processInput(recorder, cobra, eagle, dir, speakerNames, fileSeq)
                if output:
                    enqueueEvent("<<InputProcessed>>")

                    prefix = "(excited) "
                    if (character == "buttercup"):
                        prefix = "(angry) "
                    annotatedOutput = prefix + output

                    speakText(dir, annotatedOutput, fileSeq)
                    waitForSilence(recorder, cobra)

                    enqueueEvent("<<WakewordHeard>>")
                else:
                    enqueueEvent("<<ConvoFinished>>")
                    conversing = False

    finally:
        recorder.delete()
        porcupine.delete()
        cobra.delete()
        eagle.delete()

def uiLoop():
    global tkRoot
    tkRoot = tk.Tk()

    screenHeight = tkRoot.winfo_screenheight()
    screenWidth = tkRoot.winfo_screenwidth()
    pad = 3
    tkRoot.geometry("{0}x{1}+0+0".format(screenWidth - pad, screenHeight - pad))
    tkRoot.attributes("-fullscreen", True)
    tkRoot.title("Let's Have A Chat!")

    canvas = tk.Canvas(
        tkRoot, bg = "black",
        height = screenHeight,
        width = screenWidth)
    canvas.pack()

    tkRoot.update()

    characterImages = dict()
    imagesOnCanvas = dict()
    for name in ['bubbles', 'buttercup']:
        characterImages[name] = loadImages(name)
        imagesOnCanvas[name] = canvas.create_image(
            screenWidth/2, screenHeight/2, image=characterImages[name]['sleeping'], anchor=tk.NW)

    outputPhoto = None
    outputImage = canvas.create_image(
        screenWidth/2, screenHeight/2, image=outputPhoto,
        state=tk.HIDDEN, anchor=tk.CENTER)

    subtitle = canvas.create_text(
        screenWidth / 2, (9 * screenHeight) / 10,
        width = (9 * screenHeight) / 10,
        text="", font=("Arial", 40, "bold"),
        fill="yellow", justify=tk.CENTER, anchor=tk.CENTER)

    interval = 2000
    sleeping = True

    def clearSubtitle():
        canvas.itemconfigure(subtitle, text="")

    def wakewordHeard(event):
        nonlocal sleeping
        sleeping = False
        canvas.itemconfig(imagesOnCanvas[character], image=characterImages[character]['listening'])
        canvas.tag_raise(imagesOnCanvas[character])

    def inputProcessed(event):
        canvas.itemconfig(imagesOnCanvas[character], image=characterImages[character]['talking'])
        tkRoot.after(3000, clearSubtitle)

    def convoFinished(event):
        nonlocal sleeping
        sleeping = True
        canvas.itemconfig(imagesOnCanvas[character], image=characterImages[character]['sleeping'])
        canvas.itemconfig(outputImage, state=tk.HIDDEN)
        tkRoot.after(3000, clearSubtitle)

    def allDone(event):
        tkRoot.destroy()

    def inputHeard(event):
        canvas.itemconfig(outputImage, state=tk.HIDDEN)
        canvas.itemconfigure(subtitle, text=subtitleQueue.get())
        canvas.tag_raise(subtitle)

    def imageGenerated(event):
        canvas.itemconfig(imagesOnCanvas[character], image=characterImages[character]['talking'])
        imgFile = imageQueue.get()

        nonlocal outputPhoto
        outputPhoto = ImageTk.PhotoImage(file=imgFile)
        canvas.itemconfig(
            outputImage,
            image=outputPhoto,
            state=tk.NORMAL,
            anchor=tk.CENTER)
        canvas.tag_raise(outputImage)

    def movePic():
        if sleeping:
            for name in characterImages.keys():
                picSleeping = characterImages[name]['sleeping']
                xNew = randrange(0, screenWidth - picSleeping.width())
                yNew = randrange(0, screenHeight - picSleeping.height())
                canvas.coords(imagesOnCanvas[name], (xNew, yNew))
        else:
            canvas.coords(imagesOnCanvas[character], (screenWidth/2, screenHeight/2))
        tkRoot.after(interval, movePic)

    tkRoot.bind("<<WakewordHeard>>", wakewordHeard)
    tkRoot.bind("<<InputProcessed>>", inputProcessed)
    tkRoot.bind("<<ConvoFinished>>", convoFinished)
    tkRoot.bind("<<AllDone>>", allDone)
    tkRoot.bind("<<InputHeard>>", inputHeard)
    tkRoot.bind("<<ImageGenerated>>", imageGenerated)
    tkRoot.bind("<Escape>", allDone)

    tkRoot.after(0, movePic)
    tkRoot.mainloop()

def main():
    # warm up the model
    client.chat.completions.create(
        model=character,
        messages=[{"role":"user", "content":""}])

    convoThread = threading.Thread(target=convoLoopThread)
    convoThread.start()
    try:
        uiLoop()
    finally:
        stopCollecting.set()
        stopConvo.set()
    convoThread.join()

main()


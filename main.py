# schizolalia main

import configparser
import os
import subprocess
import sys
import struct
import wave
import threading
import queue
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

whisperModel = whisper.load_model(config['whisper']['Model'], device="cuda")

def call_subprocess(script_path, *args):
    subprocess.run(["python", script_path, *args], capture_output=True, text=True, check=True)

def constructFilename(dir, name, seq):
    return f"{dir}/{name}-{seq}.wav"

def speakText(dir, text, seq):
    print("Saying:  " + text)
    call_subprocess(
        "fish-speech/tools/api_client.py",
        "--url", config['tts-client']['Url'],
        "-t", text,
        "--reference_id", "bubbles",
        "--output", constructFilename(dir, "output", seq),
        "--latency", "balanced",
        "--use_memory_cache", "on",
        "--seed", "10",
        "--streaming", "True")

def loadSpeaker(filename):
    with open(filename, 'rb') as speakerFile:
        return EagleProfile.from_bytes(speakerFile.read())
    
def createWav(dir, filename, seq):
    wavFile = wave.open(constructFilename(dir, filename, seq), "w")
    wavFile.setnchannels(1)
    wavFile.setsampwidth(2)
    wavFile.setframerate(16000)
    return wavFile

def collectInputChunk(recorder, cobra, wavFile):
    silenceCount = 0
    while (silenceCount < 40) and not stopCollecting.is_set():
        pcm = recorder.read()
        voiceProb = cobra.process(pcm)
        wavFile.writeframes(struct.pack("h" * len(pcm), *pcm))
        if voiceProb > 0.5:
            global stillTalking
            stillTalking = True
            silenceCount = 0
        else:
            silenceCount += 1
    wavFile.close()

def listenForWakeword(recorder, porcupine):
    awake = False
    while not awake and not stopConvo.is_set():
        pcm = recorder.read()
        result = porcupine.process(pcm)
        if result >= 0:
            enqueueEvent("<<WakewordHeard>>")
            awake = True

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
            constructFilename(dir, inputFilename, seq), fp16=False, language='English')['text']
        print("Heard:  " + whisperResult)
        subtitleQueue.put(whisperResult)
        enqueueEvent("<<InputHeard>>")
        tentativeResponse = sendChat(whisperResult)
        print("Tentative response:  " + tentativeResponse)
        stopCollecting.set()
        collectThread.join()
        if stillTalking:
            print('Concatenating')
            infiles = [inputFilename, extraInputFilename]
            data = []
            for infile in infiles:
                with wave.open(constructFilename(dir, infile, seq), 'rb') as w:
                    data.append([w.getparams(), w.readframes(w.getnframes())])
            with wave.open(constructFilename(dir, inputFilename, seq), 'wb') as output:
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

def sendChat(input):
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
            'description': 'Call this tool only when the user tells the assistant to go to sleep.',
            'parameters': {
            },
        },
    }
    tools = [defaultTool, snoozeTool]
    chatResponse = client.chat.completions.create(
        model="bubbles",
        messages=tentative,
        tools=tools)
    print(chatResponse)
    response = chatResponse.choices[0]
    for tool in response.message.tool_calls or []:
        if tool.function.name == "snoozeTool":
            return ""
    chatResponse = client.chat.completions.create(
        model="bubbles",
        messages=tentative)
    response = chatResponse.choices[0]
    responseText = response.message.content
    return responseText

def enqueueEvent(event):
    if not stopConvo.is_set():
        global tkRoot
        tkRoot.event_generate(event)

def convoLoopThread():
    try:
        convoLoop()
    finally:
        enqueueEvent("<<AllDone>>")

def convoLoop():
    picovoiceKey = config['picovoice']['AccessKey']

    porcupineKeywordPaths = ['wakewords/Hi-Bubbles_en_linux_v3_0_0.ppn']
    try:
        porcupine = pvporcupine.create(
            access_key=picovoiceKey,
            keyword_paths=porcupineKeywordPaths,
            sensitivities = [0.5] * len(porcupineKeywordPaths))
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
            dir = f"convos-{dirUnique}/c{dirSeq}"
            os.makedirs(dir)
            fileSeq = 0

            while conversing:
                fileSeq += 1
                output = processInput(recorder, cobra, eagle, dir, speakerNames, fileSeq)
                if output:
                    enqueueEvent("<<InputProcessed>>")

                    annotatedOutput = "(excited) " + output

                    speakText(dir, annotatedOutput, fileSeq)

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
    tkRoot=tk.Tk()

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

    picSleeping = ImageTk.PhotoImage(file="images/bubbles/sleeping.jpg")
    picListening = ImageTk.PhotoImage(file="images/bubbles/listening.jpg")
    picTalking = ImageTk.PhotoImage(file="images/bubbles/talking.jpg")

    imgOnCanvas = canvas.create_image(
        screenWidth/2, screenHeight/2, image=picSleeping, anchor="nw")

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
        canvas.itemconfig(imgOnCanvas, image=picListening)

    def inputProcessed(event):
        canvas.itemconfig(imgOnCanvas, image=picTalking)
        tkRoot.after(3000, clearSubtitle)

    def convoFinished(event):
        nonlocal sleeping
        sleeping = True
        canvas.itemconfig(imgOnCanvas, image=picSleeping)
        tkRoot.after(3000, clearSubtitle)

    def allDone(event):
        tkRoot.destroy()

    def inputHeard(event):
        canvas.itemconfigure(subtitle, text=subtitleQueue.get())

    def movePic():
        if sleeping:
            xNew = randrange(0, screenWidth - picSleeping.width())
            yNew = randrange(0, screenHeight - picSleeping.height())
            canvas.coords(imgOnCanvas, (xNew, yNew))
        else:
            canvas.coords(imgOnCanvas, (screenWidth/2, screenHeight/2))
        tkRoot.after(interval, movePic)

    tkRoot.bind("<<WakewordHeard>>", wakewordHeard)
    tkRoot.bind("<<InputProcessed>>", inputProcessed)
    tkRoot.bind("<<ConvoFinished>>", convoFinished)
    tkRoot.bind("<<AllDone>>", allDone)
    tkRoot.bind("<<InputHeard>>", inputHeard)
    tkRoot.bind("<Escape>", allDone)
    tkRoot.after(interval, movePic)

    tkRoot.mainloop()

def main():
    # warm up the model
    client.chat.completions.create(
        model="bubbles",
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


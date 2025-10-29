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
import random
import string
import json
import asyncio
import tempfile
import imageio

import pvcobra
import pvporcupine
import pveagle

from pvrecorder import PvRecorder
from pveagle import EagleProfile
from datetime import datetime
from random import randrange
from datetime import datetime

import openai

import aime_api_client_interface

import tkinter as tk

from PIL import ImageTk, Image

config = configparser.ConfigParser()
config.read('config.ini')
aimeUrl = config['aime-client']['Url']

client = openai.OpenAI(
    api_key="...",
    base_url=config['openai-client']['Url'])

import whisper

inputFilename = "input"
extraInputFilename = "extra"
stopCollecting = threading.Event()
stopConvo = threading.Event()
messages = []
lastImgPath = None

subtitleQueue = queue.SimpleQueue()
imageQueue = queue.SimpleQueue()
switchQueue = queue.SimpleQueue()

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
    for state in ['sleeping', 'listening', 'talking', 'drawing']:
        images[state] = ImageTk.PhotoImage(file=f"images/{name}/{state}.jpg")
    return images

def listenForWakeword(recorder, porcupineEn, porcupineKo):
    global character
    awake = False
    while not awake and not stopConvo.is_set():
        pcm = recorder.read()
        resultEn = porcupineEn.process(pcm)
        resultKo = porcupineKo.process(pcm)
        if resultEn >= 0:
            match resultEn:
                case 1:
                    character = "buttercup"
                case _:
                    character = "bubbles"
            enqueueEvent("<<WakewordHeard>>")
            awake = True
        else:
            if resultKo >= 0:
                character = "sejong"
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

async def processRestOfInput(recorder, cobra, dir, seq):
    global stillTalking

    while not stopConvo.is_set():
        stillTalking = False
        stopCollecting.clear()
        wavFile = createWav(dir, extraInputFilename, seq)
        collectThread = threading.Thread(
            target=collectInputChunk, args=(recorder, cobra, wavFile))
        collectThread.start()
        match character:
            case 'sejong':
                lang = 'Korean'
            case _:
                lang = 'English'
        whisperResult = whisperModel.transcribe(
            constructWavFilename(dir, inputFilename, seq), fp16=False, language=lang)['text']
        print("Heard:  " + whisperResult)
        subtitleQueue.put(whisperResult)
        enqueueEvent("<<InputHeard>>")
        (tentativeResponse, toolCalled) = await sendChat(dir, whisperResult, seq)
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

async def processInput(recorder, cobra, eagle, dir, speakerNames, seq):
    if collectInitialInput(recorder, cobra, eagle, dir, speakerNames, seq):
        return await processRestOfInput(recorder, cobra, dir, seq)
    else:
        return ""

def cleanName(text):
    return text.lower()

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
editTool = {
    'type': 'function',
    'function': {
        'name': 'editTool',
        'strict': 'true',
        'description': 'Call this tool whenever the user wants to modify the current image.',
        'parameters': {
            'type': 'object',
            'required': ['prompt'],
            'properties': {
                'prompt': {
                    'type': 'string',
                    'description': 'The prompt to send to stable diffusion '
                    'for editing the current image.',
                },
            },
        },
    }
}
switchTool = {
    'type': 'function',
    'function': {
        'name': 'switchTool',
        'strict': 'true',
        'description':
        'Call this tool whenever the user wants to talk to a different character.',
        'parameters': {
            'type': 'object',
            'required': ['character'],
            'properties': {
                'character': {
                    'type': 'string',
                    'enum': ['bubbles', 'buttercup', 'sejong'],
                    'description': 'The character to switch to.',
                },
            },
        },
    }
}
snapshotTool = {
    'type': 'function',
    'function': {
        'name': 'snapshotTool',
        'strict': 'true',
        'description': 'Call this tool whenever the user asks to take a picture.',
        'parameters': {
        },
    },
}
tools = [defaultTool, snoozeTool, drawTool, editTool, switchTool]

def saveImg(dir, seq, img):
    imgFile = constructImgFilename(dir, "output", seq)
    with open(imgFile, 'wb') as f:
        f.write(img)
    return imgFile

def displayImg(imgFile):
    global lastImgPath
    lastImgPath = imgFile
    imageQueue.put(imgFile)
    enqueueEvent("<<ImageGenerated>>")
    time.sleep(2)

def loadLastImg():
    with open(lastImgPath, "rb") as file:
        return file.read()

async def drawImg(prompt, initImage):
    params = {
        'prompt': prompt,
        'seed': 10,
        'height': 512,
        'width': 512,
        'steps': 25,
        'image': initImage,
        'provide_progress_images': 'latent',
        'wait_for_result': True
    }

    # Call the AIME API
    aimeApi = aime_api_client_interface.ModelAPI(
        aimeUrl,
        'stable_diffusion_3_5',
        'admin',
        '6a17e2a5-b706-03cb-1a32-94b4a1df67da')
    aimeApi.do_api_login()

    def progressCallback(progressInfo, progressData):
        print(progressInfo)
        if progressData:
            if (progressInfo.get('progress') or 0) >= 20:
                images = progressData.get('progress_images')
                for i, img_b64 in enumerate(images or []):
                    header, imgData = img_b64.split(',', 1) if ',' in img_b64 else (None, img_b64)
                    (fd, imgFile) = tempfile.mkstemp(dir='/tmp')
                    with os.fdopen(fd, 'wb') as f:
                        f.write(base64.b64decode(imgData))
                    imageQueue.put(imgFile)
                    enqueueEvent("<<ImageGenerated>>")

    final = await aimeApi.do_api_request_async(
        params,
        None,
        progressCallback)
    await aimeApi.close_session()

    images = final.get('images') or final.get('job_result', {}).get('images', [])
    for i, img_b64 in enumerate(images):
        header, img_data = img_b64.split(',', 1) if ',' in img_b64 else (None, img_b64)
        return img_data
    return ""
    
async def sendChat(dir, input, seq):
    tentative = messages.copy()
    recordInput(tentative, input)
    firstModel = character
    if character == "sejong":
        firstModel = "hf.co/unsloth/Llama-3.3-70B-Instruct-GGUF:IQ2_XXS"
    else:
        firstModel = character
    chatResponse = client.chat.completions.create(
        model = firstModel,
        messages = tentative,
        tools = tools)
    print(chatResponse)
    response = chatResponse.choices[0]
    for tool in response.message.tool_calls or []: 
        match tool.function.name:
            case 'snoozeTool':
                return ("", True)
            case 'switchTool':
                newCharacter = cleanName(json.loads(tool.function.arguments)['character'])
                print(f"Switch request:  {newCharacter}")
                match newCharacter:
                    case 'bubbles' | 'buttercup' | 'sejong':
                        switchQueue.put(newCharacter)
                        return ("", True)
            case 'drawTool':
                imgPrompt = json.loads(tool.function.arguments)['prompt']
                print("Image prompt: " +  imgPrompt)
                match character:
                    case 'sejong':
                        stylePrompt = "a pen and ink drawing in traditional Korean style"
                    case 'buttercup':
                        stylePrompt = "a child's crayon drawing, (brown:1.5), (black:1.5), "
                        "(green:1.5), (emo), (scribbling:2), (sloppy:2)"
                    case _:
                        stylePrompt = "a kindergartener's drawing, (cute:2), (pretty:2), (crayon)"
                enqueueEvent("<<DrawingStarted>>")
                response = await drawImg(f"{imgPrompt}, {stylePrompt}", None)
                imgFile = saveImg(dir, seq, base64.b64decode(response))
                displayImg(imgFile)
                return ("#", True)
            case 'editTool':
                if lastImgPath:
                    editPrompt = json.loads(tool.function.arguments)['prompt']
                    print("Edit prompt: " +  editPrompt)
                    response = await drawImg(editPrompt, loadLastImg())
                    imgFile = saveImg(dir, seq, base64.b64decode(response))
                    displayImg(imgFile)
                    return ("#", True)
            case 'snapshotTool':
                print("Taking snapshot.")
                camera = imageio.get_reader("<video0>")
                screenshot = camera.get_data(0)
                imgFile = constructImgFilename(dir, "output", seq)
                imageio.imwrite(imgFile, screenshot)
                displayImg(imgFile)
                return ("#", True)

    if tools:
        chatResponse = client.chat.completions.create(
            model=character,
            messages=tentative)
        print(chatResponse)
        response = chatResponse.choices[0]
    responseText = response.message.content
    if character == "sejong":
        responseText = "(pause) " + responseText
    return (responseText, False)

def enqueueEvent(event):
    if not stopConvo.is_set():
        tkRoot.event_generate(event)

def convoLoopThread():
    try:
        asyncio.run(convoLoop())
    finally:
        enqueueEvent("<<AllDone>>")

async def convoLoop():
    global character
    picovoiceKey = config['picovoice']['AccessKey']

    porcupineKeywordPathsEn = [
        'wakewords/Hi-Bubbles_en_linux_v3_0_0.ppn',
        'wakewords/Hey-Buttercup_en_linux_v3_0_0.ppn'
    ]
    porcupineKeywordPathsKo = [
        'wakewords/jeonha_ko_linux_v3_0_0.ppn',
    ]
    try:
        porcupineEn = pvporcupine.create(
            access_key=picovoiceKey,
            keyword_paths=porcupineKeywordPathsEn,
            sensitivities = [0.25] * len(porcupineKeywordPathsEn))
        porcupineKo = pvporcupine.create(
            access_key=picovoiceKey,
            keyword_paths=porcupineKeywordPathsKo,
            model_path='wakewords/porcupine_params_ko.pv',
            sensitivities = [0.5] * len(porcupineKeywordPathsKo))
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
        frame_length=porcupineEn.frame_length)
    recorder.start()

    dirUnique = datetime.now().strftime("%Y%m%d-%H%M%S")
    dirSeq = 0

    try:
        while not stopConvo.is_set():
            dirSeq += 1
            messages.clear()
            if not switchQueue.empty():
                character = switchQueue.get()
                enqueueEvent("<<WakewordHeard>>")
            else:
                listenForWakeword(recorder, porcupineEn, porcupineKo)

            conversing = True
            dir = f"convos-{dirUnique}/c{dirSeq}-{character}"
            os.makedirs(dir)
            fileSeq = 0

            while conversing:
                fileSeq += 1
                output = await processInput(recorder, cobra, eagle, dir, speakerNames, fileSeq)
                match output:
                    case "#":
                        enqueueEvent("<<WakewordHeard>>")
                    case "":
                        enqueueEvent("<<ConvoFinished>>")
                        conversing = False
                    case _:
                        enqueueEvent("<<InputProcessed>>")

                        match character:
                            case 'bubbles':
                                prefix = "(excited) "
                            case 'buttercup':
                                prefix = "(angry) "
                            case _:
                                prefix = ""
                        annotatedOutput = prefix + output
                        speakText(dir, annotatedOutput, fileSeq)
                        waitForSilence(recorder, cobra)
                        enqueueEvent("<<WakewordHeard>>")

    finally:
        recorder.delete()
        porcupineEn.delete()
        porcupineKo.delete()
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
    for name in ['bubbles', 'buttercup', 'sejong']:
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

    def drawingStarted(event):
        canvas.itemconfig(imagesOnCanvas[character], image=characterImages[character]['drawing'])

    def imageGenerated(event):
        imgFile = imageQueue.get()

        nonlocal outputPhoto
        img = Image.open(imgFile)
        if imgFile.startswith("/tmp"):
            os.remove(imgFile)
        img = img.resize((512, 512), Image.Resampling.BICUBIC)
        outputPhoto = ImageTk.PhotoImage(img)
        canvas.itemconfig(
            outputImage,
            image=outputPhoto,
            state=tk.NORMAL,
            anchor=tk.CENTER)
        canvas.tag_raise(outputImage)
        clearSubtitle()

    def movePic():
        if sleeping:
            for name in characterImages.keys():
                picSleeping = characterImages[name]['sleeping']
                xNew = randrange(0, screenWidth - picSleeping.width())
                yNew = randrange(0, screenHeight - picSleeping.height())
                canvas.coords(imagesOnCanvas[name], (xNew, yNew))
        else:
            canvas.coords(imagesOnCanvas[character], (0, 0))
        tkRoot.after(interval, movePic)

    tkRoot.bind("<<WakewordHeard>>", wakewordHeard)
    tkRoot.bind("<<InputProcessed>>", inputProcessed)
    tkRoot.bind("<<ConvoFinished>>", convoFinished)
    tkRoot.bind("<<AllDone>>", allDone)
    tkRoot.bind("<<InputHeard>>", inputHeard)
    tkRoot.bind("<<DrawingStarted>>", drawingStarted)
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


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

from pvrecorder import PvRecorder
from datetime import datetime
from random import randrange
from datetime import datetime

import openai

import aime_api_client_interface

import tkinter as tk

import cv2
import face_recognition

from PIL import ImageTk, Image, ImageOps, ImageDraw

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
humanRecognized = threading.Event()
messages = []
lastImgPath = None

subtitleQueue = queue.SimpleQueue()
imageQueue = queue.SimpleQueue()
switchQueue = queue.SimpleQueue()

whisperModel = whisper.load_model(config['whisper']['Model'], device="cuda")

immichUrl = config['immich']['Url']
immichHeaders = {
    'Accept': 'application/json',
    'x-api-key': config['immich']['ApiKey']
}

global character
character = "no one"

global human
human = "A Stranger"

global tkRoot
global stillTalking
global webcam
global webcamFile

def drawBoundingBox(img, coords):
    draw = ImageDraw.Draw(img)
    color = (0, 255, 0)
    draw.rectangle(coords, outline=color, width=3)

class Webcam:
    def __init__(self, window, width=512, height=512):
        self.window = window
        self.width = width
        self.height = height

    def show(self):
        assert(webcamFile)
        self.frameCount = 0
        self.found = False
        self.cap = cv2.VideoCapture(0)
        self.label = tk.Label(self.window, width=self.width, height=self.height)
        self.label.pack()
        self.label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.showFrames()

    def showFrames(self):
        recognized = False
        cv2image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)
        locs = face_recognition.api.face_locations(cv2image, 1, "hog")
        imgOrig = Image.fromarray(cv2image)
        img = imgOrig.transpose(Image.FLIP_LEFT_RIGHT)
        color = (255, 0, 0)
        if self.found:
            img.save(webcamFile)
            recognizeFace(webcamFile)
            recognized = True
            print(human)
            draw = ImageDraw.Draw(img)
            draw.text((50, 50), human, color, font_size=50)
        else:
            self.frameCount += 1
            if locs and (self.frameCount > 20):
                loc = locs[0]
                xLeft = loc[3]
                yTop = loc[0]
                xRight = loc[1]
                yBottom = loc[2]
                # FLIP_LEFT_RIGHT for bounding box as well
                xLeft = img.width - xLeft
                xRight = img.width - xRight
                coords = (xRight, yTop, xLeft, yBottom)
                drawBoundingBox(img, coords)
                self.found = True
            else:
                draw = ImageDraw.Draw(img)
                for _ in range(10):
                    x1 = random.randint(0, img.width)
                    x2 = random.randint(0, img.width)
                    y1 = random.randint(0, img.height)
                    y2 = random.randint(0, img.height)
                    draw.line([(x1,y1),(x2,y2)], color)
        imgtk = ImageTk.PhotoImage(image=img)  # Convert image to PhotoImage
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)
        if recognized:
            humanRecognized.set()
        else:
            self.label.after(20, self.showFrames)

    def hide(self):
        if self.label:
            self.label.destroy()
        self.label = None
        if self.cap:
            self.cap.release()
        self.cap = None
        global webcamFile
        webcamFile = None

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

def waitForHumanRecognition():
    while not stopConvo.is_set() and not humanRecognized.is_set():
        time.sleep(1)

def waitForSilence(recorder, cobra):
    silenceCount = 0
    while not stopConvo.is_set() and (silenceCount < 50):
        pcm = recorder.read()
        voiceProb = cobra.process(pcm)
        if (voiceProb > 0.5):
            silenceCount = 0
        else:
            silenceCount += 1

def collectInitialInput(recorder, cobra, dir, seq):
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

def recordExchange(target, input, output, dir, seq):
    recordInput(messages, input)
    recordOutput(messages, output)
    with open(dir + "/transcript.txt", 'a') as transcript:
        print(f"[{human}:{seq}] {input}", file=transcript)
        print(f"[{character}:{seq}] {output}", file=transcript)

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
            recordExchange(messages, whisperResult, tentativeResponse, dir, seq)
            return tentativeResponse

async def processInput(recorder, cobra, dir, seq):
    if collectInitialInput(recorder, cobra, dir, seq):
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
        'description': 'Call this tool whenever the user asks to take a picture, or asks what you see.',
        'parameters': {
        },
    },
}
tools = [defaultTool, snoozeTool, drawTool, editTool, switchTool, snapshotTool]

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

def uploadImmich(file):
    stats = os.stat(file)
    data = {
        'deviceAssetId': f'{file}-{stats.st_mtime}',
        'deviceId': 'schizolalia',
        'fileCreatedAt': datetime.fromtimestamp(stats.st_mtime),
        'fileModifiedAt': datetime.fromtimestamp(stats.st_mtime),
        'isFavorite': 'false',
    }
    with open(file, 'rb') as f:
        files = {
            'assetData': f
        }
        response = requests.post(
            f'{immichUrl}/assets', headers=immichHeaders, data=data, files=files)
    json = response.json()
    print(json)
    return json['id']

def waitForImmich():
    busy = True
    while busy:
        response = requests.get(
            f'{immichUrl}/jobs', headers=immichHeaders)
        json = response.json()
        busy = False
        for index, key in enumerate(json):
            active = json[key]['queueStatus']['isActive']
            if active:
                busy = True

def recognizeFace(imgFile):
    assetId = uploadImmich(imgFile)
    waitForImmich()
    response = requests.get(
        f'{immichUrl}/assets/{assetId}', headers=immichHeaders)
    json = response.json()
    print(json)
    people = json['people']
    if people:
        person = people[0]
        global human
        human = person['name']
        faces = person['faces']
        if len(faces) == 1:
            face = faces[0]
            img = Image.open(imgFile)
            coords = (
                face['boundingBoxX1'], face['boundingBoxY1'],
                face['boundingBoxX2'], face['boundingBoxY2']
            )
            drawBoundingBox(img, coords)
            img.save(imgFile)
            print(faces[0])

def cameraSnapshot(dir, seq):
    print("Taking snapshot.")
    with imageio.get_reader("<video0>") as camera:
        screenshot = camera.get_data(0)
    imgFile = constructImgFilename(dir, "output", seq)
    imageio.imwrite(imgFile, screenshot)
    displayImg(imgFile)

async def drawImg(prompt, initImage):
    params = {
        'prompt': prompt,
        'seed': 10,
        'height': 512,
        'width': 512,
        'steps': 30,
        'denoise': 0.4,
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
                    enqueueEvent("<<DrawingStarted>>")
                    response = await drawImg(editPrompt, loadLastImg())
                    imgFile = saveImg(dir, seq, base64.b64decode(response))
                    displayImg(imgFile)
                    return ("#", True)
            case 'snapshotTool':
                cameraSnapshot(dir, seq)
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
        'wakewords/annyeonghasimnikka_ko_linux_v3_0_0.ppn',
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

    recorder = PvRecorder(
        frame_length=porcupineEn.frame_length)
    recorder.start()

    dirUnique = datetime.now().strftime("%Y%m%d-%H%M%S")
    dirSeq = 0

    try:
        while not stopConvo.is_set():
            dirSeq += 1
            dir = f"convos-{dirUnique}/c{dirSeq}-{character}"
            os.makedirs(dir)
            fileSeq = 1
            messages.clear()
            initialInput = ""
            if not switchQueue.empty():
                character = switchQueue.get()
                enqueueEvent("<<ConvoStarted>>")
            else:
                global webcamFile
                webcamFile = constructImgFilename(dir, "output", fileSeq)
                listenForWakeword(recorder, porcupineEn, porcupineKo)
                waitForHumanRecognition()
                if stopConvo.is_set():
                    break
                initialInput = f"Hello, I am {human}"

            conversing = True

            while conversing:
                fileSeq += 1
                if initialInput:
                    (response, toolCalled) = await sendChat(dir, initialInput, fileSeq)
                    print("Greeting response:  " + response)
                    recordExchange(messages, initialInput, response, dir, fileSeq)
                    output = response
                    initialInput = ""
                else:
                    output = await processInput(recorder, cobra, dir, fileSeq)
                match output:
                    case "#":
                        enqueueEvent("<<ConvoStarted>>")
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
                        enqueueEvent("<<ConvoStarted>>")

    finally:
        recorder.delete()
        porcupineEn.delete()
        porcupineKo.delete()
        cobra.delete()

def uiLoop():
    global tkRoot
    global webcam
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

    webcam = Webcam(canvas)

    interval = 2000
    sleeping = True

    def clearSubtitle():
        canvas.itemconfigure(subtitle, text="")

    def wakewordHeard(event):
        nonlocal sleeping
        sleeping = False
        canvas.itemconfig(imagesOnCanvas[character], image=characterImages[character]['talking'])
        canvas.tag_raise(imagesOnCanvas[character])
        humanRecognized.clear()
        webcam.show()

    def convoStarted(event):
        webcam.hide()
        canvas.itemconfig(imagesOnCanvas[character], image=characterImages[character]['listening'])
        canvas.tag_raise(imagesOnCanvas[character])

    def inputProcessed(event):
        canvas.itemconfig(imagesOnCanvas[character], image=characterImages[character]['talking'])
        tkRoot.after(3000, clearSubtitle)

    def convoFinished(event):
        nonlocal sleeping
        sleeping = True
        global human
        human = "A Stranger"
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
        img = ImageOps.contain(img, (512, 512))
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
    tkRoot.bind("<<ConvoStarted>>", convoStarted)
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


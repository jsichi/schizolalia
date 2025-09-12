# schizolalia main

import configparser
import os
import subprocess
import sys
import struct
import wave
import threading
import time
from datetime import datetime

import pvcobra
import pvporcupine
import pveagle
from pvrecorder import PvRecorder
from pveagle import EagleProfile

from flux_led import BulbScanner, WifiLedBulb

from threading import Timer

import openai

client = openai.OpenAI(
    api_key="...",
    base_url="http://localhost:5000/v1")

import whisper

gray = (10, 10, 10)
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
magenta = (255, 0, 255)

inputFilename = "input.wav"
extraInputFilename = "extra.wav"
stopCollecting = threading.Event()
messages = []

global bulb

whisperModel = whisper.load_model('base')

def call_subprocess(script_path, *args):
    subprocess.run(["python", script_path, *args], capture_output=True, text=True, check=True)

def speakText(text):
    print("Saying:  " + text)
    call_subprocess(
        "fish-speech/tools/api_client.py",
        "--url", "http://localhost:8080/v1/tts",
        "-t", text,
        "--reference_id", "bubbles",
        "--output", "output",
        "--latency", "balanced",
        "--use_memory_cache", "on",
        "--streaming", "True")

def lightBulb(color):
    global bulb
    bulb.setRgb(*color)

def loadSpeaker(filename):
    with open(filename, 'rb') as speakerFile:
        return EagleProfile.from_bytes(speakerFile.read())
    
def createWav(filename):
    wavFile = wave.open(filename, "w")
    wavFile.setnchannels(1)
    wavFile.setsampwidth(2)
    wavFile.setframerate(16000)
    return wavFile

def collectInputChunk(recorder, cobra, wavFile):
    silenceCount = 0
    while (silenceCount < 40) or not stopCollecting.is_set():
        pcm = recorder.read()
        voiceProb = cobra.process(pcm)
        wavFile.writeframes(struct.pack("h" * len(pcm), *pcm))
        if voiceProb > 0.5:
            global stillTalking
            lightBulb(green)
            stillTalking = True
            silenceCount = 0
        else:
            silenceCount += 1
    wavFile.close()

def listenForWakeword(recorder, porcupine):
    awake = False
    while not awake:
        pcm = recorder.read()
        result = porcupine.process(pcm)
        if result >= 0:
            awake = True

def collectInitialInput(recorder, cobra, eagle, speakerNames):
    eagle.reset()
    wavFile = createWav(inputFilename)
    collecting = True
    heardVoice = False
    silenceCount = 0
    while collecting:
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

def processRestOfInput(recorder, cobra):
    global stillTalking

    while True:
        stillTalking = False
        stopCollecting.clear()
        wavFile = createWav(extraInputFilename)
        collectThread = threading.Thread(
            target=collectInputChunk, args=(recorder, cobra, wavFile))
        collectThread.start()
        whisperResult = whisperModel.transcribe(
            inputFilename, fp16=False, language='English')['text']
        print("Heard:  " + whisperResult)
        tentativeResponse = sendChat(whisperResult)
        print("Tentative response:  " + tentativeResponse)
        stopCollecting.set()
        collectThread.join()
        if stillTalking:
            print('Concatenating')
            infiles = [inputFilename, extraInputFilename]
            data = []
            for infile in infiles:
                with wave.open(infile, 'rb') as w:
                    data.append([w.getparams(), w.readframes(w.getnframes())])
            with wave.open(inputFilename, 'wb') as output:
                output.setparams(data[0][0])
                output.writeframes(data[0][1])
                output.writeframes(data[1][1])
        else:
            recordInput(messages, whisperResult)
            recordOutput(messages, tentativeResponse)
            return tentativeResponse

def processInput(recorder, cobra, eagle, speakerNames):
    if collectInitialInput(recorder, cobra, eagle, speakerNames):
        lightBulb(blue)
        return processRestOfInput(recorder, cobra)
    else:
        return ""

def sendChat(input):
    tentative = messages.copy()
    recordInput(tentative, input)
    chatResponse = client.chat.completions.create(
        model="bubbles",
        messages=tentative)
    responseText = chatResponse.choices[0].message.content
    return responseText

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

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
    
    global bulb
    bulbAddress = config['fluxled']['BulbAddress']
    bulb = WifiLedBulb(bulbAddress)
    bulb.turnOn()
    bulb.refreshState()

    recorder = PvRecorder(
        frame_length=porcupine.frame_length)
    recorder.start()

    bail = False

    try:
        while not bail:
            lightBulb(gray)
            messages.clear()
            listenForWakeword(recorder, porcupine)

            conversing = True

            while conversing:
                lightBulb(green)
                output = processInput(recorder, cobra, eagle, speakerNames)
                if output:
                    annotatedOutput = "(excited) " + output

                    bulb.setCustomPattern(
                        (magenta, blue),
                        100,
                        "gradual")
                    speakText(annotatedOutput)
                else:
                    conversing = False

    finally:
        lightBulb(black)
        recorder.delete()
        porcupine.delete()
        cobra.delete()
        eagle.delete()

main()


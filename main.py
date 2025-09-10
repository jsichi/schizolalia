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
from pvrecorder import PvRecorder

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

def lightBulb(color):
    global bulb
    bulb.setRgb(*color)

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

def collectInitialInput(recorder, cobra):
    wavFile = createWav(inputFilename)
    collecting = True
    silenceCount = -1
    while collecting:
        pcm = recorder.read()
        voiceProb = cobra.process(pcm)
        wavFile.writeframes(struct.pack("h" * len(pcm), *pcm))
        if voiceProb > 0.5:
            silenceCount = 0
        else:
            if silenceCount > -1:
                silenceCount += 1
                if silenceCount > 30:
                    lightBulb(blue)
                    wavFile.close()
                    collecting = False

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
        stopCollecting.set()
        collectThread.join()
        if stillTalking:
            print('Concatenating')
            infiles = [inputFilename, extraInputFilename]
            data = []
            for infile in infiles:
                w = wave.open(infile, 'rb')
                data.append([w.getparams(), w.readframes(w.getnframes())])
                w.close()
            output = wave.open(inputFilename, 'wb')
            output.setparams(data[0][0])
            output.writeframes(data[0][1])
            output.writeframes(data[1][1])
            output.close()
        else:
            recordInput(messages, whisperResult)
            recordOutput(messages, tentativeResponse)
            return tentativeResponse

def processInput(recorder, cobra):
    collectInitialInput(recorder, cobra)
    lightBulb(blue)
    return processRestOfInput(recorder, cobra)

def sendChat(input):
    tentative = messages
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

    porcupineKeywordPaths = ['../wakewords/Hi-Bubbles_en_linux_v3_0_0.ppn']
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
            bulb.setRgb(*gray)
            listenForWakeword(recorder, porcupine)

            bulb.setRgb(*green)
            output = processInput(recorder, cobra)
            print("Saying:  " + output)

    finally:
        bulb.setRgb(*black)
        recorder.delete()
        porcupine.delete()
        cobra.delete()

main()


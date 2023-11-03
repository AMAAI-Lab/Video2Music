# ezchord - convert complex chord names to midi notes

import sys
import math
import argparse
from enum import Enum, auto
from midiutil import MIDIFile

class Mode(Enum):
    DIM = auto()
    MIN = auto()
    MAJ = auto()
    DOM = auto()
    AUG = auto()
    SUS2 = auto()
    SUS = auto()
    FIVE = auto()

TEXT_TO_MODE = {
    "maj":  Mode.MAJ,
    "dim":  Mode.DIM,
    "o":    Mode.DIM,
    "min":  Mode.MIN,
    "m":    Mode.MIN,
    "-":    Mode.MIN,
    "aug":  Mode.AUG,
    "+":    Mode.AUG,
    "sus2":  Mode.SUS2,
    "sus":  Mode.SUS,
    "5":    Mode.FIVE,
    "five": Mode.FIVE
}

MODE_TO_SHIFT = {
    Mode.MAJ:   {3:0, 5:0},
    Mode.DOM:   {3:0, 5:0},
    Mode.DIM:   {3:-1, 5:-1},
    Mode.MIN:   {3:-1, 5:0},
    Mode.AUG:   {3:0, 5:1},
    Mode.SUS2:  {3:-2, 5:0},
    Mode.SUS:   {3:1, 5:0},
    Mode.FIVE:  {3:3, 5:0},
}

NOTE_TO_PITCH = {
    "a": 9,
    "b": 11,
    "c": 12,
    "d": 14,
    "e": 16,
    "f": 17,
    "g": 19
}

PITCH_TO_NOTE = {}

for note, pitch in NOTE_TO_PITCH.items():
    PITCH_TO_NOTE[pitch] = note

RM_TO_PITCH = {
    "vii":  11,
    "iii":  4,
    "vi":   9,
    "iv":   5,
    "ii":   2,
    "i":    0,
    "v":    7
}

ACC_TO_SHIFT = {
    "b": -1,
    "#": 1
}

SCALE_DEGREE_SHIFT = {
    1: 0,
    2: 2,
    3: 4,
    4: 5,
    5: 7,
    6: 9,
    7: 11
}

def getNumber(string):
    numStr = ""
    
    for char in string:
        if char.isdigit():
            numStr += char
    
    if len(numStr) > 0:
        return int(numStr)
    
    return

def textToPitch(text, key = "c", voice = True):
    text = text.lower()
    isLetter = text[0] in NOTE_TO_PITCH.keys()

    if isLetter:
        pitch = NOTE_TO_PITCH[text[0]]
    else:
        for rm in RM_TO_PITCH.keys():
            if rm in text:
                pitch = RM_TO_PITCH[rm] + textToPitch(key)
                isRomanNumeral = True
                break

    for i in range(1 if isLetter else 0, len(text)):
        if text[i] in ACC_TO_SHIFT.keys():
            pitch += ACC_TO_SHIFT[text[i]]   
    
    return pitch

def pitchToText(pitch):
    octave = math.floor(pitch / 12)
    pitch = pitch % 12
    pitch = pitch + (12 if pitch < 9 else 0)
    accidental = ""

    if not (pitch in PITCH_TO_NOTE.keys()):
        pitch = (pitch + 1) % 12
        pitch = pitch + (12 if pitch < 9 else 0)
        accidental = "b"
    
    return PITCH_TO_NOTE[pitch].upper() + accidental + str(octave)

def degreeToShift(deg):
    return SCALE_DEGREE_SHIFT[(deg - 1) % 7 + 1] + math.floor(deg / 8) * 12

def voice(chords):
    center = 0
    voiced_chords = []
    chord_ct = 0 
    pChord = None

    for i, currChord in enumerate(chords):

        if len(currChord) == 0:
            voiced_chords.append( [] )
            continue
        else:
            if chord_ct == 0:
                voiced_chords.append( currChord )
                chord_ct += 1
                center = currChord[1] + 3
                pChord = currChord
                continue

        prevChord = pChord
        voiced_chord = []

        for i_, currNote in enumerate(currChord):
            # Skip bass note
            if i_ == 0:
                prevNote = prevChord[0]
                if abs(currNote - prevNote) > 7:
                    if currNote < prevNote and abs(currNote + 12 - prevNote) < abs(currNote - prevNote):
                        bestVoicing = currNote + 12
                    elif currNote > prevNote and abs(currNote - 12 - prevNote) < abs(currNote - prevNote):
                        bestVoicing = currNote - 12
                else:
                    bestVoicing = currNote 

                voiced_chord.append(bestVoicing)
                continue
            
            bestNeighbor = None
            allowance = -1

            while bestNeighbor == None:
                allowance += 1
                for i__, prevNote in enumerate(prevChord):
                    if i__ == 0:
                        continue
                    
                    if (
                        abs(currNote - prevNote) % 12 == allowance
                        or abs(currNote - prevNote) % 12 == 12 - allowance
                    ):
                        bestNeighbor = prevNote
                        break

            if currNote <= bestNeighbor:
                bestVoicing = currNote + math.floor((bestNeighbor - currNote + 6) / 12) * 12
            else:
                bestVoicing = currNote + math.ceil((bestNeighbor - currNote - 6) / 12) * 12

            bestVoicing = bestVoicing if (abs(bestVoicing - center) <= 8 or allowance > 2) else currNote
            voiced_chord.append(bestVoicing)
            

        voiced_chord.sort()
        voiced_chords.append(voiced_chord)
        pChord = voiced_chord
    
    return voiced_chords

class Chord:
    def __init__(self, string):
        self.string = string
        self.degrees = {}
    
        string += " "
        self.split = []
        sect = ""

        notes = list(NOTE_TO_PITCH.keys())
        rms = list(RM_TO_PITCH.keys())
        accs = list(ACC_TO_SHIFT.keys())
        modes = list(TEXT_TO_MODE.keys())

        rootAdded = False
        modeAdded = False

        isRomanNumeral = False
        isSlashChord = False
        isMaj7 = False

        for i in range(0, len(string) - 1):
            sect += string[i]
            currChar = string[i].lower()
            nextChar = string[i+1].lower()

            rootFound = not rootAdded and (currChar in notes+rms+accs and not nextChar in rms+accs) 
            modeFound = False
            numFound = (currChar.isdigit() and not nextChar.isdigit())

            if (
                (i == len(string) - 2)
                or rootFound
                or numFound
                or nextChar == "/"
                or currChar == ")"
            ):
                if rootFound:
                    self.root = sect
                    rootAdded = True

                    isRomanNumeral = self.root in rms
                elif sect[0] == "/":
                    # case for 6/9 chords
                    if sect[1] == "9":
                        self.degrees[9] = 0
                    else:
                        isSlashChord = True
                        self.bassnote = sect[1:len(sect)]
                else:
                    if not modeAdded:
                        for mode in modes:
                            modeFound = mode in sect[0:len(mode)]
                            if modeFound:
                                self.mode = TEXT_TO_MODE[mode]
                                modeAdded = True
                                break
                    
                    if not modeAdded:
                        if not isRomanNumeral and str(getNumber(sect)) == sect:
                            self.mode = Mode.DOM
                            modeFound = True
                            modeAdded = True
                    
                    deg = getNumber(sect)
                    if deg != None:
                        shift = 0

                        for char in sect:
                            if char == "#":
                                shift += 1
                            elif char == "b":
                                shift -= 1

                        if (not modeFound) or deg % 2 == 0:
                            self.degrees[deg] = shift
                        elif deg >= 7:
                            for i in range(7, deg+1):
                                if i % 2 != 0:
                                    self.degrees[i] = shift

                self.split.append(sect)
                sect = ""

        if not modeAdded:
            # Case for minor roman numeral chords
            if self.root in rms and self.root == self.root.lower():
                self.mode = Mode.MIN
            else:
                self.mode = Mode.DOM
        
        if not isSlashChord:
            self.bassnote = self.root

        for sect in self.split:
            isMaj7 = ("maj" in sect) or isMaj7
        
        if (7 in self.degrees.keys()) and not isMaj7:
            self.degrees[7] = -1
    
    def getMIDI(self, key="c", octave=4):
        notes = {}

        notes[0] = textToPitch(self.bassnote, key) - 12

        root = textToPitch(self.root, key)
        notes[1] = root
        notes[3] = root + degreeToShift(3) + MODE_TO_SHIFT[self.mode][3]
        notes[5] = root + degreeToShift(5) + MODE_TO_SHIFT[self.mode][5]

        for deg in self.degrees.keys():
            notes[deg] = root + degreeToShift(deg) + self.degrees[deg]

        for deg in notes.keys():
            notes[deg] += 12 * octave

        return list(notes.values())

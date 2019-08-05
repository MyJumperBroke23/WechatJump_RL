import cv2 as cv
import os
import numpy as np
import shutil
import torch

def getProcessedImage(show = False):
    getImage()
    image = cv.imread('/Users/alex/PycharmProjects/Wechat_Jump_RL/state.png')
    image = removeBackgroundColor(getBackgroundColor(image), image)
    image = removeShadows(image)
    image = changeCharacterColor(image)
    image = changePlatformColor(image)
    image = image[490:, :]
    if show:
        cv.imshow('buba', image)
        cv.waitKey(5000)
    image = np.delete(image, 0, axis = 2)
    return image

def getLessProcessed(show = False):
    getImage()
    image = cv.imread('/Users/alex/PycharmProjects/Wechat_Jump_RL/state.png')
    image = changeCharacterColor(removeBackgroundColor(getBackgroundColor(image), image))
    image = image[490:, :]
    if show:
        cv.imshow('buba', image)
        cv.waitKey(99000)
    return image

def getImage():
    os.system('adb shell screencap -p /sdcard/1.png')
    os.system('adb pull /sdcard/1.png state.png')
    return cv.imread("state.png")

# Background is on a gradient, so returns the lowest and highest values
def getBackgroundColor(image):
    return image[0, 0], image[2315, 1079]


def removeBackgroundColor(rgbVal, image, show = False):
    upper, lower = rgbVal
    mask = cv.inRange(image, lower, upper)
    mask = cv.bitwise_not(mask)
    image = cv.bitwise_and(image, image, mask=mask)
    if show:
        cv.imshow('mask', mask)
        cv.waitKey(3000)
        cv.imshow('Thing', image)
        cv.waitKey(3000)
    return image

# Shadows are of color: [148 141 139]
def removeShadows(image, show = False):
    lower = np.array([138, 131, 129])
    upper = np.array([160, 160, 170])
    mask = cv.inRange(image, lower, upper)
    mask = cv.bitwise_not(mask)
    image = cv.bitwise_and(image, image, mask=mask)
    if show:
        cv.imshow('mask',mask)
        cv.waitKey(1000)
        cv.imshow('Shadowless',image)
        cv.waitKey(5000)
    return image


# Character is color [102  65  66]
def changeCharacterColor(image, show = False):
    lower = np.array([92, 55, 56])
    upper = np.array([112, 75, 76])
    mask = cv.inRange(image, lower, upper)
    image[mask > 0] = [0, 0, 254]
    if show:
        cv.imshow('Highlight',image)
        cv.waitKey(90000)
    return image


def changePlatformColor(image, show = False):
    lower = np.array([0, 0, 253])
    upper = np.array([0, 0, 255])
    mask1 = cv.inRange(image, lower, upper)
    mask2 = cv.inRange(image, np.array([0,0,0]), np.array([0,0,0])) # filter out background
    mask = cv.bitwise_or(mask1, mask2)
    mask = cv.bitwise_not(mask)
    image[mask > 0] = [0, 254, 0]
    if show:
        cv.imshow("Thing",mask)
        cv.waitKey(5000)
    return image

def putCircle(image, x, y, wait = 5000):
    cv.circle(image, (x,y), 8, (0, 0, 255), thickness=-1, lineType=cv.FILLED)
    cv.imshow("image", image)
    cv.waitKey(wait)
    return image

#image = getImage()
#changeCharacterColor(removeBackgroundColor(getBackgroundColor(image), image), True)
#print(image[1000,200])
#thing = image[1000,200] == [255,255,255]
#print(thing)s

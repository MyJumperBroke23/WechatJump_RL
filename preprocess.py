import cv2 as cv
import os
import numpy as np


# Returns image where character color is changed to red and platform color is changed to green, background is black
def getProcessedImage(show=False):
    getImage()
    image = cv.imread('/Users/alex/PycharmProjects/Wechat_Jump_RL/state.png')
    image = removeBackgroundColor(getBackgroundColor(image), image)
    image = changeCharacterColor(image)
    image = changePlatformColor(image)
    image = image[490:, :]
    if show:
        cv.imshow('buba', image)
        cv.waitKey(5000)
    image = np.delete(image, 0, axis = 2)
    return image


# Returns image where background is changed to black
def getLessProcessed(show=False):
    getImage()
    image = cv.imread('/Users/alex/PycharmProjects/Wechat_Jump_RL/state.png')
    image = changeCharacterColor(removeBackgroundColor(getBackgroundColor(image), image))
    image = image[490:, :]
    if show:
        cv.imshow('buba', image)
        cv.waitKey(99000)
    return image


# Returns image
def getImage():
    os.system('adb shell screencap -p /sdcard/1.png')
    os.system('adb pull /sdcard/1.png state.png')
    return cv.imread("state.png")


# Background is on a gradient, so returns the lowest and highest values
def getBackgroundColor(image):
    return image[0, 0], image[2315, 1079]


# Returns image with background changed to black
def removeBackgroundColor(rgbVal, image, show=False):
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


# Returns image with character changed to red
def changeCharacterColor(image, show=False):
    # Character is color [102  65  66]
    lower = np.array([92, 55, 56])
    upper = np.array([112, 75, 76])
    mask = cv.inRange(image, lower, upper)
    image[mask > 0] = [0, 0, 254]
    if show:
        cv.imshow('Highlight',image)
        cv.waitKey(90000)
    return image


# Returns image with platform changed to green
def changePlatformColor(image, show=False):
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




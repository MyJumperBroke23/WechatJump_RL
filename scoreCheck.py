'''
Score will be checked before and after every action to calculate reward
Game Over will return score of -10
Otherwise pass in previous score and return difference
'''

from preprocess import getImage
import cv2 as cv
import numpy as np


def getReward(prev_score):
    curr_score = getScore()
    #print("CURR:", curr_score, "PREV:", prev_score)
    if (curr_score == prev_score) or (curr_score == -10):
        return -10
    else:
        return curr_score - prev_score


def getScore():
    getImage()
    image = cv.imread('/Users/alex/PycharmProjects/Wechat_Jump_RL/state.png')

    # Get part of the screen where the score is
    scoreImg = image[400:490, :]

    # Get background color and remove it, change numbers to white
    upper, lower = scoreImg[0][0], scoreImg[89][1079]
    mask = cv.inRange(scoreImg, lower, upper)
    mask = cv.bitwise_not(mask)
    scoreImg[mask == 0] = [0,0,0]
    scoreImg[mask > 0] = [255,255,255]

    # Find location of numbers and append them to list
    locations = [] # List where index 0 corresponds to location where 1 is found, index 1 is where 2 is found, etc.
    for number in range(10):
        path = 'resource/numbers/' + str(number) + ".png"
        num = cv.imread(path)
        res = cv.matchTemplate(scoreImg, num, cv.TM_CCOEFF_NORMED)
        threshold = 0.99
        loc = np.where(res >= threshold)
        locations.append(loc[1].tolist())

    score = ""
    while not(isEmpty(locations)):
        row,col = find_min_idx(locations)
        score += str(row)
        del locations[row][col]

    if score == "":
        return -10
    else:
        return int(score)


def find_min_idx(l):
    currMin = None
    currMinIdx = None
    for subListNum, subList in enumerate(l):
        for elemNum, elem in enumerate(subList):
            if elem != None:
                if currMin == None or elem < currMin:
                    currMin = elem
                    currMinIdx = (subListNum, elemNum)
    return currMinIdx


def isEmpty(l):
    try:
        return all(map(isEmpty, l))
    except TypeError:
        return False

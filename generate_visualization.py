from reconstruct_robust_feature import train_X, train_Y
from reconstruct_robust_feature import reconstruct_feature as recon_robust
from reconstruct_nonrobust_feature import reconstruct_feature as recon_nonrobust
import numpy as np
import skimage
import matplotlib.pyplot as plt
from PIL import Image


label_Y = np.argmax(train_Y, axis=1)
locations = [np.where(label_Y == i)[0] for i in range(10)]


def generate_visualization(imgGridSize, digit, reconFunc):
    assert imgGridSize[0] > 0 and imgGridSize[1] > 0
    padding = 3
    imgSize = train_X.shape[1 : 3]
    totalI = imgGridSize[0] * imgSize[0] + padding * (imgGridSize[0] - 1)
    totalJ = imgGridSize[1] * imgSize[1] + padding * (imgGridSize[1] - 1)
    newImg = Image.new('RGB', (totalJ, totalI), (255, 255, 255))

    iLoc = 0
    jLoc = 0
    imgCounter = 0
    for i in range(imgGridSize[0]):
        for j in range(imgGridSize[1]):
            imgId = locations[digit][imgCounter]
            reconImg = skimage.img_as_ubyte(reconFunc(imgId).squeeze())
            pilRecon = Image.fromarray(reconImg)
            newImg.paste(pilRecon, (jLoc, iLoc))
            jLoc += padding + imgSize[0]
            imgCounter += 1
        jLoc = 0
        iLoc += padding + imgSize[1]

    return newImg


def generate_original_visualization(imgGridSize, digit):
    assert imgGridSize[0] > 0 and imgGridSize[1] > 0
    padding = 3
    imgSize = train_X.shape[1: 3]
    totalI = imgGridSize[0] * imgSize[0] + padding * (imgGridSize[0] - 1)
    totalJ = imgGridSize[1] * imgSize[1] + padding * (imgGridSize[1] - 1)
    newImg = Image.new('RGB', (totalJ, totalI), (255, 255, 255))

    iLoc = 0
    jLoc = 0
    imgCounter = 0
    for i in range(imgGridSize[0]):
        for j in range(imgGridSize[1]):
            imgId = locations[digit][imgCounter]
            reconImg = skimage.img_as_ubyte(train_X[imgId, :, :, :].squeeze())
            pilRecon = Image.fromarray(reconImg)
            newImg.paste(pilRecon, (jLoc, iLoc))
            jLoc += padding + imgSize[0]
            imgCounter += 1
        jLoc = 0
        iLoc += padding + imgSize[1]

    return newImg

def generate_robust_images():
    outputPath = 'images/robust_recon_{}.png'
    for i in range(10):
        img = generate_visualization((5, 5), i, recon_robust)
        img.save(outputPath.format(i))


def generate_nonrobust_images():
    outputPath = 'images/nonrobust_recon_{}.png'
    for i in range(10):
        img = generate_visualization((5, 5), i, recon_nonrobust)
        img.save(outputPath.format(i))

def generate_original_images():
    outputPath = 'images/original_{}.png'
    for i in range(10):
        img = generate_original_visualization((5, 5), i)
        img.save(outputPath.format(i))

def generate_markdown_table():
    originals = ['images/original_{}.png'.format(i) for i in range(10)]
    robusts = ['images/robust_recon_{}.png'.format(i) for i in range(10)]
    nonrobusts = ['images/nonrobust_recon_{}.png'.format(i) for i in range(10)]

    cellFmt = '|{}'
    sepFmt = '|:---:'
    imgFmt = '![]({})'

    headers = ['Original', 'Reconstruction (Robust)', 'Reconstruction (Nonrobust)']
    mdLines = []

    headerStr = ''.join([cellFmt.format(s) for s in headers]) + '|'
    mdLines.append(headerStr)
    sepStr = sepFmt * len(headers) + '|'
    mdLines.append(sepStr)

    lineFmt = cellFmt * 3 + '|'
    for i in range(len(originals)):
        md_img = lambda x : imgFmt.format(x)
        lineStr = lineFmt.format(md_img(originals[i]), md_img(robusts[i]), md_img(nonrobusts[i]))
        mdLines.append(lineStr)

    print('\n'.join(mdLines))

generate_markdown_table()
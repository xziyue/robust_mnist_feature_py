from reconstruct.reconstruct_robust_feature import train_X, train_Y
import numpy as np
import skimage
from util.result_loader import load_robust_output, load_nonrobust_output
from PIL import Image
import pickle


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


robustOutput = load_robust_output()
nonrobustOutput = load_nonrobust_output()

'''
with open('robust_dim_rec.bin', 'rb') as infile:
    robustDimRecOutput = pickle.load(infile)
'''

with open('../robust_morph.bin', 'rb') as infile:
    robustMorphOutput = pickle.load(infile)

with open('../nonrobust_morph.bin', 'rb') as infile:
    nonrobustMorphOutput = pickle.load(infile)

def get_nonrobust_morph(ind):
    return nonrobustMorphOutput[ind:ind+1, :, :, :]

def get_robust(ind):
    return robustOutput[ind:ind+1, :, :, :]

def get_nonrobust(ind):
    return nonrobustOutput[ind:ind+1, :, :, :]

def get_robust_dim_rec(ind):
    return robustDimRecOutput[ind:ind+1, :, :, :]

def get_robust_morph(ind):
    return robustMorphOutput[ind:ind+1, :, :, :]

def generate_robust_images():
    outputPath = '../images/robust_recon_{}.png'
    for i in range(10):
        img = generate_visualization((5, 5), i, get_robust)
        img.save(outputPath.format(i))

def generate_robust_dimrec_images():
    outputPath = '../images/robust_recon_dimrec_{}.png'
    for i in range(10):
        img = generate_visualization((5, 5), i, get_robust_dim_rec)
        img.save(outputPath.format(i))

def generate_robust_morph_images():
    outputPath = '../images/robust_recon_morph_{}.png'
    for i in range(10):
        img = generate_visualization((5, 5), i, get_robust_morph)
        img.save(outputPath.format(i))

def generate_nonrobust_morph_images():
    outputPath = '../images/nonrobust_recon_morph_{}.png'
    for i in range(10):
        img = generate_visualization((5, 5), i, get_nonrobust_morph)
        img.save(outputPath.format(i))

def generate_nonrobust_images():
    outputPath = '../images/nonrobust_recon_{}.png'
    for i in range(10):
        img = generate_visualization((5, 5), i, get_nonrobust)
        img.save(outputPath.format(i))

def generate_original_images():
    outputPath = '../images/original_{}.png'
    for i in range(10):
        img = generate_original_visualization((5, 5), i)
        img.save(outputPath.format(i))

def generate_markdown_table_1():
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


def generate_markdown_table_2():
    robusts = ['images/robust_recon_morph_{}.png'.format(i) for i in range(10)]
    robust_dimrecs = ['images/nonrobust_recon_morph_{}.png'.format(i) for i in range(10)]

    cellFmt = '|{}'
    sepFmt = '|:---:'
    imgFmt = '![]({})'

    headers = ['Denoised Robust Features', 'Denoised Nonrobust Features']
    mdLines = []

    headerStr = ''.join([cellFmt.format(s) for s in headers]) + '|'
    mdLines.append(headerStr)
    sepStr = sepFmt * len(headers) + '|'
    mdLines.append(sepStr)

    lineFmt = cellFmt * len(headers) + '|'
    for i in range(len(robusts)):
        md_img = lambda x : imgFmt.format(x)
        lineStr = lineFmt.format(md_img(robusts[i]), md_img(robust_dimrecs[i]))
        mdLines.append(lineStr)

    print('\n'.join(mdLines))

#generate_robust_dimrec_images()
#generate_robust_images()
#generate_markdown_table_2()
#generate_robust_morph_images()
#generate_robust_morph_images()
generate_nonrobust_morph_images()
import subprocess
from load_mnist import load_mnist_train_XY
import time
import re

train_X, _ = load_mnist_train_XY()

concurrency=5
readLineTimeout = 0.2

def reconstruct_robust_dataset(left=0, right=-1):
    if right == -1:
        right = train_X.shape[0]

    assert right > left

    numSamples = right - left
    taskStep =  numSamples // concurrency

    params = []

    for i in range(concurrency):
        params.append([
            i * taskStep,
            min((i + 1) * taskStep, right),
            'robust_out_{}.bin'.format(i)
        ])

    processes = []
    for i in range(concurrency):
        left, right, outname = params[i]
        left = repr(left)
        right = repr(right)
        proc = subprocess.Popen(['python3', 'recon_dataset_cli.py', 'robust', left, right, outname], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        processes.append(proc)

    startTime = time.time()
    time.sleep(1.0)

    while True:
        nowTime = time.time()
        finishedSum = 0

        exitCount = 0
        exitBits = [False] * concurrency
        for i in range(concurrency):
            if processes[i].poll() is not None:
                exitCount += 1
                exitBits[i] = True
                finishedSum += params[i][1] - params[i][0]
        if exitCount == concurrency:
            break

        for i in range(concurrency):
            if exitBits[i]:
                continue

            lastLine = ''
            readLinesStart = time.time()
            while True:
                readLineNow = time.time()
                if readLineNow - readLinesStart >= readLineTimeout:
                    break
                line = processes[i].stdout.readline()
                if not line:
                    break
                else:
                    lastLine =line

            if len(lastLine) > 0:
                match = re.match(r'(.*?)\/(.*?)', lastLine.decode('utf8'))
                if match is not None:
                    try:
                        doneCount = int(match.group(1))
                    except Exception:
                        doneCount = 0
                    finishedSum += doneCount

        timeUsed = nowTime - startTime
        numSamplePerSecond = int(finishedSum / timeUsed)
        if numSamplePerSecond == 0:
            print('{}/{} est. time left: inf'.format(finishedSum, numSamples))
        else:
            timeLeft = (numSamples - finishedSum) / numSamplePerSecond
            print('{}/{} est. time left: {:.1f}s'.format(finishedSum, numSamples, timeLeft))

        time.sleep(2.0)


reconstruct_robust_dataset()
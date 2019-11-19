import subprocess
from util.load_mnist import load_mnist_train_XY
import time
import re
import threading
from queue import Queue

train_X, _ = load_mnist_train_XY()

concurrency=5
readLineTimeout = 0.1

def enqueue_output(id, fileobj, queue):
    for line in iter(fileobj.readline, b''):
        queue.put((id, line))


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

    evtQueue = Queue()
    processes = []
    threads = []
    for i in range(concurrency):
        left, right, outname = params[i]
        left = repr(left)
        right = repr(right)
        proc = subprocess.Popen(['python3', 'recon_dataset_cli.py', 'robust', left, right, outname], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        processes.append(proc)
        # create a polling thread for each process
        t = threading.Thread(target=enqueue_output, args=(i, proc.stdout, evtQueue))
        t.daemon = True
        t.start()

    startTime = time.time()
    time.sleep(1.0)

    lastestCounter = [0] * concurrency

    while True:
        nowTime = time.time()

        exitCount = 0
        exitBits = [False] * concurrency
        for i in range(concurrency):
            if processes[i].poll() is not None:
                exitCount += 1
                exitBits[i] = True
                lastestCounter[i] = params[i][1] - params[i][0]
        if exitCount == concurrency:
            break

        for i in range(concurrency):
            if exitBits[i]:
                continue

            # update latest counter
            evts = []
            while True:
                try:
                    evt = evtQueue.get(timeout=readLineTimeout)
                    evts.append(evt)
                except Exception:
                    break

            for id, msg in evts:
                match = re.match(r'(.*?)\/(.*?)', msg.decode('utf8'))
                if match is not None:
                    try:
                        num = int(match.group(1))
                        lastestCounter[id] = num
                    except Exception:
                        pass

        finishedSum = sum(lastestCounter)
        timeUsed = nowTime - startTime
        numSamplePerSecond = finishedSum / timeUsed

        if abs(numSamplePerSecond) < 1.0e-5:
            print('{}/{} est. time left: inf'.format(finishedSum, numSamples))
        else:
            timeLeft = (numSamples - finishedSum) / numSamplePerSecond
            print('{}/{} est. time left: {:.1f}s'.format(finishedSum, numSamples, timeLeft))

        time.sleep(2.0)

    for t in threads:
        t.terminate()

def reconstruct_nonrobust_dataset(left=0, right=-1):
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
            'nonrobust_out_{}.bin'.format(i)
        ])

    evtQueue = Queue()
    processes = []
    threads = []
    for i in range(concurrency):
        left, right, outname = params[i]
        left = repr(left)
        right = repr(right)
        proc = subprocess.Popen(['python3', 'recon_dataset_cli.py', 'nonrobust', left, right, outname], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        processes.append(proc)
        # create a polling thread for each process
        t = threading.Thread(target=enqueue_output, args=(i, proc.stdout, evtQueue))
        t.daemon = True
        t.start()

    startTime = time.time()
    time.sleep(1.0)

    lastestCounter = [0] * concurrency

    while True:
        nowTime = time.time()

        exitCount = 0
        exitBits = [False] * concurrency
        for i in range(concurrency):
            if processes[i].poll() is not None:
                exitCount += 1
                exitBits[i] = True
                lastestCounter[i] = params[i][1] - params[i][0]
        if exitCount == concurrency:
            break

        for i in range(concurrency):
            if exitBits[i]:
                continue

            # update latest counter
            evts = []
            while True:
                try:
                    evt = evtQueue.get(timeout=readLineTimeout)
                    evts.append(evt)
                except Exception:
                    break

            for id, msg in evts:
                match = re.match(r'(.*?)\/(.*?)', msg.decode('utf8'))
                if match is not None:
                    try:
                        num = int(match.group(1))
                        lastestCounter[id] = num
                    except Exception:
                        pass

        finishedSum = sum(lastestCounter)
        timeUsed = nowTime - startTime
        numSamplePerSecond = finishedSum / timeUsed

        if abs(numSamplePerSecond) < 1.0e-5:
            print('{}/{} est. time left: inf'.format(finishedSum, numSamples))
        else:
            timeLeft = (numSamples - finishedSum) / numSamplePerSecond
            print('{}/{} est. time left: {:.1f}s'.format(finishedSum, numSamples, timeLeft))

        time.sleep(2.0)

    for t in threads:
        t.terminate()

reconstruct_nonrobust_dataset()


'''
# old version
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

        sys.stdout.flush()
        time.sleep(2.0)

'''
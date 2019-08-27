import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

def get_vertical_line_grid(imgShape, numLines, lineWidth, lineColor):
    linePos = np.round(np.linspace(0, imgShape[1] - lineWidth, numLines)).astype(np.int)
    result = np.zeros(imgShape, np.float32)

    for pos in linePos:
        result[:, pos : pos + lineWidth].fill(lineColor)

    return result

def get_horizontal_line_grid(imgShape, numLines, lineWidth, lineColor):
    linePos = np.round(np.linspace(0, imgShape[0] - lineWidth, numLines)).astype(np.int)
    result = np.zeros(imgShape, np.float32)

    for pos in linePos:
        result[pos: pos + lineWidth, :].fill(lineColor)

    return result


def vertical_lines(image, numLines, lineWidth, lineColor):
    lineGrid = get_vertical_line_grid(image.shape, numLines, lineWidth, lineColor)
    result = np.clip(image + lineGrid, 0.0, 1.0)
    return result


def horizontal_lines(image, numLines, lineWidth, lineColor):
    lineGrid = get_horizontal_line_grid(image.shape, numLines, lineWidth, lineColor)
    result = np.clip(image + lineGrid, 0.0, 1.0)
    return result


def gaussian_noise(image):
    noise = np.abs(np.random.normal(0.0, 0.2, image.size))
    noise = noise.reshape(image.shape).astype(image.dtype)
    result = np.clip(image + noise, 0.0, 1.0)
    return result


def random_removal(image):
    originalShape = image.shape
    image = image.squeeze()
    imageContentPos = np.where(image > 0.6)

    indices = np.arange(0, imageContentPos[0].size)
    # randomly remove some content
    numIndicesToRemove = int(np.round(0.1 * indices.size))
    indexRemovePos = np.random.randint(0, indices.size - numIndicesToRemove - 1)

    result = np.copy(image)

    xPos = imageContentPos[0][indexRemovePos : indexRemovePos + numIndicesToRemove]
    yPos = imageContentPos[1][indexRemovePos : indexRemovePos + numIndicesToRemove]
    result[xPos, yPos] = 0.0

    result = result.reshape(originalShape)

    return result

class PerturbationMan:

    @staticmethod
    def _split_task(lst, n):
        k, m = divmod(len(lst), n)
        return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    @staticmethod
    def _compute_perturbated_data(X, indices, perts):
        result = []

        for index in indices:
            one = [pert(X[index, ...]) for pert in perts]
            one.append(X[index, ...])  # add original vector
            one = np.asarray(one)
            result.append(one)

        return result

    def __init__(self, X, Y, perts, concurrency=10):
        assert len(X.shape) == 4
        self.data = []
        self.Y = Y
        self.tensorX = []
        self.tensorY = None
        self.concurrency = concurrency

        assert X.dtype == np.float32

        taskIndices = list(self._split_task(list(range(X.shape[0])), self.concurrency))
        tasks = []
        for i in range(self.concurrency):
            thisTask = [
                X,
                taskIndices[i],
                perts
            ]
            tasks.append(thisTask)

        pool = multiprocessing.Pool(self.concurrency)
        parallelResult = pool.starmap(self._compute_perturbated_data, tasks, chunksize=1000)

        allPert = np.concatenate(parallelResult, axis=0)

        for i in range(len(perts) + 1):
            self.data.append(allPert[:, i, :, :, :])

    def show_content(self, numPerGroup = 1):
        for i in range(self.get_num_groups()):
            for j in range(numPerGroup):
                plt.imshow(self.get_perturbated_data(i)[j, :, :, 0])
                plt.show()

    def create_tensor(self):
        import tensorflow as tf
        self.tensorX.clear()

        for i in range(len(self.data)):
            tensor = tf.convert_to_tensor(self.data[i])
            self.tensorX.append(tensor)

        self.tensorY = tf.convert_to_tensor(self.Y)


    def get_tensor_slice(self, groupId, left, right):
        group = self.tensorX[groupId]
        x = group[left : right, :, :, :]
        y = self.tensorY[left : right, :]
        return x, y

    def get_tensor_x(self, groupId, index):
        return self.tensorX[groupId][index, :, :, :]


    def get_num_groups(self):
        return len(self.data)

    def get_perturbated_data(self, i):
        return self.data[i]
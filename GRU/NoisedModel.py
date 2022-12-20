import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
import os


class NoiseModel:
    trainingEpochsCount = 1
    modelFileLocation = "NoisedModel"
    model = None
    trainSamples, trainLabels = None, None
    testSamples, testLabels = None, None

    def __init__(self):

        if os.path.exists(self.modelFileLocation):
            self.model = tf.keras.models.load_model(self.modelFileLocation)
            self.model.summary()
        else:
            self.model = self.defineModel()
            (trainSamples, trainLabels), (testSamples, testLabels) = self.getSamples()
            self.model.fit(trainSamples, trainLabels, epochs=self.trainingEpochsCount, validation_data=(testSamples, testLabels))
            self.model.save(self.modelFileLocation)

    def defineModel(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Reshape((32, 96), input_shape=(32, 32, 3)),
            tf.keras.layers.GRU(256, input_shape=(32, 96), return_sequences=True),
            tf.keras.layers.GRU(128, return_sequences=True),
            tf.keras.layers.GRU(64, return_sequences=True),
            tf.keras.layers.GRU(32, return_sequences=True),
            tf.keras.layers.GRU(16, return_sequences=True),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation='sigmoid'),
        ])
        model.summary()
        model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def makeNoisedPredictions(self, filedir):
        for filename in os.listdir(filedir):
            f = os.path.join(filedir, filename)
            if not os.path.isfile(f):
                continue

            image = NoiseModel.loadImage(f)
            predictions = self.model.predict(image)
            score = tf.nn.softmax(predictions[0])
            class_labels = ["cat", "dog"]
            print("This image most likely belongs to {} with a {:.2f} percent confidence."
                  .format(class_labels[np.argmax(score)], 100 * np.max(score)))

    @staticmethod
    def getSamples():
        trainingSamples, trainingLabels, testingSamples, testingLabels = NoiseModel.loadDataset()
        trainingSamples, testingSamples = NoiseModel.preparePixels(trainingSamples, testingSamples)
        return (trainingSamples, trainingLabels), (testingSamples, testingLabels)

    @staticmethod
    def preparePixels(trainSamples, testSamples):
        shift = 25.0
        trainNormed = ((trainSamples + shift) % 256).astype("float32") / 255.0
        testNormed = ((testSamples + shift) % 256).astype("float32") / 255.0
        return trainNormed, testNormed

    @staticmethod
    def loadImage(filepath):
        img = tf.io.read_file(filepath)
        tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32)
        tensor = tf.image.resize(tensor, [32, 32])
        tensor = np.expand_dims(tensor, axis=0)
        return tensor

    @staticmethod
    def loadDataset():
        (trainSamples, trainLabels), (testSamples, testLabels) = tf.keras.datasets.cifar10.load_data()
        trainSamples, trainLabels = NoiseModel.removeNonCatsDogs(trainSamples, trainLabels)
        testSamples, testLabels = NoiseModel.removeNonCatsDogs(testSamples, testLabels)
        trainLabels = to_categorical(trainLabels)
        testLabels = to_categorical(testLabels)
        return trainSamples, trainLabels, testSamples, testLabels

    @staticmethod
    def removeNonCatsDogs(samples, labels):
        filteredSamples = []
        filteredLabels = []
        for sample, label in list(zip(samples, labels)):
            if label[0] != 3 and label[0] != 5:
                continue
            filteredSamples.append(sample)
            filteredLabels.append([0] if label[0] == 3 else [1])
        return np.array(filteredSamples), np.array(filteredLabels)


import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
import os


class MainModel:
    trainingEpochsCount = 1
    modelFileLocation = "MainModel"
    model = None
    trainSamples, trainLabels = None, None
    testSamples, testLabels = None, None

    def __init__(self):
        if os.path.exists(self.modelFileLocation):
            self.model = tf.keras.models.load_model(self.modelFileLocation)
            self.model.summary()
        else:
            self.model = self.defineModel()
            self.model.summary()
            (self.trainSamples, self.trainLabels), (self.testSamples, self.testLabels) = self.getSamples()
            self.model.fit(self.trainSamples, self.trainLabels, epochs=self.trainingEpochsCount, validation_data=(self.testSamples, self.testLabels))
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
        model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def makePredictions(self, filedir):
        for filename in os.listdir(filedir):
            f = os.path.join(filedir, filename)
            if not os.path.isfile(f):
                continue

            image = MainModel.loadImage(f)
            predictions = self.model.predict(image)
            score = tf.nn.softmax(predictions[0])
            class_labels = ["cat", "dog"]
            print("This image most likely belongs to {} with a {:.2f} percent confidence."
                  .format(class_labels[np.argmax(score)], 100 * np.max(score)))

    @staticmethod
    def getSamples():
        trainingSamples, trainingLabels, testingSamples, testingLabels = MainModel.loadDataset()
        trainingSamples, testingSamples = MainModel.preparePixels(trainingSamples, testingSamples)
        return (trainingSamples, trainingLabels), (testingSamples, testingLabels)

    @staticmethod
    def preparePixels(trainSamples, testSamples):
        trainNormed = trainSamples.astype("float32") / 255.0
        testNormed = testSamples.astype("float32") / 255.0
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
        trainSamples, trainLabels = MainModel.removeNonCatsDogs(trainSamples, trainLabels)
        testSamples, testLabels = MainModel.removeNonCatsDogs(testSamples, testLabels)
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


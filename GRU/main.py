import sys
from MainModel import MainModel
from NoisedModel import NoiseModel

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Bad input arguments", file=sys.stderr)
        exit(1)

    if sys.argv[1] == "Usuall":
        model = MainModel()
        model.makePredictions("Images")

    if sys.argv[1] == "WithNoise":
        model = NoiseModel()
        model.makeNoisedPredictions("Images")


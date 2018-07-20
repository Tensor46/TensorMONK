""" TensorMONK's :: NeuralEssentials                                         """


class BaseModel:
    netEmbedding = None
    netLoss = None
    netAdversarial = None
    meterTop1 = []
    meterTop5 = []
    meterLoss = []
    meterTeAC = []
    meterSpeed = []
    meterIterations = 0
    fileName = None
    isCUDA = False

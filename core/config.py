from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.IMAGE = edict()
__C.IMAGE.HEIGHT = 512
__C.IMAGE.WIDTH = 512
__C.IMAGE.CHANNELS = 3

__C.BACKBONE = edict()
__C.BACKBONE.DATASET = edict()
__C.BACKBONE.DATASET.BASE_PATH = "./datasets/transfer/train"
__C.BACKBONE.DATASET.VAL_BASE_PATH = "./datasets/transfer/validation"
__C.BACKBONE.MODEL = edict()
__C.BACKBONE.MODEL.NAME = "DarkNet53"
__C.BACKBONE.MODEL.CLASSES = 50
__C.BACKBONE.TRAINING = edict()
__C.BACKBONE.TRAINING.EPOCHS = 100
__C.BACKBONE.TRAINING.BATCH_SIZE = 4
__C.BACKBONE.TRAINING.SAVE_PATH = "./save/backbone"
__C.BACKBONE.OPTIMIZER = edict()
__C.BACKBONE.OPTIMIZER.NAME = "Adam"
__C.BACKBONE.OPTIMIZER.LEARNING_RATE = 0.001
__C.BACKBONE.LEARNING_RATE_SCHEDULE = edict()
__C.BACKBONE.LEARNING_RATE_SCHEDULE.EPOCHS = [20, 40, 60]
__C.BACKBONE.LEARNING_RATE_SCHEDULE.SCALES = [0.1, 0.1, 0.1]
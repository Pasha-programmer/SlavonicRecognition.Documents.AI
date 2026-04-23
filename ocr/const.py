DIR = './' # work directory
WEIGHTS_PATH =  "./ocr_transformer_4h2l_simple_conv_64x256.pt"

### MODEL ###
HIDDEN = 512
ENC_LAYERS = 2
DEC_LAYERS = 2
N_HEADS = 4


ALPHABET = [
    'Ⰰ', 'Ⰱ', 'Ⰲ', 'Ⰳ', 'Ⰴ', 'Ⰵ', 'Ⰶ', 'Ⰷ',
    'Ⰸ', 'Ⰹ', 'Ⰺ', 'Ⰻ', 'Ⰼ', 'Ⰽ', 'Ⰾ', 'Ⰿ',
    'Ⱀ', 'Ⱁ', 'Ⱂ', 'Ⱃ', 'Ⱄ', 'Ⱅ', 'Ⱆ', 'Ⱇ', 'Ⱈ',
    'Ⱉ', 'Ⱊ', 'Ⱋ', 'Ⱌ', 'Ⱍ', 'Ⱎ', 'Ⱏ', 'ⰟⰊ',
    'Ⱐ', 'Ⱑ', 'Ⱒ', 'Ⱓ', 'Ⱔ', 'Ⱖ', 'Ⱗ', 'Ⱘ', 'Ⱙ', 'Ⱚ', 'Ⱛ']

### TRAINING ###
DROPOUT = 0.2
DEVICE = 'cpu:0' # or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### INPUT IMAGE PARAMETERS ###
WIDTH = 256
HEIGHT = 64
CHANNELS = 1 # 3 channels if model1
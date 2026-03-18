from help import *
import sys

def start_recognition(image: Image):
    if len(sys.argv) != 3:
        print("Usage: python main.py <directory_path> <prediction_directory_path>")
        sys.exit(1)

    # Создание DataFrame с входными числами

    imagesDirectory = sys.argv[1]
    resultDirectory = sys.argv[2]
    resultFile = resultDirectory + 'predictions.tsv'

    # load weights
    model = TransformerModel(len(ALPHABET), hidden=HIDDEN, enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,
                            nhead=N_HEADS, dropout=DROPOUT).to(DEVICE)

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))

    preds = prediction(model, image, char2idx, idx2char)

    f = open(resultFile, 'w', encoding="utf-8")
    f.write('filename\tprediction\n')
    for item in preds.items():
        f.write(item[0]+'\t'+item[1]+'\n')
    f.close()

    print(resultFile)
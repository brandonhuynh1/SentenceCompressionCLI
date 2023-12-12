import argparse
import torch
import torch.nn as nn
import pickle
from nltk import word_tokenize


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, dropout_p=0.1):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=3, dropout=dropout_p)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        # shape = (row x column), N = batch size
        # input shape: (seq_length, N)
        embedded = self.dropout(self.embedding(input))
        # print("encoder input shape", input.shape)
        # print("nn.embedding shape", embedded.shape)
        # print("first col of embedding:", embedded[:, 1])
        # print("first row of input:", input[0])
        # embeddings = []
        # unknowns = []
        # for i in input:
        #     if i in word2Vec_model:
        #         embeddings.append(word2Vec_model.get_vector(i))
        #     else:
        #         unknowns.append(i)
        # tensor = self.dropout(torch.tensor(embeddings))
        # embedded_tensor = tensor.unsqueeze(1)
        # # print(unknowns)
        # print("our embedding shape", embedded_tensor.shape)
        # embeded shape: (seq_length, N, embedding_size)
        # hidden,cell is the context vector of the encoder
        outputs, (hidden, cell) = self.lstm(embedded)
        # when we call self.lstm(embedded) it runs the lstm for every token embedding in
        # the seq_length
        return hidden, cell


class DecoderLSTM(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size,
                 dropout_p):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=3, dropout=dropout_p)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, hidden, cell):
        # shape of x (N) but we want (1,N)
        # print("translated input to decoder", translateTensor(input))
        input = input.unsqueeze(0)
        # print("decoder input", input.shape)
        embedding = self.dropout(self.embedding(input))
        # print("decoder embedding", embedding.shape)

        # embedding shape: (1, N, embedding_size)
        outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        # output is the output word
        # print("decoder LSTM output shape", outputs.shape)
        # shape of outputs(1,N,hidden_size)
        # print("decoder self.fc input shape:", outputs.squeeze(0).shape)
        predictions = self.fc(outputs.squeeze(0))
        # print("decoder nn.Linear", predictions.shape)
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sigmoid = nn.Sigmoid()

    def forward(self, source, target, teacher_force_ratio=0.5):
        '''source is the unCompressed, target is the golden ratio'''

        batch_size = source.shape[1]
        # (trg_len, N)
        target_len = target.shape[0]
        target_vocab_size = self.decoder.output_size
        target_vocab_size = 1

        outputs = torch.zeros(target_len, batch_size,
                              target_vocab_size).to(device)

        # print("seq2seq source shape:", source.shape)
        hidden, cell = self.encoder(source)

        # grab start token
        # print("seq2seq target:", target)
        input = target[0, :]

        for t in range(1, target_len):
            # print("seq2seq input shape:", input.shape)
            output, hidden, cell = self.decoder(input, hidden, cell)
            # print("seq2seq decoder output:", output.shape)
            # print("best_guess:", output.softmax(1))

            outputs[t] = output
            input = target[t]
            # output size is (N, english_vocab_size)
            # best_guess = output.argmax(1)
            # print("seq2seq best_guess", best_guess)

            # input = target[t] if random.random() < teacher_force_ratio else best_guess
            # print(TRG.vocab.itos[input])
        output_probs = self.sigmoid(outputs)
        # threshold = 0.5
        # binary_decis = (output_probs > threshold).float()
        # print("decoder outputs shape:", outputs.shape)
        # print("binary_decis:", binary_decis)

        return output_probs


def tokenize_compressed(text):
    return word_tokenize(text)


def tokenize_uncompressed(text):
    return word_tokenize(text)[::-1]


def encoding(array):
    return array


def translate_sentence(uncompSentence: str, model, ratio: int):
    '''This function takes in one sentence -> output of model'''
    uncom_tokenized = SRC.tokenize(uncompSentence)
    uncom_indexed = SRC.process([uncom_tokenized])
    uncom_tensor = torch.tensor(uncom_indexed).to(device)

    words = [SRC.vocab.itos[idx] for idx in uncom_indexed.squeeze(1).tolist()]
    # print(words)

    gold_tokenized = TRG.tokenize(uncompSentence)
    gold_indexed = TRG.process([gold_tokenized])
    words = [TRG.vocab.itos[idx] for idx in gold_indexed.squeeze(1).tolist()]
    gold_tensor = torch.tensor(gold_indexed).to(device)
    # gold_tensor = gold_indexed.clone().detach().requires_grad_(True)
    # gold_tensor = gold_indexed.clone().detach()
    # print(words)

    with torch.no_grad():
        model.eval()
        output = model(uncom_tensor, gold_tensor, ratio)

    # print(output.shape)
    # print(output)
    output_dim = output.shape[-1]
    # print(output_dim)

    outputEmbedding = output[1:].view(-1, output_dim)
    # print(outputEmbedding.shape)
    # print(outputEmbedding.tolist())
    uncompSentence = word_tokenize(uncompSentence)
    compressed = []
    for ind, val in enumerate(outputEmbedding.tolist()):
        val = val[0]
        if (val < 0.5):
            compressed.append(uncompSentence[ind])
    # indices = outputEmbedding.argmax(dim = 1).tolist()
    # compressed = []
    # words = [TRG.vocab.itos[idx] for idx in indices]
    # print(words)

    return compressed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sentence Compression CLI Tool')
    parser.add_argument('--sentence', type=str, help='Input sentence for compression')
    parser.add_argument('--file', type=str, help='Input file with one sentence per line')

    args = parser.parse_args()

    if not args.sentence and not args.file:
        parser.error('Please provide either --sentence or --file argument')

    # Load in pickled tokenizers
    with open("./new_SRC_field.pickle", "rb") as file:
        SRC = pickle.load(file)

    with open("./new_TRG_field.pickle", "rb") as file:
        TRG = pickle.load(file)

    # Set device to run model on
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Variables needed to specify loaded model
    # in tutorial this was length of vocab of input
    input_size_encoder = len(SRC.vocab)
    # in tutorial this was length of vocab of output
    output_size_decoder = len(TRG.vocab)
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    hidden_size = 512
    num_layers = 2
    enc_dropout = 0.2
    dec_dropout = 0.2

    encoder_net = EncoderLSTM(input_size_encoder, encoder_embedding_size, hidden_size,
                              enc_dropout).to(device)
    decoder_net = DecoderLSTM(output_size_decoder, decoder_embedding_size, hidden_size,
                              dec_dropout).to(device)

    loaded_model = Seq2Seq(encoder_net, decoder_net)
    loaded_model.load_state_dict(torch.load('./new-tut1-model.pt', map_location=torch.device("mps")))
    loaded_model.to(device)

    # uncompressed_sentence = "Serge Ibaka -- the Oklahoma City Thunder forward who was born in the Congo but played in Spain -- has been granted Spanish citizenship and will play for the country in EuroBasket this summer, the event where spots in the 2012 Olympics will be decided."
    # compressed_sentence = translate_sentence(
    #     uncompressed_sentence, loaded_model, 1)

    # # compressed_sentence.remove(
    # #     "UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor). gold_tensor = torch.tensor(gold_indexed).to(device)")

    # print(compressed_sentence)

    # Process input sentence or file
    if args.sentence:
        compressed_sentence = translate_sentence(args.sentence, loaded_model, 1)
        print(compressed_sentence)
    elif args.file:
        with open(args.file, 'r') as file:
            for line in file:
                compressed_sentence = translate_sentence(line.strip(), loaded_model, 1)
                comperssed_sentence = " ".join(compressed_sentence)
                print(compressed_sentence)

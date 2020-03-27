import torch
import torchtext
import time
from torchtext.data.utils import get_tokenizer
from torch.utils.tensorboard import SummaryWriter
import math
import torch.nn as nn
import torch.nn.functional as F

from models import TransformerModel, PositionalEncoding


import os
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from tqdm import trange
import os

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('Reading Arguments')

parser = argparse.ArgumentParser()

# Preprocessing ---------------------------------------------------------------
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_length', type=int, default=20)
parser.add_argument('--buffer_size', type=int, default=2000)

# Model -----------------------------------------------------------------------
parser.add_argument('--emsize', type=int, default=200)
parser.add_argument('--nhid', type=int, default=200)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--nhead', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.2)

# Training --------------------------------------------------------------------
parser.add_argument('--epochs', type=int, default=3)

# Saving / Logging ------------------------------------------------------------
parser.add_argument('--log_dir', type=str, default='./logs/')
parser.add_argument("--log_freq", type=int, default=2)
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
parser.add_argument('--checkpoint_freq', type=int, default=2)
parser.add_argument('--extension', type=str, default=None)

cfg = parser.parse_args()

TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

bptt = 35


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


ntokens = len(TEXT.vocab.stoi)  # the size of vocabulary
emsize = 200  # embedding dimension
nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value


model = TransformerModel(ntokens, cfg.emsize, cfg.nhead, cfg.nhid,
                         cfg.nlayers, cfg.dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def build_writers():

    logger.info('Initializing Build Writers')

    if not Path(cfg.checkpoint_dir).is_dir():
        os.mkdir(cfg.checkpoint_dir)

    if not Path(cfg.log_dir).is_dir():
        os.mkdir(cfg.log_dir)

    if cfg.extension is None:
        cfg.extension = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    log_path = Path(cfg.log_dir)
    return SummaryWriter(log_path)


def log_scalar(name, scalar, n_iter):
    writer.add_scalar(name, scalar, n_iter)


def train():
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    logger.info('Training')
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time

            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | '
                        'lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                            epoch, batch, len(
                                train_data) // bptt, scheduler.get_lr()[0],
                            elapsed * 1000 / log_interval,
                            cur_loss, math.exp(cur_loss)))

            total_loss = 0

            start_time = time.time()


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


best_val_loss = float("inf")
best_model = None

writer = build_writers()

for epoch in range(1, cfg.epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    log_scalar('loss', val_loss, epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
    scheduler.step()

test_loss = evaluate(best_model, test_data)

print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

#torch.save(best_model, (cfg.checkpoint_dir + cfg.extension + '.pt'))

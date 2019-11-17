import argparse, random, gc, time
from os import path

import numpy as np
import torch
import torch.optim as optim

from utility import *

from PyCoreNLP import PyCoreNLP
from mylog import mylog
from options_process import optionsLoader
from data_process import *
from vocabulary import Vocabulary
from layers.framework import framework


seed = 19940609

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

LOG = mylog(reset = True)

def train(config):
    # Load Options
    options = optionsLoader(LOG, config.optionsFrame, disp=True)

    # Build Vocabulary
    Vocab = loadFromPKL('vocab/newData.Vocab')

    # Load data
    datasets = dataLoader(LOG, options['dataset'], Vocab)

    # Embedding Matrix for the model
    if options['network']['type'] == 'LSTM2_MeanDiff_FlatParse':
        emb_init = np.concatenate([random_weights(2 + options['network']['n_nt'],options['network']['Embedding']['n_dim'], 0.01), Vocab.i2e], axis = 0)
    elif options['network']['type'] == 'LSTM2_MeanDiff_deRNNG':
        emb_init = np.concatenate([random_weights(3, options['network']['Embedding']['n_dim'], 0.01), Vocab.i2e], axis = 0)
    else:
        emb_init = Vocab.i2e

    net = framework(options, LOG, emb_tok_init=torch.from_numpy(emb_init))

    if torch.cuda.is_available():
        LOG.log('Using Device: %s' % torch.cuda.get_device_name(torch.cuda.current_device()))
        net = net.cuda()

    print(net)

    if (options['training']['optimizer']['type'] == "Adam"):
        optimizer = optim.Adam(net.parameters(), **options['training']['optimizer']['params'])


    startEpoch = 0
    Q = []
    best_vloss = 1e99
    use_earlyStop = options['training']['stopConditions']['earlyStopping']
    if use_earlyStop:
        reduce_counter = 0
        stop_counter = 0
        flag = False

    for epoch_idx in range(startEpoch, options['training']['stopConditions']['max_epoch']):
        LOG.log('Batch Shuffle')
        datasets.batchShuffle('train')
        for batch_idx in range(datasets.Parts['train'].n_batches()):
            if ((batch_idx + 1) % 10000 == 0):
                gc.collect()
            start_time = time.time()
            source, target, sfeat, rfeat = datasets.get_Kth_Batch(batch_idx, 'train')

            # Updating
            loss = net.getLoss(source, target, sfeat, rfeat)

            Q.append(float(loss))
            if len(Q) > 200:
                Q.pop(0)
            loss_avg = sum(Q) / len(Q)

            optimizer.zero_grad()
            loss.backward()

            for p in net.parameters():
                p.grad.data.clamp_(-5, 5)

            optimizer.step()

            LOG.log('Epoch %3d, Batch %6d, Loss %11.8f, Average Loss %11.8f, Time %11.8f' % (
            epoch_idx + 1, batch_idx + 1, loss, loss_avg, time.time() - start_time))
            loss = None

            # Checkpoints
            idx = epoch_idx * datasets.Parts['train'].n_batches() + batch_idx + 1
            if (idx >= options['training']['checkingPoints']['checkMin']) and (
                    idx % options['training']['checkingPoints']['checkFreq'] == 0):
                vloss = 0
                for bid in range(datasets.Parts['valid'].n_batches()):
                    source, target, sfeat, rfeat = datasets.get_Kth_Batch(bid, 'valid')
                    vloss += float(net.getLoss(source, target, sfeat, rfeat))
                vloss /= datasets.Parts['valid'].n_batches()

                is_best = vloss < best_vloss
                best_vloss = min(vloss, best_vloss)
                save_check_point({
                    'epoch': epoch_idx + 1,
                    'options': options,
                    'state_dict': net.state_dict(),
                    'best_vloss': best_vloss,
                    'optimizer': optimizer.state_dict()},
                    is_best,
                    fileName='./model/checkpoint_Epoch' + str(epoch_idx + 1) + '_Batch' + str(batch_idx) + '.pth.tar'
                )
                LOG.log('CheckPoint: Validation Loss %11.8f, Best Loss %11.8f' % (vloss, best_vloss))
                if (use_earlyStop):
                    if is_best:
                        reduce_counter = 0
                        stop_counter = 0
                    else:
                        reduce_counter += options['training']['checkingPoints']['checkFreq']
                        stop_counter += options['training']['checkingPoints']['checkFreq']
                        if stop_counter >= options['training']['stopConditions']['earlyStopping_bound']:
                            flag = True
                            LOG.log('EarlyStop Here')
                            break
                        if reduce_counter >= options['training']['stopConditions']['rateReduce_bound']:
                            reduce_counter = 0
                            options['training']['optimizer']['params']['lr'] *= 0.5
                            if (options['training']['optimizer']['type'] == "Adam"):
                                optimizer = optim.Adam(net.parameters(), **options['training']['optimizer']['params'])
                            LOG.log(
                                'Reduce Learning Rate to %11.8f' % (options['training']['optimizer']['params']['lr']))
                vloss = None

        if options['training']['checkingPoints']['everyEpoch']:
            save_check_point({
                'epoch': epoch_idx + 1,
                'options': options,
                'state_dict': net.state_dict(),
                'best_vloss': best_vloss,
                'optimizer': optimizer.state_dict()},
                False,
                fileName='./model/checkpoint_Epoch' + str(epoch_idx + 1) + '.pth.tar'
            )
            LOG.log('Epoch Summary: Best Loss %11.8f' % (best_vloss))

        gc.collect()

        if (use_earlyStop and flag):
            break

def test(config):
    options = optionsLoader(LOG, config.optionsFrame, disp=True)
    Vocab = loadFromPKL("vocab/"+config.data+".Vocab")
    Best_Model = torch.load("model/"+config.model + "_" + config.data + "/model_best.pth.tar")

    if options['network']['type'] == 'LSTM2_MeanDiff_FlatParse':
        emb_init = np.concatenate([random_weights(2 + options['network']['n_nt'],options['network']['Embedding']['n_dim'], 0.01), Vocab.i2e], axis = 0)
    elif options['network']['type'] == 'LSTM2_MeanDiff_deRNNG':
        emb_init = np.concatenate([random_weights(3, options['network']['Embedding']['n_dim'], 0.01), Vocab.i2e], axis = 0)
    else:
        emb_init = Vocab.i2e

    net = framework(options, LOG, emb_tok_init=torch.from_numpy(emb_init))
    net.load_state_dict(Best_Model['state_dict'])

    if torch.cuda.is_available():
        LOG.log('Using Device: %s' % torch.cuda.get_device_name(torch.cuda.current_device()))
        net = net.cuda()

    print(net)
    f_in = open(config.inputFile, 'r')
    f = open('summary.txt', 'w')
    fp = open('parse.txt', 'w')

    Annotator = PyCoreNLP()
    for idx, line in enumerate(f_in):
        source_ = line.strip()
        anno = Annotator.annotate(source_.encode('ascii', 'ignore'), eolonly=True)
        source_token = []
        for sent in anno['sentences']:
            for token in sent["tokens"]:
                source_token.append(token["originalText"].lower())
        source = ListOfWord2ListOfIndex(source_token, Vocab)
        [text, parse] = net.genSummary([source], Vocab, source_token)
        print(idx)
        print(text[0])
        print(parse[0])
        print(text[0], file=f)
        print(parse[0], file=fp)

def datasetBuilding(config):
    LOG.log('Building Dataset Setting')
    settingPath = "settings/dataset/newData.json"
    data = {
        "name":"newData",
        "method":"build",
        "srcFeatList": [],
        "refFeatList": ["action_map"],
        "Parts":
        {
            "train":
            {
                "name":"train",
                "path":config.train_prefix,
                "sorted": True,
                "shuffled": False,
            },
            "valid":
            {
                "name":"valid",
                "path":config.valid_prefix,
                "sorted": True,
                "shuffled": False
            }
        }
    }
    saveToJson(settingPath, data)
    return settingPath

def vocabularyBuilding(config):
    LOG.log('Building Vocabulary')
    setting = {
        "cased": False,
        "rmDigit" : True,
        "sortBy": "output",
        "minFreq": 5,
        "dim": 100,
        "initPath": "glove.6B.100d.txt",
        "inputCorpus":[config.train_prefix + ".Ndocument", config.valid_prefix + ".Ndocument"],
        "outputCorpus":[config.train_prefix + ".Nsummary", config.valid_prefix + ".Nsummary"]
    }
    Vocab = Vocabulary(setting)
    saveToPKL('vocab/newData.Vocab', Vocab)
    f = open('newData.i2w', 'w', encoding='utf-8')
    for item in Vocab.i2w:
        if (item == '<pad>' or item == '<unk>' or item == '<bos>' or item == '<eos>' or item == '<mask>'):
            print(item, 'NAN', file=f)
        else:
            print(item, Vocab.typeFreq[item], file=f)
    setting["full_size"] = Vocab.full_size
    setting["input_size"] = Vocab.n_in
    setting["output_size"] = Vocab.n_out
    setting["savePath"] = "settings/vocab/newData.json"
    saveToJson(setting["savePath"], setting)
    return setting

def networkBuilding(config):
    LOG.log('Building Network Setting')
    settingPath = "settings/network/" + config.model + "_newData.json"
    setting = loadFromJson("settings/network/" + config.model + "_gigaword.json")
    setting["Embedding"]["n_type"] = config.vocab_size + 3
    if setting["n_vocab"] > config.vocab_size:
        setting["n_vocab"] = config.vocab_size
        if "predict" in setting["Generator"]["Decoder"]:
            setting["Generator"]["Decoder"]["predict"]["n_out"] = config.vocab_size
        else:
            setting["Generator"]["Decoder"]["predict_tok"]["n_out"] = config.vocab_size
    saveToJson(settingPath, setting)
    return settingPath

def ParsingTreeBuilding(config):
    featList = ['action']

    def DeParse(file_prefix):
        Annotator = PyCoreNLP()
        f_in = open(file_prefix + '.Nsummary', 'r', encoding='utf-8')
        f_out = open(file_prefix + '.json', 'w', encoding='utf-8')
        Index = 0
        for line in f_in:
            Index += 1
            text = line.strip()
            anno = Annotator.annotate(text.encode('ascii', 'ignore'), eolonly=True)
            json.dump(anno, f_out)
            f_out.write('\n')
        return

    def extractFromDependency(data):
        number = len(data)
        incomeArc = [""] * number
        edge = [[]] * number
        token = [""] * number
        ROOT = -1

        for item in data:
            Id = item["dependent"] - 1
            father = item["governor"] - 1

            incomeArc[Id] = item["dep"]
            token[Id] = item["dependentGloss"]

            if (incomeArc[Id] != "ROOT"):
                if edge[father] != []:
                    edge[father].append(Id)
                else:
                    edge[father] = [Id]
            else:
                ROOT = Id

        Tree = {
            "number": number,
            "incomeArc": incomeArc,
            "edge": edge,
            "token": token,
            "ROOT": ROOT
        }

        return Tree

    def oracle(T):
        N = len(T['token'])
        right_most_child = [max([-1] + T['edge'][i]) for i in range(N)]
        head = [-1] * N
        for i in range(N):
            for j in T['edge'][i]:
                head[j] = i
        stack = []
        acts = []
        buffer = list(reversed(range(N)))

        while (len(buffer) > 0) or (len(stack) > 1):
            if (len(stack) < 2):
                stack.append(buffer.pop())
                acts.append(('GEN', T['token'][stack[-1]]))
            elif head[stack[-1]] == stack[-2] and \
                    (len(buffer) == 0 or right_most_child[stack[-1]] < buffer[-1] or right_most_child[stack[-2]] == -1):
                acts.append(('REDUCE_R', 0))
                stack.pop()
            elif head[stack[-2]] == stack[-1] and \
                    (len(buffer) == 0 or right_most_child[stack[-1]] < buffer[-1] or right_most_child[stack[-2]] == -1):
                acts.append(('REDUCE_L', 0))
                temp = stack.pop()
                stack.pop()
                stack.append(temp)
            elif len(buffer) > 0:
                stack.append(buffer.pop())
                acts.append(('GEN', T['token'][stack[-1]]))
            else:
                break
        acts.append(('REDUCE_R', 0))
        return acts

    def ParsingFeatures(file_prefix):
        f = open(file_prefix + '.json', 'r')
        Features = {}
        for feat in featList:
            Features[feat] = []
        Index = 0
        for l in f:
            data = json.loads(l)
            Tree_D = extractFromDependency(data["sentences"][0]["basicDependencies"])
            action = oracle(Tree_D)
            for feat in featList:
                Features[feat].append(eval(feat))

            Index += 1

        for feat in featList:
            f_out = open(file_prefix + '.' + feat, 'w')
            for item in Features[feat]:
                print(json.dumps(item), file=f_out)
            Features[feat] = []

        return

    def Compress_Action_De(file_prefix, Vocab):
        f_in = open(file_prefix + '.action', 'r')
        f_out = open(file_prefix + '.action_map', 'w')
        for l in f_in:
            data = eval(l.strip())
            newData = []
            for item in data:
                if (item[0] == 'GEN'):
                    tok = item[1].lower()
                    if tok not in Vocab.w2i:
                        tok = '<unk>'
                    newData.append((3, Vocab.w2i[tok]))
                elif item[0] == 'REDUCE_L':
                    newData.append((1, 0))
                elif item[0] == 'REDUCE_R':
                    newData.append((2, 0))
                print(newData, file=f_out)
        return

    def Merge(file_prefix, featList):
        ff = {}
        fl = {}
        data = {}
        f_out = open(file_prefix + '.sfeature', 'w')
        for feat in featList:
            ff[feat] = open(file_prefix + '.' + feat, 'r')
        Done = False
        while (True):
            for feat in featList:
                fl[feat] = ff[feat].readline()
                if not fl[feat]:
                    Done = True
                    break
                data[feat] = eval(fl[feat].strip())
            if (Done):
                break
            print(data, file=f_out)
        return

    def parsing(file_prefix):
        """
            STEP 1: Extract Features
            STEP 2: Parsing the Features
            STEP 3: Mapping to Vocab
            STEP 4: Merge Features
        """
        DeParse(file_prefix)
        ParsingFeatures(file_prefix)
        Vocab = loadFromPKL('newData.Vocab')
        Compress_Action_De(file_prefix, Vocab)
        Merge(file_prefix, ["action_map"])
        return

    LOG.log("Parsing Files")
    parsing(config.train_prefix)
    parsing(config.valid_prefix)


def argLoader():
    parser = argparse.ArgumentParser()

    # Options Setting
    # Actions
    parser.add_argument('--do_train', action='store_true', help="Whether to run training")
    parser.add_argument('--do_test', action='store_true', help="Whether to run test")

    # Dataset for training
    parser.add_argument('--data', type=str, default='gigaword')

    # Model setting
    parser.add_argument('--model', type=str, default='deRNNG')

    # Path for Input
    parser.add_argument('--inputFile', type=str, default='none')
    parser.add_argument('--train_prefix', type=str, default='train')
    parser.add_argument('--valid_prefix', type=str, default='valid')

    args = parser.parse_args()

    if args.do_test:
        if (args.inputFile == 'none'):
            print('No testing input file. Please use "--inputFile example.txt".')
            return args
        args.optionsFrame = {
            'dataset': 'settings/dataset/' + args.data + '.json',
            'vocabulary': 'settings/vocab/' + args.data + '.json',
            'network': 'settings/network/' + args.model + '_' + args.data + '.json',
            'test':'settings/test/test.json'
        }

    elif args.do_train:
        if (not path.exists(args.train_prefix+'.Ndocument')) or (not path.exists(args.train_prefix + '.Nsummary')):
            print('No training input file. Please use "--train_prefix train" to assign "train.Ndocument" and "train.Nsummary"')
            return args

        if (not path.exists(args.valid_prefix+'.Ndocument')) or (not path.exists(args.valid_prefix + '.Nsummary')):
            print('No validation input file. Please use "--valid_prefix valid" to assign "valid.Ndocument" and "valid.Nsummary"')
            return args


        args.optionsFrame = {}
        args.optionsFrame['dataset'] = datasetBuilding(args)
        vocab_setting = vocabularyBuilding(args)
        args.optionsFrame['vocabulary'] = vocab_setting['savePath']
        args.vocab_size = vocab_setting["full_size"]
        args.optionsFrame['network'] = networkBuilding(args)
        args.optionsFrame['training'] = "settings/training/gigaword.json"
        ParsingTreeBuilding(args)

    elif args.do_eval:
        args.optionFrames['evaluation']='settings/evaluation/evaluation.json'

    return args


def main():
    args = argLoader()
    print(args)

    print('CUDA', torch.cuda.current_device())

    if args.do_train:
        train(args)

    elif args.do_test:
        test(args)

if __name__ == '__main__':
    main()
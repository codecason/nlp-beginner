from numpy import *
import pickle, zipfile

vec_file = zipfile.ZipFile('../data/glove.840B.300d.zip', 'r').open('glove.840B.300d.txt', 'r')
all_vocab = {}
print('loading vocab...')
wmap = pickle.load(open('../data/sentiment/wordMapAll.bin', 'rb'))
revMap = {}
for word in wmap:
    revMap[wmap[word]] = word

for line in vec_file:
    split = [b.decode() for b in line.split()]
    try:
        x = wmap[split[0]]
        all_vocab[split[0]] = array(split[1:])
        all_vocab[split[0]] = all_vocab[split[0]].astype(float)
    except:
        pass

vec_file.close()

print(len(wmap), len(all_vocab))
d = len(all_vocab['the'])

We = empty((d, len(wmap)))

print('creating We for ', len(wmap), ' words')
unknown = []

for i in range(0, len(wmap)):
    word = revMap[i]
    try:
        We[:, i] = all_vocab[word]
    except KeyError:
        unknown.append(word)
        print('unknown: ', word)
        We[:, i] = all_vocab['unknown']

print('num unknowns: ', len(unknown))
print(We.shape)

print('dumping...')
pickle.dump( We, open('../data/sentiment_We', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

import cPickle

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = cPickle.load(fo)
  return dict

if __name__ == '__main__':
  print unpickle('bin/data_batch_0')

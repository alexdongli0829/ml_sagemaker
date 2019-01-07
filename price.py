#!/usr/bin/env python
# coding: utf-8

# In[22]:


import argparse
import os
import json


# In[15]:


def parse_args():
    parser = argparse.ArgumentParser()

    # retrieve the hyperparameters we set in notebook (with some defaults)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    #parser.add_argument('--log-interval', type=int, default=1000)
    #parser.add_argument('--embedding-size', type=int, default=50)

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training_channel', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))

    return parser.parse_args()


# In[26]:


from mxnet.gluon import loss as gloss, nn, data as gdata
from mxnet import autograd, gluon, init,nd
import mxnet as mx

loss = gloss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    return net


# In[17]:


def train(num_gpus, train_dir, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_dir = train_dir
    data_file = train_dir+'/sorted_features.csv'
    label_file = train_dir+'/sorted_labels.csv'
    
    CSVIter = mx.io.CSVIter(data_csv=data_file,data_shape=(331,),label_csv=label_file,label_shape=(1,),batch_size=batch_size)
    
    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()

    net = get_net()
    net.initialize(ctx=ctx)
    
    # 这里使用了 Adam 优化算法。
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        #for X, y in train_iter:
        for i, batch in enumerate(CSVIter):
            X = batch.data[0].as_in_context(ctx)
            y = batch.label[0].as_in_context(ctx).reshape(-1,1)
            with autograd.record():
                l = loss(net(X), y)
                l.backward()
            trainer.step(batch_size)
        CSVIter.reset()
       # train_ls.append(log_rmse(net, train_features, train_labels))
    return net


# In[18]:

'''
def log_rmse(net, train_features, train_labels):
    # 将小于 1 的值设成 1，使得取对数时数值更稳定。
    clipped_preds = nd.clip(net(train_features), 1, float('inf'))
    rmse = nd.sqrt(2 * loss(clipped_preds.log(), train_labels.log()).mean())
    return rmse.asscalar()
'''

# In[23]:


      
def save(net, model_dir):
    # save the model
    net.save_parameters('%s/model.params' % model_dir)

if __name__ =='__main__':

    args = parse_args()
    num_cpus = int(os.environ['SM_NUM_CPUS'])
    num_gpus = int(os.environ['SM_NUM_GPUS'])
    model = train(num_gpus, args.training_channel, args.epochs, args.learning_rate, args.weight_decay, args.batch_size)
    save(model, args.model_dir)


# In[ ]:

# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    net = get_net()
    net.load_parameters('%s/model.params' % model_dir, ctx=mx.cpu())
    return net


def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.
    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # we can use content types to vary input/output handling, but
    # here we just assume json for both
    net = net
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
    parsed = nd.array(json.loads(data))
    print(parsed.shape)
    output = net(parsed)
    print(output)
    response_body = json.dumps(output.asnumpy().tolist())
    return response_body, output_content_type


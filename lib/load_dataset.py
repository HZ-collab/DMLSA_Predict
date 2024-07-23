import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('../data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PeMSD8/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD3':
        data_path = os.path.join('../data/PeMSD3/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7':
        data_path = os.path.join('../data/PEMSD7/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset =='MetrLA':
        data_path = os.path.join('../data/MetrLA/metr-la.npz')
        data = np.load(data_path)['data'][:, :]  #onley the first dimension, traffic flow data
    elif dataset =='Electricity':
        data_path = os.path.join('../data/Electricity/Electricity.npz')
        data = np.load(data_path)['data'][:, :]  #onley the first dimension, traffic flow data
    elif dataset =='PEMSBAY':
        data_path = os.path.join('../data/PEMSBAY/pems-bay.npz')
        data = np.load(data_path)['data'][:, :]  #onley the first dimension, traffic flow data
    elif dataset =='TX':
        data_path = os.path.join('../data/covid_data/TX_COVID.npz')
        data = np.load(data_path)['arr_0'][:, :]  
    elif dataset =='milan':
        data_path = os.path.join('../data/milan/milan_data_part_nonorm2.npz')
        data = np.load(data_path)['arr_0'][:, :]  #onley the first dimension, traffic flow data
    elif dataset =='ExchangeRate':
        data_path = os.path.join('../data/ExchangeRate/ExchangeRate.npz')
        data = np.load(data_path)['data'][:, :]  #onley the first dimension, traffic flow data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

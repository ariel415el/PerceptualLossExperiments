import pickle
import numpy as np
from os.path import join
from os import listdir
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import torch
from time import time
from tqdm import tqdm
import cv2

def read_cifar_10_data(root_dir, restrict_labels=None, limit_samples=None):
    img_size = 32
    dataset_name = f"Cifar10-{restrict_labels}"
    start = time()
    print("Reading cifar10.. ", end='')
    data = []
    labels = []
    for file_ in listdir(root_dir):
        if file_.split('_')[0] == 'data':
            dict = unpickle(join(root_dir, file_))
            data.extend(dict[b'data'])
            labels.extend(dict[b'labels'])
        if file_.split('_')[0] == 'test':
            dict = unpickle(join(root_dir, file_))
            data.extend(dict[b'data'])
            labels.extend(dict[b'labels'])
    data, labels = np.array(data), np.array(labels).reshape(-1)
    if restrict_labels:
        indices = np.isin(labels, restrict_labels)
        data = data[indices]
        labels = labels[indices]
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]
    if limit_samples:
        limit = min(len(data), limit_samples)
        data = data[:limit]
        labels = labels[:limit]

    reshaped_data = data.reshape(-1, 3, img_size, img_size).astype(np.float32) / 255
    print(f"done in {time() - start:.2f} s, data shape = {reshaped_data.shape}")
    return reshaped_data, labels, img_size, dataset_name

def read_lfw_data(root_dir, num_celebs):
    img_size = 64
    dataset_name = f"LFW-top{num_celebs}"
    start = time()
    print(f"Reading LFW ({num_celebs} largest celebs).. ", end='')
    res = {}
    celeb_names = listdir(join(root_dir, 'lfw-deepfunneled'))
    celeb_names = ['Ariel_Sharon', 'Serena_Williams', 'Andre_Agassi', 'Luiz_Inacio_Lula_da_Silva', 'Jennifer_Lopez',
                   'Nicole_Kidman', 'Halle_Berry', 'Laura_Bush', 'Roh_Moo-hyun', 'Kofi_Annan']
    for celeb_name in celeb_names:
        paths = []
        for fn in listdir(join(root_dir, 'lfw-deepfunneled', celeb_name)):
            paths.append(join(root_dir, 'lfw-deepfunneled', celeb_name, fn))
        res[celeb_name] = paths

    # restricted_keys = sorted(res.keys(), key=lambda x: len(res[x]))[::-1]
    # largest_celebs = largest_celebs[:num_celebs]

    data, labels = [], []
    for i, celeb in enumerate(res):
        for im_path in res[celeb]:
            img = cv2.imread(im_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = img[50:200, 50:200]
            img = img[70:180, 70:180]
            img = cv2.resize(img, (img_size, img_size))
            img = img.transpose(2,0,1).astype(np.float32) / 255
            data.append(img)
            labels.append(i)

    data, labels = np.array(data), np.array(labels).reshape(-1)
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]

    print(f"done in {time() - start:.2f} s, data shape = {data.shape}")

    return data, labels, img_size, dataset_name


def reshape_to_plot(data, img_dim):
    return data.reshape(data.shape[0], 3, img_dim,  img_dim).transpose(0, 2, 3, 1).astype("uint8")


def plot_imgs(in_data, img_dim, path):
    data = np.array([d for d in in_data])
    data = reshape_to_plot(data, img_dim)
    if len(data) > 1:
        r = int(np.ceil(np.sqrt(len(data))))
        fig, ax = plt.subplots(r, r, figsize=(5, 5))
        i = 0
        for j in range(r):
            for k in range(r):
                if i < len(data):
                    ax[j, k].imshow(data[i:i + 1][0])
                ax[j, k].set_axis_off()
                i += 1

    else:
        plt.imshow(data[0])

    plt.savefig(path)
    plt.clf()
    plt.close('all')


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def extrac_features(model, data, batch_size, device=torch.device("cpu")):
    features = []
    pbar = tqdm(total=data.shape[0])
    pbar.set_description_str("Extracting features from data")
    model = model.to(device)
    for b in range(data.shape[0] // batch_size):
        batch_images = data[b * batch_size: (b + 1) * batch_size]
        tensor = torch.from_numpy(batch_images).to(device)
        output = model.get_fv(tensor)
        output = output.detach().cpu().numpy()
        features.append(output)
        pbar.update(batch_size)
    if (b + 1) * batch_size < len(data):
        batch_images = data[(b+1) * batch_size: ]
        tensor = torch.from_numpy(batch_images).to(device)
        output = model.get_fv(tensor)
        output = output.detach().cpu().numpy()
        features.append(output)

    features = np.concatenate(features, axis=0)

    pbar.close()
    return features


def compute_metrics(model, gt_labels, report_path):
    f = open(report_path, 'w')
    f.write("Cluster-idx, recall, precision, distribution\n")
    precisions = []
    recals = []
    pred_labels = gt_labels.copy()
    N = len(gt_labels)
    num_labels = len(np.unique(gt_labels))
    for i in range(model.n_clusters):
        cluster_indices = model.cluster_assignments == i
        counts = np.bincount(gt_labels[cluster_indices], minlength=num_labels)
        predicted_class = np.argmax(counts)
        pred_labels[cluster_indices] = predicted_class
        n_tp = counts[predicted_class]
        recals.append(n_tp / np.sum(gt_labels == predicted_class))
        precisions.append(n_tp / np.sum(counts))
        f.write(f"{i}          , {recals[-1]:.1f}   , {precisions[-1]:.1f}      , [{','.join([str(x).rjust(3, ' ') for x in counts])}]\n")
    f.close()

    f1 = f1_score(gt_labels, pred_labels, average='weighted')
    recal = np.mean(recals)
    precision = np.mean(precisions)

    return f1, recal, precision


class TorchMetricWrapper:
    def __init__(self, metric, device):
        self.metric = metric.to(device)
        self.device = device

    def __call__(self, x, y):
        x_t = torch.from_numpy(x).to(self.device).unsqueeze(0)
        y_t = torch.from_numpy(y).to(self.device).unsqueeze(0)
        return self.metric(x_t, y_t)


def np_pairwise_distances(X, Y, metric):
    pbar = tqdm(total=len(X)*len(Y))
    pbar.set_description_str(f"Computing pairwize similarities between {len(X)} and {len(Y)} samples.. ")
    result = np.zeros((len(X), len(Y)))
    for j in range(len(Y)):
        result[:, j] = metric(X, Y[j])
        pbar.update(len(X))

    return result

def slow_pairwise_distances(X, Y, metric):
    # pbar = tqdm(total=len(X)*len(Y))
    # pbar.set_description_str(f"Computing pairwize similarities between {len(X)} and {len(Y)} samples.. ")
    result = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            result[i, j] = metric(X[i], Y[j])
            # pbar.update(1)

    return result
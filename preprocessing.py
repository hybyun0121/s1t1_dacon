import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data

import kora.install.rdkit
import rdkit

from rdkit import Chem
from rdkit import RDLogger

from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': list(range(0,2)),
    'is_in_ring': list(range(0,2)),
    'num_OuterElecs' : list(range(0,8)),
    'PrQuNum' : list(range(0,6))
}

x_onehot = {
    'chirality' : list(range(0,4)),
    'hybridization' : list(range(0,8)),
    'is_aromatic' : list(range(0,2)),
    'is_in_ring' : list(range(0,2))
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}

e_onehot = {
    'bond_type' : list(range(0,5)),
    'stereo' : list(range(0,6)),
    'is_conjugated' : list(range(0,2))}

class smile2graph:
    def __init__(self, file_path):
        self.file_path = file_path

    def add_mol_info(self):
        train = pd.read_csv(self.file_path + '/train.csv')
        dev = pd.read_csv(self.file_path + '/dev.csv')
        test = pd.read_csv(self.file_path + '/test.csv')

        num_train = len(train)
        num_dev = len(dev)
        num_test = len(test)

        data = pd.concat([train, dev, test])

        data['mol'] = data['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))

        trainset = data.iloc[:int(num_train+num_dev)]
        testset = data.iloc[int(num_train+num_dev):]

        print(f'Size of train data : {len(trainset)}, Size of test data : {len(testset)}')

        return trainset, testset

    def get_mol_graph(self, dataset, is_train=False):

        pt = rdkit.Chem.GetPeriodicTable()

        graph_list = []
        for row in tqdm(dataset.itertuples()):
            if is_train:
                mol = row[4]
                y = np.array(row[-2:])
            else:
                mol = row[4]

            xs = []
            for atom in mol.GetAtoms():
                num_atom = atom.GetAtomicNum()
                x = []
                x.extend(np.eye(len(x_map['atomic_num']))[atom.GetAtomicNum()])
                idx = x_map['chirality'].index(str(atom.GetChiralTag()))
                x.extend(np.eye(len(x_onehot['chirality']))[idx])
                x.extend(np.eye(len(x_map['degree']))[atom.GetTotalDegree()])
                x.extend(np.eye(len(x_map['formal_charge']))[atom.GetFormalCharge()])
                x.extend(np.eye(len(x_map['num_hs']))[atom.GetTotalNumHs()])
                x.extend(np.eye(len(x_map['num_radical_electrons']))[atom.GetNumRadicalElectrons()])
                idx = x_map['hybridization'].index(str(atom.GetHybridization()))
                x.extend(np.eye(len(x_onehot['hybridization']))[idx])
                idx = x_map['is_aromatic'].index(atom.GetIsAromatic())
                x.extend(np.eye(len(x_onehot['is_aromatic']))[idx])
                idx = x_map['is_in_ring'].index(atom.IsInRing())
                x.extend(np.eye(len(x_onehot['is_in_ring']))[idx])
                x.extend(np.eye(len(x_map['num_OuterElecs']))[pt.GetNOuterElecs(num_atom)])
                x.extend(np.eye(len(x_map['PrQuNum']))[GetPrincipleQuantumNumber(num_atom)])

                xs.append(x)

            x = torch.tensor(xs, dtype=torch.long).view(-1, len(xs[0]))

            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                e = []
                idx = e_map['bond_type'].index(str(bond.GetBondType()))
                e.extend(np.eye(len(e_onehot['bond_type']))[idx])
                idx = e_map['stereo'].index(str(bond.GetStereo()))
                e.extend(np.eye(len(e_onehot['stereo']))[idx])
                idx = e_map['is_conjugated'].index(bond.GetIsConjugated())
                e.extend(np.eye(len(e_onehot['is_conjugated']))[idx])

                edge_indices += [[i, j], [j, i]]
                edge_attrs += [e, e]

            edge_index = torch.tensor(edge_indices)
            edge_index = edge_index.t().to(torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(len(edge_indices), -1)

            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

            if is_train:
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            else:
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

            graph_list.append(data)

        return graph_list
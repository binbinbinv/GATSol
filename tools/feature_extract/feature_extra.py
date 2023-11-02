import os
import pandas as pd
import iFeatureOmegaCLI
import torch
import numpy as np
from torch_geometric.data import Data
import warnings
warnings.filterwarnings("ignore")
import pickle
from tqdm import tqdm
import multiprocessing
import logging
from torch_geometric.utils import add_self_loops
import contextlib
import io
import esm

# Load ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

def name_label_dict(path):
    pdb_chain_list = pd.read_csv(path, header=0)
    dict_pdb_chain = pdb_chain_list.set_index('afname')['label'].to_dict()
    return dict_pdb_chain
  
def name_seq_dict(path):
    pdb_chain_list = pd.read_csv(path, header=0)
    dict_pdb_chain = pdb_chain_list.set_index('afname')['sequence'].to_dict()
    return dict_pdb_chain

def column_normalize(normalized_tensor):
  # 对每列进行归一化
  column_max, _ = torch.max(normalized_tensor, dim=0)
  column_min, _ = torch.min(normalized_tensor, dim=0)
  normalized_tensor = (normalized_tensor - column_min) / (column_max - column_min)
  # 返回归一化后的结果
  return normalized_tensor

def process_file(file, batch_converter = batch_converter, model = model):
  dict_path = '/home/bli/homology/feature_extract/list371.csv'
  name_dict = name_label_dict(dict_path)
  seq_dict = name_seq_dict(dict_path)
  # 执行指令的代码
  cm_directory = "/home/bli/homology/colabfold_cm_371"
  fasta_directory = "/home/bli/homology/colabfold_fasta_371"

  try:
    cm_path = os.path.join(cm_directory,file+"-model_v4.cm")
    fasta_path = os.path.join(fasta_directory,file+"-model_v4.fasta")
    pkl_path = os.path.join("/home/bli/homology/colabfold_pkl_371",file+"-model_v4.pkl")
    
    #esm feature
    batch_labels, batch_strs, batch_tokens = batch_converter([(file, seq_dict[file])])
    with torch.no_grad():
      results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    
    protein = iFeatureOmegaCLI.iProtein(fasta_path)
    
    # 创建一个空的文件对象
    null_file = io.StringIO()

    # 使用 redirect_stdout 上下文管理器将标准输出重定向到空文件
    with contextlib.redirect_stdout(null_file):
      protein.import_parameters('/home/bli/homology/feature_extract/Protein_parameters_setting.json')

    protein.get_descriptor("BLOSUM62")

    node_feature = torch.from_numpy((protein.encodings.values.reshape(-1,20))).float()
    node_feature1 = results['representations'][33][0, 1:-1].reshape(-1,1280)
    
    node_features = torch.cat((node_feature,node_feature1),1)
    
    with open(cm_path, 'r') as f:
      content = f.read()
      # 将内容解析为二维数组
      data = np.array([list(map(float, line.split(','))) for line in content.split('\n') if line])
      # 将数组转换为 PyTorch Tensor
      edges = torch.from_numpy(data).type(torch.LongTensor)
      
    label = torch.tensor(name_dict[file]).reshape(1,)
    data = Data(x=node_features, edge_index=edges.t().contiguous() - 1, y=label)
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=node_features.shape[0])
    with open(pkl_path, 'wb') as fpkl:
      pickle.dump(data, fpkl)
  except Exception as e:
    logging.error(f"Error processing {file}: {str(e)}")
    
def main():
  # 配置logging
  logging.basicConfig(filename='/home/bli/homology/feature_extract/log.log', level=logging.ERROR, format='%(asctime)s %(levelname)s: %(message)s')
  
  dict_path = '/home/bli/homology/feature_extract/list371.csv'
  name_dict = name_label_dict(dict_path)
  file_names = list(name_dict.keys())
  with multiprocessing.Pool(processes=5) as pool: # 在这里指定进程数为24
      with tqdm(total=len(file_names)) as pbar: # 创建进度条，总长度为len(file_names)
          for _ in pool.imap_unordered(process_file, file_names):
            pbar.update() # 更新进度条
      pool.close()
      pool.join()
  
if __name__ == '__main__':
  main()

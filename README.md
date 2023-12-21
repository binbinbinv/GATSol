# GATSol

GATSol, an enhanced predictor of protein solubility through the synergy of 3D structure graph and large language modeling

## 1.Re-train the model

1. 克隆项目到本地

   ```shell
   git clone git@github.com:binbinbinv/GATSol.git
   cd GATSol/
   ```

2. Download the datasets follow the description in ./dataset/readme.md

3. Extract the dataset by the command:

   ```shell
   tar -zxvf  ~/GATSol_dataset.tar.gz -C ~/GATSol/dataset/
   ```

4. Retrain the model

   ```shell
   python re-train.py
   ```

   
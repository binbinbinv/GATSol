# GATSol

GATSol, an enhanced predictor of protein solubility through the synergy of 3D structure graph and large language modeling

## 0.Conda environmennt create

If you want to run this model in a standalone mode on your own linux system, git-clone ProtSol with the following code:

```shell
git clone https://github.com/binbinbinv/GATSol.git
```

Then install the GATSol environment with the following commands, but first make sure you have conda or miniconda installed on your server.

Then you can install GATSol environment manually by following the instructions:

```shell
conda create -n GATSol python=3.9
conda activate GATSol
pip install torch==2.2.2 torchvision torchaudio
pip install pandas bio seaborn matplotlib_inline
pip install scikit-learn transformers Ipython
pip install iFeatureOmegaCLI rdkit
pip install torch_geometric==2.3.0 fair-esm
```

## 1.Predict your own protein

1. Download the best model after training by following the **readme.md** file in GATSol/check_point/best_model/readme.md, and put it into the best_model folder.

2. You must prepare your protein as the format **fasta** and **pdb**, and then put them in the folder below:

   ①GATSol/Predict/NEED_to_PREPARE/fasta

   ②GATSol/Predict/NEED_to_PREPARE/pdb

   And you need to prepare a **list.csv** as the example in the /home/bli/GATSol/Predict/NEED_to_PREPARE.

3. After preparing all the files, cd to the prediction work folder and execute the following command, you will get the **Output.csv** file, which contains the prediction results you need.

   ```shell
   cd GATSol/Predict
   bash ./tools/Predict.sh
   ```
## 2.Re-train the model

1. cd to the GAT project directory

   ```shell
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

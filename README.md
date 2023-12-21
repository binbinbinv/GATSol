# GATSol

GATSol, an enhanced predictor of protein solubility through the synergy of 3D structure graph and large language modeling

## 0.Conda environmennt create

```shell
git clone git@github.com:binbinbinv/GATSol.git
cd GATSol/
conda activate base
conda env create -f environment.yaml
conda activate pyg
```

## 1.Re-train the model

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

## 2.Predict your own protein

1. Download the best model after training by following the readme.md file in GATSol/check_point/best_model/readme.md, and put it into the best_model folder.

2. You must prepare your protein as the format fasta and pdb, and then put them in the folder below:

   ①GATSol/Predict/NEED_to_PREPARE/fasta

   ②GATSol/Predict/NEED_to_PREPARE/pdb

   And you need to prepare a list.csv as the example in the /home/bli/GATSol/Predict/NEED_to_PREPARE.

2. After preparing all the files from step ①, go to your working folder and execute the following command, you will get the Output.csv file, which contains the prediction results you need。

   ```shell
   cd GATSol/Predict
   bash ./tools/Predict.py
   ```


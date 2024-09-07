#!/bin/sh

#################### activate conda env #######################

eval "$(conda shell.bash hook)"
# conda activate pyg

#################### prcessing all fna files #######################
counter=0
sourceDir=`ls /home/bli/GNN/Graph_bin/data/homology/alphafold_test/fold_completed_pdb_2697`
num=`ls /home/bli/GNN/Graph_bin/data/homology/alphafold_test/fold_completed_pdb_2697 | wc -l`
for name in $sourceDir
do
	((counter=counter+1))
	name="${name/.pdb/}"
	python pdb_to_cm.py "/home/bli/GNN/Graph_bin/data/homology/alphafold_test/fold_completed_pdb_2697/"$name".pdb" "/home/bli/GNN/Graph_bin/data/homology/alphafold_test/fold_completed_cm_2697/$name.cm" -t 10.0
	echo "$name $counter/$num --.pdb files completed"
done

#################finished#####################

echo "All .pdb files has been convertcd to protein contact map" 

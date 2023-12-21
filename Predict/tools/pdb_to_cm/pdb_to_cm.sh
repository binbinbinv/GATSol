#!/bin/sh

#################### activate conda env #######################

eval "$(conda shell.bash hook)"
conda activate pyg

#################### prcessing all fna files #######################
counter=0
sourceDir=`ls ./NEED_to_PREPARE/pdb/`
num=`ls ./NEED_to_PREPARE/pdb/ | wc -l`
for name in $sourceDir
do
	((counter=counter+1))
	name="${name/.pdb/}"
	python ./tools/pdb_to_cm/pdb_to_cm.py "./NEED_to_PREPARE/pdb/"$name".pdb" "./NEED_to_PREPARE/cm/$name.cm" -t 10.0
	echo "$name $counter/$num --.pdb files completed"
done

#################finished#####################

echo "All .pdb files has been convertcd to protein distance map" 
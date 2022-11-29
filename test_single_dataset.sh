############################################################
# ViT with normal case
############################################################
export code="./scripts/base_test.sh"
#dataset=$1
#num_label=$2
#batch=$3
sh ${code} "CottonWeedID15" 15 16


############################################################
# ViT with few-shot case
############################################################
#export code="./scripts/base_test_few_shot.sh"
##dataset=$1
##num_label=$2
##batch=$3
##gpu=$4
#sh ${code} "CottonWeedID15" 15 8 0


############################################################
# CNN with normal case
############################################################
#export code="./cnn_scripts/base_test.sh"
##dataset=$1
##num_label=$2
##batch=$3
#sh ${code} "CottonWeedID15" 15 16


############################################################
# CNN with few-shot case
############################################################
#export code="./cnn_scripts/base_test_few_shot.sh"
##dataset=$1
##num_label=$2
##batch=$3
##gpu=$4
#sh ${code} "CottonWeedID15" 15 8 0
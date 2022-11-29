# Apple2020
# Apple2021
# cassava
# CGIAR_wheat
# ChineseStrawberry
# CitrusLeaf
# IVADL_rose
# IVADL_tomato
# PlantDoc_cls
# PlantVillage
# Rice1462
# Rice2020
# TaiwanTomato
# CottonWeedID15

export code="./analysis_results/compare_val.py"

#############################################
# validation acc
#############################################
python ${code} --dset "Apple2020"
python ${code} --dset "Apple2021"
python ${code} --dset "cassava"
python ${code} --dset "CGIAR_wheat"
python ${code} --dset "ChineseStrawberry"
python ${code} --dset "CitrusLeaf"
python ${code} --dset "IVADL_rose"
python ${code} --dset "IVADL_tomato"
python ${code} --dset "PlantDoc_cls"
python ${code} --dset "PlantVillage"
python ${code} --dset "Rice1462"
python ${code} --dset "Rice2020"
python ${code} --dset "TaiwanTomato"
python ${code} --dset "CottonWeedID15"

#############################################
# validation loss
#############################################
python ${code} --dset "Apple2020" --mode "loss"
python ${code} --dset "Apple2021" --mode "loss"
python ${code} --dset "cassava" --mode "loss"
python ${code} --dset "CGIAR_wheat" --mode "loss"
python ${code} --dset "ChineseStrawberry" --mode "loss"
python ${code} --dset "CitrusLeaf" --mode "loss"
python ${code} --dset "IVADL_rose" --mode "loss"
python ${code} --dset "IVADL_tomato" --mode "loss"
python ${code} --dset "PlantDoc_cls" --mode "loss"
python ${code} --dset "PlantVillage" --mode "loss"
python ${code} --dset "Rice1462" --mode "loss"
python ${code} --dset "Rice2020" --mode "loss"
python ${code} --dset "TaiwanTomato" --mode "loss"
python ${code} --dset "CottonWeedID15" --mode "loss"
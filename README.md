conda env create -f environment.yml
conda activate hbn-ml

export HBN_PUBLIC_CSV_URL="http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/File/_pheno/HBN_R11_Pheno.csv"

#### Run API test + pipeline validation
python tests.py

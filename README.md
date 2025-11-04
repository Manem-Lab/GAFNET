<!-- #region -->
# Interpretable identification of multi-omics biomarkers across biological networks via attention-fused GraphSAGE representation learning


<img src="figure/final_multiomics_im.jpg" alt="drawing" width="75%"/>


## Abstract
Multi-omics data provide comprehensive and complementary signals that are highly valuable for understanding disease mechanisms and predicting patient outcomes, particularly in cancer, which remains a leading cause of mortality worldwide. Traditional graph-based models rely on static neighborhood aggregation or layer-specific transformations, often limiting their ability to generalize across heterogeneous biomedical data. In contrast, Graph Sample and Aggregate (GraphSAGE) introduces an inductive learning framework that samples and aggregates information from neighboring nodes, enabling efficient learning on large or unseen graphs. To harness these rich yet heterogeneous data sources, we developed a GraphSAGE-based deep learning framework with an attention mechanism to classify cancer patients as short-term or long-term survivors, incorporating mRNA, miRNA, and DNA methylation profiles. The proposed model learns modality-specific embeddings from each omics layer using, effectively capturing local patient similarity patterns within each molecular domain, which are subsequently integrated through an attention mechanism to derive a unified multimodal representation. 
Across 18 TCGA cancer cohorts, the proposed model consistently achieved higher AUC values than existing graph-based methods (average AUC = 0.88, an improvement of 5–10% over baseline methods). Model performance was further enhanced by comparing multiple edge-creation strategies, with correlation-based edges yielding the most robust results. The framework was validated using stratified 5-fold cross-validation to ensure reliability and generalizability across datasets.
Finally, we applied the SHAP (SHapley Additive exPlanations) GradientExplainer to quantify the contribution of each molecular feature within mRNA, miRNA, and DNA methylation data, enabling the identification of biologically meaningful biomarkers associated with patient survival. Collectively, these results demonstrate that our GraphSAGE-Attention framework effectively integrates heterogeneous omics data, enhances graph connectivity, and improves predictive accuracy across diverse cancer types.


## Installation
```
conda env create -n lung-ddpm python=3.9
conda activate lung-ddpm
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Data Preparation

## Training 
```
python 


## Citation


## Acknowledgements




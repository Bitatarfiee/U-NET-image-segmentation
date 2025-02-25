# Automated Brain Image Segmentation for Radiotherapy Planning 

## Description:

Despite advancements in cancer treatment, brain tumors remain challenging to detect and treat due to their complexity. Radiotherapy, a common treatment for cancer, requires precise segmentation of tumor regions and organs at risk (OARs) for optimal outcomes. This thesis explores automated brain image segmentation using U-Net and Vision Transformer (ViT) deep learning techniques to enhance radiotherapy planning. We used the Wilcoxon signed-rank test to determine performance differences. We examined U-Net model uncertainty with Monte Carlo (MC) dropout, optimal input configurations, pretraining effects, and the impact of separate versus unified models for OARs and tumors. Despite its superiority over ViT in brain image segmentation, our research showed that U-Net encounters challenges in clinical target volume (CTV) segmentation and edge detection. In our dataset, T1w-CE images and combined CT/MRI sequences emerged as the most informative for segmentation. Pre-training with the BraTS dataset and using separate models for tumors and OARs did not significantly improve the models' performance.

https://urn.kb.se/resolve?urn=urn:nbn:se:liu:diva-208518

## files: 
Util_Creat_Mask.py : Creat target files containing:

- 8 targets: 'OpticNerve_R', 'OpticNerve_L', 'Eye_R', 'Eye_L', 'BrainStem', 'Chiasm', 'CTV', 'GTV' 
- 5 targets: 'OpticNerve_R', 'OpticNerve_L', 'Eye_R', 'Eye_L', 'BrainStem','Chiasm', 
- 3 targets: 'CTV', 'GTV', PTV




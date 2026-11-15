"""Generation 0: Baseline MRG transferred to hyperspectral data (Section 8.3).

Regions: MRG on the V channel of PC1-3, seeds in histogram-frequency order,
intensity + orientation constraint to the seed, growth confined to a 3x3 grid.
Region description: 88 GLCM features + mean spectrum. Classifier: linear SVM.

PARAMS values are placeholders. Replace them with the values from Table 6.2 /
Section 6.3.3 so the harness reproduces 89.82% test OA on Indian Pines.
"""
import numpy as np
from llm_mrg import blocks as B
DESCRIPTION = 'Modified region growing on the PCA V channel with histogram-frequency seeds, intensity and orientation constraints to the seed inside a 3x3 grid; regions described by 88 GLCM features and their mean spectrum.'
PARAMS = {'t_intensity': 0.09177596629691573, 't_orientation': 0.5640330370037304, 'n_bins': 65, 'grid': 3, 'connectivity': 8, 'glcm_levels': 15}
CLASSIFIER = {'kernel': 'linear', 'C': 0.2001088271097828, 'gamma': 'scale'}
LABELING = 'region'

def segment(ctx, params):
    return B.mrg(ctx, t_intensity=params['t_intensity'], t_orientation=params['t_orientation'], n_bins=params['n_bins'], grid=params['grid'], connectivity=params['connectivity'])

def region_features(ctx, regions, rows, params):
    g = B.glcm88(ctx, regions, rows, levels=params['glcm_levels'])
    spec = B.region_means(ctx.bands, regions, rows)
    return np.hstack([g, spec])

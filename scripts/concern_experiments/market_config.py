"""
Canonical market configuration for all ICAIF paper experiments.

Single source of truth for universe sizes, cost multipliers, and shared constants.
All experiment scripts should import from here instead of defining their own MARKETS dict.

Canonical config matches expAR (EMA robustness) which produces the paper's headline results.
"""

MARKETS = {
    'THA': {'file': 'public/data/ThaiEquitiesData.csv', 'type': 'thai_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
    'KOR': {'file': 'data/KOREquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 200, 'nq': 5},
    'TWN': {'file': 'data/TWNEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 200, 'nq': 5},
    'IND': {'file': 'data/INDEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 200, 'nq': 5},
    'CHN': {'file': 'data/CHNEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 300, 'nq': 5},
    'MYS': {'file': 'data/MYSEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
    'POL': {'file': 'data/POLEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
    'IDN': {'file': 'data/IDNEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
    'ZAF': {'file': 'data/ZAFEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
    'TUR': {'file': 'data/TUREquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
    'BRA': {'file': 'data/BRAEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
    'PHL': {'file': 'data/PHLEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
    'GRC': {'file': 'data/GRCEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 50, 'nq': 5},
    'SAU': {'file': 'data/SAUEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
    'MEX': {'file': 'data/MEXEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
}

COST_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0]
BUFFER_PCT = 0.20
MIN_TRAIN_MONTHS = 60

# Fine EMA alpha grid (used by expAR, expBA)
EMA_ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
              0.60, 0.70, 0.80, 0.90, 1.00]

# Persistence filter constants (used by expAN)
PERSISTENCE_THRESHOLD = 0.75
PERSISTENCE_FALLBACK_THRESHOLD = 0.60
PERSISTENCE_MIN_FEATURES = 10
PERSISTENCE_RECOMPUTE_INTERVAL = 12
PERSISTENCE_LOOKBACK = 60
PERSISTENCE_MIN_MONTHS_AC = 24

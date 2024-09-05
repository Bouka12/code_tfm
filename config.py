# config.py

experiments = {
"IS0_TSR1_IF5" : {'IS':0,'TSR':1,   'IF':5},
"IS0_TSR1_IF4" : {'IS':0,'TSR':1,   'IF':4}, 
"IS0_TSR1_IF3" : {'IS':0,'TSR':1,   'IF':3},
"IS0_TSR1_IF2" : {'IS':0,'TSR':1,   'IF':2},
"IS0_TSR1_IF1" : {'IS':0,'TSR':1,   'IF':1},

"IS2_TSR1_IF5" : {'IS':2,'TSR':1,   'IF':5} ,
"IS2_TSR1_IF4" : {'IS':2,'TSR':1,   'IF':4},
"IS2_TSR1_IF3" : {'IS':2,'TSR':1,   'IF':3},
"IS2_TSR1_IF2" : {'IS':2,'TSR':1,   'IF':2},
"IS2_TSR1_IF1" : {'IS':2,'TSR':1,   'IF':1},

"IS0_TSR2_IF5" : {'IS':0,'TSR':2,   'IF':5},
"IS0_TSR2_IF4" : {'IS':0,'TSR':2,   'IF':4},
"IS0_TSR2_IF3" : {'IS':0,'TSR':2,   'IF':3},
"IS0_TSR2_IF2" : {'IS':0,'TSR':2,   'IF':2},
"IS0_TSR2_IF1" : {'IS':0,'TSR':2,   'IF':1},

"IS2_TSR2_IF5" : {'IS':2,'TSR':2,   'IF':5},
"IS2_TSR2_IF4" : {'IS':2,'TSR':2,   'IF':4},
"IS2_TSR2_IF3" : {'IS':2,'TSR':2,   'IF':3},
"IS2_TSR2_IF2" : {'IS':2,'TSR':2,   'IF':2},
"IS2_TSR2_IF1" : {'IS':2,'TSR':2,   'IF':1},

"IS0_TSR3_IF5" : {'IS':0,'TSR':3,   'IF':5},
"IS0_TSR3_IF4" : {'IS':0,'TSR':3,   'IF':4},
"IS0_TSR3_IF3" : {'IS':0,'TSR':3,   'IF':3},
"IS0_TSR3_IF2" : {'IS':0,'TSR':3,   'IF':2},
"IS0_TSR3_IF1" : {'IS':0,'TSR':3,   'IF':1},

"IS2_TSR3_IF5" : {'IS':2,'TSR':3,   'IF':5},
"IS2_TSR3_IF4" : {'IS':2,'TSR':3,   'IF':4},
"IS2_TSR3_IF3" : {'IS':2,'TSR':3,   'IF':3},
"IS2_TSR3_IF2" : {'IS':2,'TSR':3,   'IF':2},
"IS2_TSR3_IF1" : {'IS':2,'TSR':3,   'IF':1}
}


# Import the necessary functions that correspond to each experiment code
from select_intervals import select_intervals_dr_cif, select_quant_intervals
from ts_representation import tsrep0, first_diff, getPeriodogramRepr, ar_coefs, second_diff
from extract_features import calculate_if1, calculate_if2, calculate_if3, calculate_if4, calculate_if5, calculate_if6

def experiment_components(experiment: str) -> dict:
    """
    Decode the experiment string into its corresponding components (functions).
    
    Parameters:
    - experiment (str): A string representing the experiment code, e.g., "IS2_TSR1_IF2".
    
    Returns:
    - components_func (list): A list of functions corresponding to the experiment code.
    """
    
    # Interval Selection Methods
    interval_selection = {
        "IS0": select_intervals_dr_cif,
        "IS2": select_quant_intervals
    }
    
    # Time Series Representation Methods
    time_series_rep = {
        "TSR1": [tsrep0, first_diff, getPeriodogramRepr],
        "TSR2": [tsrep0, first_diff, getPeriodogramRepr, ar_coefs],
        "TSR3": [tsrep0, first_diff, getPeriodogramRepr, second_diff]
    }
    
    # Interval Feature Calculation Methods
    interval_features = {
        "IF1": calculate_if1,
        "IF2": calculate_if2,
        "IF3": calculate_if3,
        "IF4": calculate_if4,
        "IF5": calculate_if5,
        "IF6": calculate_if6
    }
    
    # Split the experiment string into its components (e.g., "IS2_TSR1_IF2")
    components = experiment.split("_")
    
    # Get the corresponding functions based on the components
    interval_selection_func = interval_selection.get(components[0], None)
    ts_rep_funcs = time_series_rep.get(components[1], [])
    interval_features_func = interval_features.get(components[2], None)
    
    # Check if any of the components are missing
    if interval_selection_func is None:
        raise ValueError(f"Invalid interval selection method: {components[0]}")
    if not ts_rep_funcs:
        raise ValueError(f"Invalid time series representation method: {components[1]}")
    if interval_features_func is None:
        raise ValueError(f"Invalid interval feature method: {components[2]}")
    
    # Combine the functions into a dictionary
    components_func = {
        "interval_selection_func": interval_selection_func,
        "tsrep_funcs": ts_rep_funcs,
        "interval_features_func": interval_features_func
    }
    
    return components_func  


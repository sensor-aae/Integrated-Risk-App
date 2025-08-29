import pandas as pd
from risk_engine.credit import compute_el_table

def test_credit_el_basic():
    df = pd.DataFrame({"PD":[0.02, 2.0], "LGD":[0.4, 40.0], "EAD":[1000, 2000]})
    out, _ = compute_el_table(df)
    # Accept decimals or percents in input; EL should be around 8 and 16
    assert abs(out["EL"].iloc[0] - 8) < 1e-6
    assert abs(out["EL"].iloc[1] - 16) < 1e-6

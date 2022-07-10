import pandas as pd
import utils
from setting import *

def create_unit_profile(pv_df, wt_df, load_df, price_df):
    # reshape tables
    pv_df = pd.pivot_table(pv_df, values='solar_generation_mw', index=['datetime_beginning_ept'], columns=['area'], sort=False)
    wt_df = pd.pivot_table(wt_df, values='wind_generation_mw', index=['datetime_beginning_ept'], columns=['area'], sort=False)
    load_df = pd.pivot_table(load_df, values='mw', index=['datetime_beginning_ept'], columns=['load_area'], sort=False)
    price_df = pd.pivot_table(price_df, values='hrly_da_demand_bid', index=['datetime_beginning_ept'], columns=['area'], sort=False)

    # scale values
    for df in [pv_df, wt_df, load_df, price_df]:
        df /= df.max()
    print(f'Unit profile: {pv_df.max()}, {wt_df.max()}, {load_df.max()}, {price_df.max()}')

    return pv_df, wt_df, load_df, price_df

def create_save_profile(pv_df, wt_df, load_df, price_df):
    pv_df, wt_df, load_df, price_df =  create_unit_profile(pv_df, wt_df, load_df, price_df)

    pv_profile = pd.DataFrame({ 
        'pv3': pv_df['MIDATL'] * P_PV3_MAX,
        'pv4': pv_df['MIDATL'] * P_PV4_MAX,
        'pv5': pv_df['MIDATL'] * P_PV5_MAX,
        'pv6': pv_df['RFC'] * P_PV6_MAX,
        'pv8': pv_df['RFC'] * P_PV8_MAX,
        'pv9': pv_df['RFC'] * P_PV9_MAX,
        'pv10': pv_df['RTO'] * P_PV10_MAX,
        'pv11': pv_df['RTO'] * P_PV11_MAX
    })
    wt_profile = pd.DataFrame({
        'wt7': wt_df['MIDATL'] * P_WT7_MAX
    })
    load_profile = pd.DataFrame({
        'load_r1': load_df['AECO'] * P_LOADR1_MAX,
        'load_r3': load_df['BC'] * P_LOADR3_MAX,
        'load_r4': load_df['DPLCO'] * P_LOADR4_MAX,
        'load_r5': load_df['EASTON'] * P_LOADR5_MAX,
        'load_r6': load_df['JC'] * P_LOADR6_MAX,
        'load_r8': load_df['ME'] * P_LOADR8_MAX,
        'load_r10': load_df['PE'] * P_LOADR10_MAX,
        'load_r11': load_df['PEPCO'] * P_LOADR11_MAX,
    })
    price_profile = pd.DataFrame({
        'price': price_df['PJM_RTO'] * C_PRICE_MAX
    })

    # create csv files
    pv_profile.to_csv('./data/profile/pv_profile.csv')
    wt_profile.to_csv('./data/profile/wt_profile.csv')
    load_profile.to_csv('./data/profile/load_profile.csv')
    price_profile.to_csv('./data/profile/price_profile.csv')

if __name__ == '__main__':
    pv_df = pd.read_csv('./data/solar_gen.csv')
    wt_df = pd.read_csv('./data/wind_gen.csv')
    load_df = pd.read_csv('./data/hrl_load_metered.csv')
    price_df = pd.read_csv('./data/hrl_dmd_bids.csv')
    create_save_profile(pv_df, wt_df, load_df, price_df)

    pv_profile = pd.read_csv('./data/profile/pv_profile.csv')
    wt_profile = pd.read_csv('./data/profile/wt_profile.csv')
    load_profile = pd.read_csv('./data/profile/load_profile.csv')
    price_profile = pd.read_csv('./data/profile/price_profile.csv')
    # excess = pv_profile.sum(axis=1) + wt_profile.sum(axis=1) - load_profile.sum(axis=1)
    # surplus = excess[excess > 0]
    # print(surplus)
    # print(surplus.shape[0] / pv_profile.shape[0])
    utils.view_profile(pv_profile, wt_profile, load_profile, price_profile)
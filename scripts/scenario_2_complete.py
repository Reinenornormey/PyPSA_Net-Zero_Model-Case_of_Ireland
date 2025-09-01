# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 08:50:23 2025

@author: user
"""

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------
# Scenario 2: Coal & Oil Phase-Out
# with Demand Response, H₂ Metrics, and Full Visuals
# ---------------------------------------------

# 1. PARAMETERS & CONFIGURATION
DATE_RANGE     = pd.date_range("2024-01-01", "2024-12-31 23:45", freq="15min")
HOURS_PER_SNAP = 15/60

equip = {
    'Gas':       4001.123,
    'Wind':      5027.601,
    'Solar':     735.519,
    'Hydro':     234.75,
    'Other_RE':  520.116,
    'Other_nonRE':188.858
}

dr_threshold = 0.90 * 5639   # 90% of peak
dr_share     = 0.10          # 10% flexible

P_EL_MAX = 1000.0; ETA_EL = 0.70
P_FC_MAX = 1000.0; ETA_FC = 0.60

# 2. DATA LOADING & PREPROCESSING
data = pd.read_excel("Data.xlsx", sheet_name=None, index_col=0, parse_dates=True)
load_raw     = data['Demand'].squeeze()
wind_avail   = data['Wind'].squeeze()  / equip['Wind']
solar_avail  = data['Solar'].squeeze() / equip['Solar']
inter_flow   = data['Interconnector'].squeeze()
energy_price = data['Energy_price'].squeeze()
hydro_avail  = pd.Series(1.0, index=DATE_RANGE)

# Marginal costs time series
tc = pd.read_excel('Marginal_costs_all.xlsx')
tc['DateTime'] = pd.to_datetime(tc['DateTime'], dayfirst=True)
tc.set_index('DateTime', inplace=True)
mc_coal = tc['Coal, EUR/MWh'].reindex(DATE_RANGE).ffill()
mc_oil  = tc['Oil, EUR/MWh'].reindex(DATE_RANGE).ffill()
mc_gas  = tc['Gas, EUR/MWh'].reindex(DATE_RANGE).ffill()

# Emission weightings
raw_factors = {"Coal":(0.32,0.455),"Oil":(0.27,0.304),"Gas":(0.18,0.60)}
em_factors  = {t: ef/eff for t,(ef,eff) in raw_factors.items()}
co2_weight  = {t: pd.Series(v*HOURS_PER_SNAP, index=DATE_RANGE) for t,v in em_factors.items()}

# Demand Response
dr_active    = (load_raw > dr_threshold).astype(float)
dr_reduction = -dr_share * load_raw * dr_active
load_dr      = dr_reduction.reindex(DATE_RANGE).fillna(0)
load_sys     = load_raw.reindex(DATE_RANGE).ffill() + load_dr

# 3. NETWORK BUILDING
def build_network():
    n = pypsa.Network()
    n.set_snapshots(DATE_RANGE)

    # Buses
    for b in ['System','GB','H2']:
        n.add('Bus', b, v_nom=(380 if b!='H2' else 1.0))

    # Interconnector
    n.add('Link','Interconnector', bus0='System', bus1='GB',
          p_nom=1000, p_max_pu=1, p_min_pu=-1,
          efficiency=1, marginal_cost=0, carrier='HVDC')

    # Loads
    n.add('Load','Load_System', bus='System', p_set=load_sys, carrier='demand')
    n.add('Load','DR',          bus='System', p_set=load_dr,  carrier='demand_response')

    # Slack
    n.add('Generator','Slack', bus='System', control='Slack',
          v_set=1.02, q_min_pu=-0.3, q_max_pu=0.3,
          p_nom=0, H=10, damping=0.02, carrier='Slack')

    # Renewables
    for tech, avail in [('Wind',wind_avail),('Solar',solar_avail)]:
        n.add('Generator', tech, bus='System', p_nom=equip[tech], p_max_pu=avail,
              control='PV', v_set=1.0, q_min_pu=-0.44, q_max_pu=0.44,
              marginal_cost=0, H=2, damping=0.01, carrier=tech)

    # Synchronous
    for tech,p_nom,mc in [('Gas',equip['Gas'],mc_gas),('Hydro',equip['Hydro'],None)]:
        attrs = dict(bus='System', p_nom=p_nom,
                     p_min_pu=0.1, p_max_pu=0.8,
                     control='PV', v_set=1.01,
                     q_min_pu=-1.08, q_max_pu=1.08,
                     ramp_limit_up=0.04*15*p_nom,
                     ramp_limit_down=0.04*15*p_nom,
                     H=5, damping=0.015, carrier=tech)
        attrs['marginal_cost'] = mc if mc is not None else 20
        n.add('Generator', tech, **attrs)
        if tech=='Hydro': n.generators_t.p_max_pu.loc[:,tech]=hydro_avail

    # Others helper
    def add_other(name,p_nom,cost,q_lim):
        n.add('Generator', name, bus='System', p_nom=p_nom,
              p_min_pu=0.1, p_max_pu=1,
              control='PV', v_set=1.0, q_min_pu=-q_lim, q_max_pu=q_lim,
              ramp_limit_up=0.04*15*p_nom, ramp_limit_down=0.04*15*p_nom,
              H=2, damping=0.01, marginal_cost=cost, carrier=name)
    add_other('Other_RE', equip['Other_RE'],   20,    0.44)
    add_other('Other_nonRE', equip['Other_nonRE'], mc_oil, 1.08)

    # Storage
    for name,p_nom,energy in [('Battery',751.9,997.41),('PumpedHydro',292,1590),('Flywheel',0.401,0.14)]:
        max_h = energy/p_nom
        soc0  = 0.5*energy*(0.9 if name=='Battery' else 0.95 if name=='PumpedHydro' else 1)
        eff   = np.sqrt(0.95 if name=='Battery' else 0.92 if name=='PumpedHydro' else 0.97)
        n.add('StorageUnit', name, bus='System', p_nom=p_nom,
              max_hours=max_h,
              state_of_charge_initial=soc0,
              state_of_charge_min=0.15*energy,
              state_of_charge_max=(0.9 if name=='Battery' else 0.95 if name=='PumpedHydro' else 0.9)*energy,
              efficiency_store=eff, efficiency_dispatch=eff,
              H=(0.5 if name=='Battery' else 1 if name=='PumpedHydro' else 0.2),
              damping=0.05, carrier='storage')

    # Electrolyser
    n.add('Link','Electrolyser', bus0='System', bus1='H2',
          p_nom=P_EL_MAX, p_min_pu=0, p_max_pu=1,
          efficiency=ETA_EL, marginal_cost=energy_price-100,
          control='PQ', carrier='electrolyser')

    # Fuel Cell
    n.add('Link','FuelCell', bus0='H2', bus1='System',
          p_nom=P_FC_MAX, p_min_pu=0, p_max_pu=1,
          efficiency=ETA_FC, marginal_cost=energy_price+100,
          control='PQ', carrier='fuel_cell')

    # Constraints
    n.add('GlobalConstraint','CO2_budget', type='budget',
          carrier_attribute='carrier', weightings=co2_weight,
          constant=7.6e6, sense='<=' )
    n.add('GlobalConstraint','SNSP_95pct', type='share',
          carrier_attribute='carrier',
          positive_carriers={'Wind':pd.Series(1,index=DATE_RANGE), 'Solar':pd.Series(1,index=DATE_RANGE)},
          negative_carriers={'Load':pd.Series(1,index=DATE_RANGE)},
          constant=0.95, sense='<=' )

    # GB Grid
    n.add('Generator','GB_Grid', bus='GB', p_nom=1000, p_max_pu=1,
          control='PV', v_set=1.0, q_min_pu=-0.5, q_max_pu=0.5,
          marginal_cost=energy_price, H=5, damping=0.02, carrier='GB_grid')

    return n

# Build & optimize
network = build_network()
network.optimize()
print("Optimization complete.")

# 4. METRICS & EXPORT
snaps = network.snapshots
avail = pd.DataFrame({t:network.generators.loc[t,'p_nom']*network.generators_t.p_max_pu[t] for t in ['Wind','Solar']},index=snaps)
disp  = network.generators_t.p[['Wind','Solar']]
curt  = (avail-disp).clip(lower=0)
total_curt_mw     = curt.sum(axis=1)
total_curt_energy = total_curt_mw.sum()*HOURS_PER_SNAP
curtailment_rate  = total_curt_mw.sum()/avail.sum().sum()

fast_units = ['Battery','Flywheel','Electrolyser']
disp_fast = pd.concat([network.storage_units_t.p[['Battery','Flywheel']], network.links_t.p0[['Electrolyser']].clip(lower=0)],axis=1)
headroom  = pd.DataFrame({u:(network.storage_units.loc[u,'p_nom'] if u!='Electrolyser' else P_EL_MAX)-disp_fast[u] for u in fast_units}, index=snaps)
mean_headroom = headroom.sum(axis=1).mean()

P_el = network.links_t.p0['Electrolyser'].clip(lower=0)
U_el = P_el.sum()/(P_EL_MAX*len(snaps))
E_H2 = (P_el*HOURS_PER_SNAP*ETA_EL).sum()
m_H2 = E_H2*1000/33.3

P_fc     = network.links_t.p1['FuelCell']
E_fc_total = (-P_fc*HOURS_PER_SNAP).sum()

with pd.ExcelWriter("scenario2_results.xlsx") as writer:
    network.generators_t.p      .to_excel(writer,'gen_dispatch')
    network.loads_t.p_set       .to_excel(writer,'loads')
    network.links_t.p0          .to_excel(writer,'link_out')
    network.links_t.p1          .to_excel(writer,'link_in')
    network.storage_units_t.p   .to_excel(writer,'storage_dispatch')
    headroom                    .to_excel(writer,'headroom')
    curt                        .to_excel(writer,'curtailment_mw')

# 5. VISUALIZATION & SUMMARY
plt.figure(figsize=(12,4))
plt.plot(load_raw['2024-01-01':'2024-01-31'],label='Original')
plt.plot(load_sys['2024-01-01':'2024-01-31'],label='With DR')
plt.fill_between(load_dr['2024-01-01':'2024-01-31'].index,0,load_dr['2024-01-01':'2024-01-31'],alpha=0.3,label='DR')
plt.legend(); plt.title('Monthly Load & DR'); plt.ylabel('MW'); plt.tight_layout(); plt.show()


snaps15=snaps
snaps_hr=snaps[::4]


def plot_all():
   
    cmap=plt.get_cmap('tab10')
    # Stack 15min
    plt.figure(figsize=(14,4))
    gens15=network.generators_t.p[['Wind','Solar','Gas','Hydro','Other_RE','Other_nonRE','GB_Grid']]
    gens15['FuelCell']=-P_fc
    plt.stackplot(snaps15,*[gens15[c] for c in gens15.columns],labels=gens15.columns,colors=[cmap(i) for i in range(len(gens15.columns))],alpha=0.7)
    plt.title('Generation (15 mins timestep)'); plt.legend(ncol=2,fontsize='small'); plt.tight_layout(); plt.show()
    # Lines 15min
    plt.figure(figsize=(14,4))
    for i,c in enumerate(gens15.columns): plt.plot(snaps15,gens15[c],label=c,color=cmap(i),linewidth=0.8)
    plt.title('Generation (15 mins timestep)'); plt.legend(ncol=2,fontsize='small'); plt.tight_layout(); plt.show()
    # Stack hourly
    plt.figure(figsize=(14,4))
    gens_hr=gens15.iloc[::4]
    plt.stackplot(snaps_hr,*[gens_hr[c] for c in gens_hr.columns],labels=gens_hr.columns,colors=[cmap(i) for i in range(len(gens_hr.columns))],alpha=0.8)
    plt.title('Stack hourly'); plt.legend(ncol=2,fontsize='small'); plt.tight_layout(); plt.show()
    # Lines hourly
    plt.figure(figsize=(14,4))
    for i,c in enumerate(gens_hr.columns): plt.plot(snaps_hr,gens_hr[c],label=c,color=cmap(i),linewidth=1.2)
    plt.title('Lines hourly'); plt.legend(ncol=2,fontsize='small'); plt.tight_layout(); plt.show()

plot_all()

def plot_h2():
    plt.figure(figsize=(14,4))
    h2_prod = P_el * HOURS_PER_SNAP
    plt.plot(snaps, h2_prod, label='H₂ Production (MWh)', linewidth=1.2)
    plt.title('Hydrogen Production Over the Year')
    plt.ylabel('MWh per 15-min')
    plt.legend()
    plt.tight_layout()
    plt.show()   

plot_h2()


plt.figure(figsize=(14,4))
for u in network.storage_units_t.state_of_charge.columns:
    plt.plot(snaps,network.storage_units_t.state_of_charge[u],label=u)
plt.legend(); plt.title('State-of-Charge'); plt.ylabel('MWh'); plt.tight_layout(); plt.show()

plt.figure(figsize=(12,4))
total_curt_mw.plot(label='Curtailment MW')
plt.title(f'Curtailment Rate: {curtailment_rate:.2%}')
plt.ylabel('MW'); plt.tight_layout(); plt.show()

plt.figure(figsize=(12,4))
P_fc.plot(label='Fuel Cell MW')
plt.title(f'Fuel Cell Total: {E_fc_total:,.0f} MWh')
plt.ylabel('MW'); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.hist(headroom.sum(axis=1),bins=50)
plt.title(f'Static Headroom (mean {mean_headroom:.1f} MW)')
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
load_dr.resample('M').sum().plot(kind='bar')
plt.title('Monthly DR MWh'); plt.ylabel('MWh'); plt.tight_layout(); plt.show()

print(f"Curtailment Rate: {curtailment_rate:.2%}")
print(f"Total Curtailed Energy: {total_curt_energy:,.1f} MWh")
print(f"Electrolyser Utilization: {U_el:.2%}")
print(f"Total H₂ Energy: {E_H2:,.1f} MWh")
print(f"Total H₂ Mass: {m_H2:,.0f} kg")
print(f"Mean Static Headroom: {mean_headroom:.1f} MW")
print(f"Total Fuel Cell Generation: {E_fc_total:,.1f} MWh")

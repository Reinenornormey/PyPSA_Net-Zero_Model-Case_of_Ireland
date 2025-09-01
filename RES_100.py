# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 10:32:05 2025

@author: user
"""

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------
# Scenario 3: 100% Renewables, On–Off DR at 70% Threshold
# 30% Flexible Demand, SNSP = 100%
# ---------------------------------------------

# 1. PARAMETERS & CONFIGURATION
DATE_RANGE     = pd.date_range("2024-01-01", "2024-12-31 23:45", freq="15min")
HOURS_PER_SNAP = 15/60
PEAK_LOAD      = 5639.0
DR_THRESHOLD   = 0.85 * PEAK_LOAD   # 70% of peak
DR_SHARE       = 0.15               # 30% flexible

# Tech capacities
equip = {
    'Wind':      5027.601,
    'Solar':     753.519,
    'Hydro':     234.75,
    'Other_RE':  4000.0
}

# Storage specs: (p_nom, energy)
storage_specs = {
    'Battery':     (1200.0, 4800.0),   # 4h, increased power
    'PumpedHydro': (600.0, 3600.0),    # 6h, increased
    'Flywheel':    (150.0, 37.5)       # 0.25h, increased
}

# Electrolyser / FuelCell
P_EL_MAX = 10000.0; ETA_EL = 0.70
P_FC_MAX = 5000.0; ETA_FC = 0.60

# 2. DATA LOADING & PREPROCESSING
data = pd.read_excel("Data.xlsx", sheet_name=None, index_col=0, parse_dates=True)
load_raw     = data['Demand'].squeeze()
wind_avail   = data['Wind'].squeeze()  / equip['Wind']
solar_avail  = data['Solar'].squeeze() / equip['Solar']
hydro_avail  = pd.Series(1.0, index=DATE_RANGE)
energy_price = data['Energy_price'].squeeze()

# Demand Response on–off
dr_active    = (load_raw > DR_THRESHOLD).astype(float)
dr_reduction = -DR_SHARE * load_raw * dr_active
load_dr      = dr_reduction.reindex(DATE_RANGE).fillna(0)
load_sys     = load_raw.reindex(DATE_RANGE).ffill() + load_dr

# 3. NETWORK BUILDING
def build_network():
    n = pypsa.Network()
    n.set_snapshots(DATE_RANGE)

    # Buses
    n.add('Bus', 'System', v_nom=380)
    n.add('Bus', 'GB', v_nom=380)
    n.add('Bus', 'H2', v_nom=1.0)

    # Loads
    n.add('Load', 'Load_System', bus='System', p_set=load_sys, carrier='demand')
    n.add('Load', 'DR',          bus='System', p_set=load_dr,  carrier='demand_response')
    # Interconnector
    n.add('Link','Interconnector', bus0='System', bus1='GB',
          p_nom=1000, p_max_pu=1, p_min_pu=-1,
          efficiency=1, marginal_cost=energy_price, carrier='HVDC')

    # Slack (to cover any residual)
    n.add('Generator','Slack', bus='System', control='Slack',
          v_set=1.02, q_min_pu=-0.3, q_max_pu=0.3,
          p_nom=0, H=10, damping=0.02, carrier='Slack')

    # Renewables
    for tech, avail, cap in [
        ('Wind', wind_avail, equip['Wind']),
        ('Solar', solar_avail, equip['Solar']),
        ('Hydro', hydro_avail, equip['Hydro']),
        ('Other_RE', hydro_avail, equip['Other_RE']),  # always available
    ]:
        n.add('Generator', tech, bus='System', p_nom=cap, p_max_pu=avail,
              control='PV', v_set=1.0, q_min_pu=-0.44, q_max_pu=0.44,
              marginal_cost=0, H=2, damping=0.01, carrier=tech)

    # Storage Units
    for name, (p_nom, energy) in storage_specs.items():
        max_h = energy / p_nom
        soc0  = 0.5 * energy
        eff   = np.sqrt(0.95 if name=='Battery' else 0.92 if name=='PumpedHydro' else 0.97)
        n.add('StorageUnit', name, bus='System', p_nom=p_nom,
              max_hours=max_h,
              state_of_charge_initial=soc0,
              state_of_charge_min=0.15 * energy,
              state_of_charge_max=0.9 * energy,
              efficiency_store=eff, efficiency_dispatch=eff,
              H=0.5, damping=0.05, carrier='storage')

    # Electrolyser ↔ H2
    n.add('Link','Electrolyser', bus0='System', bus1='H2',
          p_nom=P_EL_MAX, p_min_pu=0, p_max_pu=1,
          efficiency=ETA_EL, marginal_cost=-300,
          control='PQ', carrier='electrolyser')
    n.add('Link','FuelCell',   bus0='H2',     bus1='System',
          p_nom=P_FC_MAX, p_min_pu=0, p_max_pu=1,
          efficiency=ETA_FC, marginal_cost=energy_price+100,
          control='PQ', carrier='fuel_cell')

    # Global SNSP 100%

    return n

# Build & Optimize
network = build_network()
network.optimize()
print("Optimization complete.")

# 4. METRICS & EXPORT
snaps = network.snapshots

# Availability & Dispatch for renewables
avail = pd.DataFrame({
    tech: equip[tech] * network.generators_t.p_max_pu[tech]
    for tech in ['Wind','Solar','Hydro','Other_RE']
}, index=snaps)
disp  = network.generators_t.p[['Wind','Solar','Hydro','Other_RE']]
curt  = (avail[['Wind','Solar']] - disp[['Wind','Solar']]).clip(lower=0)

# Curtailment metrics
total_curt_mw     = curt.sum(axis=1)
total_curt_energy = total_curt_mw.sum() * HOURS_PER_SNAP
curtailment_rate  = total_curt_energy / (avail[['Wind','Solar']].sum().sum() * HOURS_PER_SNAP)

# Fast units headroom
disp_fast = pd.concat([
    network.storage_units_t.p[['Battery','Flywheel']],
    network.links_t.p0[['Electrolyser']].clip(lower=0)
],axis=1)
fast_units = ['Battery','Flywheel','Electrolyser']
headroom  = pd.DataFrame({
    u: ( (storage_specs[u][0] if u!='Electrolyser' else P_EL_MAX) - disp_fast[u])
    for u in fast_units
}, index=snaps)
mean_headroom = headroom.sum(axis=1).mean()

# H2 & Fuel cell
P_el       = network.links_t.p0['Electrolyser'].clip(lower=0)
U_el       = P_el.sum() / (P_EL_MAX * len(snaps))
E_H2       = (P_el * HOURS_PER_SNAP * ETA_EL).sum()
m_H2       = E_H2 * 1000 / 33.3
P_fc       = network.links_t.p1['FuelCell']
E_fc_total = (-P_fc * HOURS_PER_SNAP).sum()

# Export
with pd.ExcelWriter("scenario3_results.xlsx") as writer:
    network.generators_t.p      .to_excel(writer,'gen_dispatch')
    network.loads_t.p_set       .to_excel(writer,'loads')
    network.storage_units_t.p   .to_excel(writer,'storage_dispatch')
    curt                        .to_excel(writer,'curtailment_mw')
    headroom                    .to_excel(writer,'headroom')

# 5. VISUALIZATION & SUMMARY
# Load & DR monthly (January)
plt.figure(figsize=(12,4))
plt.plot(load_raw['2024-01-01':'2024-01-31'],label='Original')
plt.plot(load_sys['2024-01-01':'2024-01-31'],label='With DR')
plt.fill_between(load_dr['2024-01-01':'2024-01-31'].index,0,load_dr['2024-01-01':'2024-01-31'],alpha=0.3,label='DR')
plt.legend(); plt.title('Monthly Load & DR'); plt.ylabel('MW'); plt.tight_layout(); plt.show()

# Generation stacks and lines
snaps15 = snaps
snaps_hr = snaps[::4]

def plot_all():
    cmap=plt.get_cmap('tab10')
    gens15=pd.concat([network.generators_t.p[['Wind','Solar','Hydro','Other_RE']],
                      pd.DataFrame({'FuelCell': -P_fc}, index=snaps15)], axis=1)
    # Stack 15min
    plt.figure(figsize=(14,4))
    plt.stackplot(snaps15,*[gens15[c] for c in gens15.columns],labels=gens15.columns,alpha=0.7)
    plt.title('Generation (15 min)'); plt.legend(ncol=2,fontsize='small', loc='upper left', bbox_to_anchor=(1, 1)); plt.tight_layout(); plt.show()
    # Lines 15min
    plt.figure(figsize=(14,4))
    for c in gens15.columns: plt.plot(snaps15,gens15[c],linewidth=0.8)
    plt.title('Generation Time Series (15 min)'); plt.legend(ncol=2,fontsize='small', loc='upper left', bbox_to_anchor=(1, 1)); plt.tight_layout(); plt.show()
    # Stack hourly
    gens_hr=gens15.iloc[::4]
    plt.figure(figsize=(14,4))
    plt.stackplot(snaps_hr,*[gens_hr[c] for c in gens_hr.columns],labels=gens_hr.columns,alpha=0.8)
    plt.title('Generation (Hourly)'); plt.legend(ncol=2,fontsize='small', loc='upper left', bbox_to_anchor=(1, 1)); plt.tight_layout(); plt.show()
    # Lines hourly
    plt.figure(figsize=(14,4))
    for c in gens_hr.columns: plt.plot(snaps_hr,gens_hr[c],linewidth=1.2)
    plt.title('Generation Time Series (Hourly)'); plt.tight_layout(); plt.legend(ncol=2,fontsize='small', loc='upper left', bbox_to_anchor=(1, 1)); plt.show()

plot_all()

# H2 production over year
plt.figure(figsize=(14,4))
plt.plot(snaps, P_el * HOURS_PER_SNAP, label='H₂ Prod (MWh)')
plt.title('Hydrogen Production Over the Year')
plt.ylabel('MWh per snapshot'); plt.legend(); plt.tight_layout(); plt.show()

# State-of-Charge
plt.figure(figsize=(14,4))
for u in network.storage_units_t.state_of_charge.columns:
    plt.plot(snaps,network.storage_units_t.state_of_charge[u],label=u)
plt.legend(); plt.title('State-of-Charge'); plt.ylabel('MWh'); plt.tight_layout(); plt.show()

# Curtailment time series
plt.figure(figsize=(12,4))
plt.plot(total_curt_mw,label='Curtailment MW')
plt.title(f'Curtailment Rate: {curtailment_rate:.2%}'); plt.ylabel('MW'); plt.tight_layout(); plt.show()

# Fuel Cell output
plt.figure(figsize=(12,4))
plt.plot(P_fc, label='Fuel Cell MW')
plt.title(f'Fuel Cell Total: {E_fc_total:,.0f} MWh'); plt.tight_layout(); plt.show()

# Headroom histogram
plt.figure(figsize=(8,4))
plt.hist(headroom.sum(axis=1),bins=50)
plt.title(f'Static Headroom (mean {mean_headroom:.1f} MW)'); plt.tight_layout(); plt.show()

# Monthly DR
plt.figure(figsize=(8,4))
load_dr.resample('M').sum().plot(kind='bar')
plt.title('Monthly DR MWh'); plt.ylabel('MWh'); plt.tight_layout(); plt.show()

# Print summary
print(f"Curtailment Rate: {curtailment_rate:.2%}")
print(f"Total Curtailed Energy: {total_curt_energy:,.1f} MWh")
print(f"Electrolyser Utilization: {U_el:.2%}")
print(f"Total H₂ Energy: {E_H2:,.1f} MWh")
print(f"Total H₂ Mass: {m_H2:,.0f} kg")
print(f"Mean Static Headroom: {mean_headroom:.1f} MW")
print(f"Total Fuel Cell Generation: {E_fc_total:,.1f} MWh")


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 1. Prepare your bus coordinates
coords = {
    'System': (-8.240473, 53.419751),
    'GB':     ( 0.1,      51.5     ),
    'H2':     (-7.5,      53.0     ),
}
network.buses[['x','y']] = pd.DataFrame(coords).T

# 2. Build a Cartopy map
fig = plt.figure(figsize=(9,7))
ax  = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
ax.set_extent([-10, 2, 50, 56], crs=ccrs.PlateCarree())

# 3. Plot buses
bus_xy = network.buses[['x','y']]
ax.scatter(
    bus_xy['x'], bus_xy['y'],
    s=80,               # marker size
    facecolor='white',
    edgecolor='black',
    linewidth=0.8,
    zorder=10,
    transform=ccrs.PlateCarree()
)

# 4. Plot links
for _, link in network.links.iterrows():
    b0, b1 = link.bus0, link.bus1
    x0, y0 = coords[b0]
    x1, y1 = coords[b1]
    ax.plot(
        [x0, x1], [y0, y1],
        color='steelblue',
        linewidth=2.0,
        alpha=0.7,
        transform=ccrs.PlateCarree(),
        zorder=5
    )

# 5. Annotate bus names
for name, (x, y) in coords.items():
    ax.text(x + 0.1, y + 0.1, name,
            fontsize=12, fontweight='bold',
            transform=ccrs.PlateCarree())

plt.title("Network Topology: Buses & Interconnectors")
plt.show()




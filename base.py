# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 08:43:19 2025

@author: user
"""

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# 1. INPUT DATA & PREPROCESSING
# ----------------------

# Define snapshots: 2024 full year at 15-min resolution
date_range = pd.date_range(
    start="2024-01-01 00:00",
    end="2024-12-31 23:45",
    freq="15min"
)

# Load time series data from Excel
load = pd.read_excel("Data.xlsx", sheet_name="Demand", index_col=0, parse_dates=True).squeeze()
wind_avail  = pd.read_excel("Data.xlsx", sheet_name="Wind", index_col=0, parse_dates=True).squeeze() / 5027.601
solar_avail = pd.read_excel("Data.xlsx", sheet_name="Solar", index_col=0, parse_dates=True).squeeze() / 753.519
inter_flow  = pd.read_excel("Data.xlsx", sheet_name="Interconnector", index_col=0, parse_dates=True).squeeze()
energy_price= pd.read_excel("Data.xlsx", sheet_name="Energy_price", index_col=0, parse_dates=True).squeeze()
hydro_avail  = pd.Series(1.0, index=date_range)

# Load marginal costs into series indexed by snapshots
mc_df = pd.read_excel('Marginal_costs_all.xlsx')
mc_df['DateTime'] = pd.to_datetime(mc_df['DateTime'], dayfirst=True)
mc_df.set_index('DateTime', inplace=True)
mc_coal = mc_df['Coal, EUR/MWh'].reindex(date_range).fillna(method='ffill')
mc_oil  = mc_df['Oil, EUR/MWh'].reindex(date_range).fillna(method='ffill')
mc_gas  = mc_df['Gas, EUR/MWh'].reindex(date_range).fillna(method='ffill')

# Emission factors & CO2 weighting per snapshot
raw_factors = {"Coal":(0.32,0.455),
               "Oil": (0.27,0.304),
               "Gas": (0.18,0.60)}
em_factors = {tech: ef/eff for tech,(ef,eff) in raw_factors.items()}
hours_per_snap = 15 / 60
co2_weightings = {tech: pd.Series(val * hours_per_snap, index=date_range)
                   for tech, val in em_factors.items()}

# ----------------------
# 2. NETWORK SETUP
# ----------------------

n = pypsa.Network()
n.set_snapshots(date_range)

# Buses
n.add("Bus", "System", v_nom=380)
n.add("Bus", "GB", v_nom=380)

# Interconnector Link
df_link = inter_flow.reindex(date_range).fillna(0)
# Use time-varying limits: set p_max_pu to df_link / capacity where needed
n.add(
    "Link", "Interconnector",
    bus0="System", bus1="GB",
    p_nom=1000,
    p_max_pu=1.0, p_min_pu=-1.0,
    efficiency=1.0,
    marginal_cost=0.0,
    carrier="HVDC"
)
# later adjust link snapshots if required

# Load
n.add("Load", "Load_System", bus="System", p_set=load, carrier="load")

# Slack for reference & voltage support
n.add(
    "Generator", "SlackBus",
    bus="System",
    control="Slack",
    v_set=1.02,
    q_min_pu=-0.3, q_max_pu=0.3,
    p_nom=0.0,
    H=10.0,
    damping=0.02,
    carrier="Slack"
)

# ----------------------
# 3. GENERATORS
# ----------------------

# Wind & Solar with virtual inertia
for tech, avail, p_nom in [("Wind", wind_avail, 5027.601), ("Solar", solar_avail, 753.519)]:
    n.add(
        "Generator", tech,
        bus="System",
        p_nom=p_nom,
        p_max_pu=avail,
        marginal_cost=0.0,
        control="PV",
        v_set=1.00,
        q_min_pu=-0.44, q_max_pu=0.44,
        H=2.0,        # virtual inertia constant
        damping=0.01, # inverter damping
        carrier=tech
    )

# Synchronous generators (Coal, Oil, Gas, Hydro)
sync_specs = [
    ("Coal", 862.5, mc_coal),
    ("Oil", 589.4,  mc_oil),
    ("Gas", 4001.123, mc_gas),
    ("Hydro", 234.75, None)
]
for tech, p_nom, mc in sync_specs:
    attrs = {
        "bus": "System",
        "p_nom": p_nom,
        "p_min_pu": 0.1,
        "p_max_pu": 0.8,
        "control": "PV",
        "v_set": 1.01,
        "q_min_pu": -1.08,
        "q_max_pu": 1.08,
        "H": 5.0,
        "damping": 0.015,
        "ramp_limit_up": 0.04 * 15 * p_nom,
        "ramp_limit_down": 0.04 * 15 * p_nom,
        "carrier": tech
    }
    if mc is not None:
        attrs["marginal_cost"] = mc
    else:
        attrs["marginal_cost"] = 20
    n.add("Generator", tech, **attrs)
    if tech == "Hydro":
        n.generators_t.p_max_pu["Hydro"] = hydro_avail

# Other renewables & non-RE
def add_other(name, p_nom, cost, q_lim):
    n.add(
        "Generator", name,
        bus="System",
        p_nom=p_nom,
        marginal_cost=cost,
        p_min_pu=0.1,
        p_max_pu=1.0,
        ramp_limit_up=0.04 * 15 * p_nom,
        ramp_limit_down=0.04 * 15 * p_nom,
        control="PV",
        v_set=1.00,
        q_min_pu=-q_lim,
        q_max_pu= q_lim,
        H=2.0,
        damping=0.01,
        carrier=name
    )

add_other("Other_RE",     520.116, 20.0, 0.44)
add_other("Other_nonRE",  188.858, mc_oil, 1.08)

# ----------------------
# 4. STORAGE
# ----------------------

# Battery
max_h = 997.41 / 751.90
soc0  = 0.5 * 0.9 * 997.41
n.add(
    "StorageUnit", "Battery",
    bus="System",
    p_nom=751.90,
    max_hours=max_h,
    state_of_charge_initial=soc0,
    state_of_charge_min=0.15 * 997.41,
    state_of_charge_max=0.9 * 997.41,
    efficiency_store=np.sqrt(0.95),
    efficiency_dispatch=np.sqrt(0.95),
    H=0.5,
    damping=0.05,
    carrier="storage"
)

# Pumped Hydro
max_h = 1590.0 / 292.00
soc0  = 0.5 * 0.95 * 1590.0
n.add(
    "StorageUnit", "PumpedHydro",
    bus="System",
    p_nom=292.00,
    max_hours=max_h,
    state_of_charge_initial=soc0,
    state_of_charge_min=0.15 * 1590.0,
    state_of_charge_max=0.95 * 1590.0,
    efficiency_store=np.sqrt(0.92),
    efficiency_dispatch=np.sqrt(0.92),
    H=1.0,
    damping=0.05,
    carrier="storage"
)

# Flywheel
max_h = 0.14 / 0.401
soc0  = 0.5 * 1.0 * 0.14
n.add(
    "StorageUnit", "Flywheel",
    bus="System",
    p_nom=0.401,
    max_hours=max_h,
    state_of_charge_initial=soc0,
    state_of_charge_min=0.0,
    state_of_charge_max=0.9 * 0.14,
    efficiency_store=np.sqrt(0.97),
    efficiency_dispatch=np.sqrt(0.97),
    H=0.2,
    damping=0.05,
    carrier="storage"
)


# ----------------------
# 5. GLOBAL CONSTRAINTS
# ----------------------

# CO2 budget
n.add(
    "GlobalConstraint", "CO2_budget",
    type="budget",
    carrier_attribute="carrier",
    weightings=co2_weightings,
    constant=7.6e6,
    sense="<="
)

# SNSP limit
positive = {"Wind": pd.Series(1.0, index=date_range),
            "Solar":pd.Series(1.0, index=date_range)}
negative = {"Load": pd.Series(1.0, index=date_range)}
n.add(
    "GlobalConstraint", "SNSP_75pct",
    type="share",
    carrier_attribute="carrier",
    positive_carriers=positive,
    negative_carriers=negative,
    constant=0.75,
    sense="<="
)

# ----------------------
# 6. VIRTUAL GB GRID
# ----------------------

n.add(
    "Generator", "GB_Grid",
    bus="GB",
    p_nom=1000,
    p_max_pu=1.0,
    marginal_cost=energy_price,
    control="PV",
    v_set=1.00,
    q_min_pu=-0.5,
    q_max_pu=0.5,
    H=5.0,
    damping=0.02,
    carrier="GB_grid"
)

# ----------------------
# 7. OPTIMIZATION & DYNAMICS
# ----------------------

# Steady-state optimal dispatch
n.optimize()
snaps = n.snapshots

# Specify just the two renewables
renewables = ["Wind", "Solar"]

# 1. Build availability: p_nom * p_max_pu (time series)
avail = pd.DataFrame({
    tech: n.generators.loc[tech, "p_nom"] * n.generators_t.p_max_pu[tech]
    for tech in renewables
}, index=snaps)

# 2. Extract dispatched power [MW]
disp = n.generators_t.p[renewables]

# 3. Compute per‐timestep curtailment (avail – disp), clipped ≥ 0
curt = (avail - disp).clip(lower=0)

# 4. Sum up and form the rate
total_curt = curt.sum().sum()      # sum over tech & time
total_avail = avail.sum().sum()    # sum over tech & time

curtailment_rate = total_curt / total_avail
lost_energy=total_curt*0.25

print(f"Renewable curtailment rate (Wind+Solar): {curtailment_rate:.2%}")

# 1. Define fast‐responding units
fast = ["Battery", "Flywheel"]

# 2. Extract their dispatched power time series (positive = discharging)
p_disp = n.storage_units_t.p[fast]

# 3. Compute per‐timestep headroom: P_nom − P_dispatch (charging doesn’t reduce headroom)
headroom = pd.DataFrame(
    {u: n.storage_units.loc[u, "p_nom"] - p_disp[u].clip(lower=0)
     for u in fast},
    index=n.snapshots
)
# Sum headroom across units
headroom["total"] = headroom.sum(axis=1)

# 4. Average over all snapshots
mean_headroom = headroom["total"].mean()

print(f"Average static headroom (Battery + Flywheel): {mean_headroom:.1f} MW")


# 1) Prepare timeseries
snaps = n.snapshots

# Load
load = n.loads_t.p_set["Load_System"]

# Generators (you can adjust the list/order/colors if you like)
gen_list = [
    "Wind", "Solar", "Coal", "Oil", "Gas",
    "Hydro", "Other_RE", "Other_nonRE", "GB_Grid"
]
gens = n.generators_t.p[gen_list]

# Storage dispatch (positive = discharge)
storage = n.storage_units_t.p[["Battery", "PumpedHydro", "Flywheel"]]
storage_discharge = storage.clip(lower=0).sum(axis=1)

# Interconnector import: assume p0 < 0 means import into System
inter = n.links_t.p0["Interconnector"]
inter_import = (-inter).clip(lower=0)

# 2) Figure 1: load + gens + shading
plt.figure(figsize=(14, 6))
for tech in gen_list:
    plt.plot(snaps, gens[tech], label=tech, linewidth=1.0)
# shade storage discharge
plt.fill_between(
    snaps,
    0,
    storage_discharge,
    color="red",
    alpha=0.3,
    label="Storage dispatch"
)
# shade interconnector import
plt.fill_between(
    snaps,
    0,
    inter_import,
    color="blue",
    alpha=0.3,
    label="Interconnector import"
)
plt.legend(ncol=2, fontsize="small")
plt.ylabel("Power (MW)")
plt.title("Generation & Contributions from Storage/Interconnector")
plt.tight_layout()
plt.show()

# 3) Figure 2: SOC of storage units
plt.figure(figsize=(14, 4))
soc = n.storage_units_t.state_of_charge[["Battery", "PumpedHydro", "Flywheel"]]
for u in soc.columns:
    plt.plot(snaps, soc[u], label=u, linewidth=1.2)
plt.legend(fontsize="small")
plt.ylabel("State of Charge (MWh)")
plt.title("Storage State‐of‐Charge Over Time")
plt.tight_layout()
plt.show()




# Export results
with pd.ExcelWriter("dynamic_results.xlsx") as writer:
    n.generators_t.p.to_excel(writer, sheet_name="gen_dispatch")
    n.loads_t.p_set.to_excel(writer, sheet_name="load_profile")
    n.links_t.p0.to_excel(writer, sheet_name="interconnector_out")
    n.links_t.p1.to_excel(writer, sheet_name="interconnector_in")
    n.storage_units_t.p.to_excel(writer, sheet_name="storage_dispatch")

# # Time-domain simulation using built-in dynamic API
# if hasattr(n, 'dynamic'):
#     dyn = n.dynamic(0.1)
# else:
#     raise ImportError("PyPSA version does not support .dynamic(). Please upgrade to v0.33+.")

# # Plot frequency
# plt.figure()
# plt.plot(dyn.time, dyn.omega['SlackBus'])
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [pu]')
# plt.title('System Frequency Response')

# # Plot voltage
# plt.figure()
# plt.plot(dyn.time, dyn.v_mag['System'])
# plt.xlabel('Time [s]')
# plt.ylabel('Voltage [pu]')
# plt.title('Bus Voltage Response')

# plt.show()




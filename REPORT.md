# Risultati finali QoS–carbon e robustezza

Il riferimento per la tesi è il [report di robustezza](results/qos_carbon/robustness/README.md),
con [tabella rappresentativa](results/qos_carbon/robustness/representative_configurations.md)
e figure finali [budget–emissioni](results/qos_carbon/robustness/figures/delay_budget_vs_emissions_saved.pdf)
e [budget–QoS](results/qos_carbon/robustness/figures/delay_budget_vs_qos.pdf).
Include quattro finestre temporali, capacità 880/512/256 nodi, griglie 15/60/240 min,
domanda terminale, cinque contrasti di ablation, fairness e classi di job.

Il [confronto principale](results/qos_carbon/README.md)
confronta EASY, greedy e improved sui budget 0/1/3/6/12/24 h e sulle quattro
combinazioni di informazione; include Pareto frontier, code QoS, costo marginale,
carbon benefit retained, controlli separati di runtime/potenza/forecast e MILP
su piccole istanze con gap certificati sullo stesso obiettivo del solver.

Riproduzione: `make robustness ARGS='--workers 2'`; solo tabelle e figure dai CSV:
`make robustness-report`. `make qos-carbon` ricrea il confronto principale.
Assumptions, limitations, threats to validity, protocollo con hash e riferimenti
EASY sono documentati nei report. Le finestre sono disgiunte per release, ma
vicine nello stesso test retrospettivo: non costituiscono un nuovo holdout o una
validazione stagionale. Nessun nuovo meccanismo di scheduling è introdotto.

Il [round storico sulla frontiera](results/qos_carbon/archived_scheduler_frontier.md)
è conservato separatamente: la decisione RESTORE di quel round non descrive lo
stato di implementazione attuale.

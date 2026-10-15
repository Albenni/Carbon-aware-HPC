# Tesi — Carbon-aware scheduling per sistemi HPC

Il sistema è implementato; il lavoro corrente è la validazione sperimentale e
la preparazione della tesi. Risultati e comandi aggiornati sono in [REPORT.md](REPORT.md).

## Sistema realizzato

- Carbon accounting con potenza media per il planning e profili PM100 misurati
  per le emissioni ex post; durata e potenza del job restano invarianti allo start.
- Carbon intensity storica IT-NO, forecasting e archivio degli snapshot emessi:
  ogni decisione realistica legge solo informazioni già disponibili.
- Simulatore a eventi con timestamp continui, cluster omogeneo espresso in nodi,
  job non preemptive, baseline FCFS/EASY e power cap.
- Scheduler greedy, receding horizon, congestion awareness, obiettivi con prezzo
  del ritardo e benefit-rate, budget fisso o proporzionale alla durata.
- Predittori di runtime/potenza e controlli che separano gli errori dei job dal
  forecast; MILP offline su piccole istanze con dominio e gap espliciti.

L'improved selezionato sulla validation usa receding horizon, benefit-rate e
soglia congelata. La verifica finale non aggiunge meccanismi allo scheduler.

## Output per la tesi

Il [confronto principale](results/qos_carbon/README.md) misura EASY, greedy e
improved sui budget 0/1/3/6/12/24 h e sulle quattro combinazioni di informazione.
Il [report di robustezza](results/qos_carbon/robustness/README.md) aggiunge quattro
finestre temporali, 880/512/256 nodi, griglie 15/60/240 minuti, domanda terminale,
cinque contrasti di ablation, fairness/tail QoS e classi short/long × small/large.
Fornisce figure PNG/PDF e una tabella rappresentativa CSV/Markdown/LaTeX.

```bash
make qos-carbon ARGS='--workers 2'
make robustness ARGS='--workers 2'
make robustness-report
```

[Formalizzazione](docs/2%20-%20Model%20Formalization.md),
[dataset e filtering](docs/PM100_features.md),
[esperimenti e riproduzione](src/experiments/README.md).

## Ambito delle conclusioni

Si stimano emissioni operative attribuite ai profili dei nodi, non l'impronta
completa del datacenter o emissioni marginali evitate. Il test è retrospettivo,
le finestre sono adiacenti e il workload filtrato non ricostruisce tutta la
domanda storica. Capacità ridotte e domanda terminale sono controlli distinti;
il budget carbonico non garantisce un limite all'attesa da contesa. Le code e
le differenze per classe vanno discusse insieme alle medie.

Rimangono la redazione dei capitoli e la revisione della letteratura.
Validazione stagionale/su altri siti, backlog iniziale completo e rolling-horizon
MILP sono future work, non componenti mancanti del sistema valutato.

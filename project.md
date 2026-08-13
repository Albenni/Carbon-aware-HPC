# Tesi — Carbon-aware scheduling per sistemi HPC

## Obiettivo generale

Progettare, implementare e valutare strategie di scheduling per sistemi HPC che minimizzino o bilancino i seguenti aspetti:

1. **Carbon footprint**
2. **Consumo energetico**
3. **Prestazioni di sistema**
4. **Qualità del servizio per gli utenti**

> Nota: anche concentrarsi inizialmente **solo sulla dimensione (1) carbon footprint** potrebbe essere già sufficiente e più gestibile.

## Strumenti e approccio metodologico

La tesi sfrutterà tre componenti principali:

- **Predizioni ML**
  - Possibile ispirarsi a modelli già esistenti in letteratura
- **Ottimizzazione matematica**
  - In particolare, possibile uso di **Constraint Programming (CP)**
- **Simulazione**
  - Valutazione su **trace reali di un sistema HPC**

---

# Fasi operative

## 1. Revisione della letteratura mirata

Effettuare una ricognizione iniziale della letteratura, senza necessariamente leggere in dettaglio tutti gli articoli, ma per costruire un quadro chiaro del contesto e preparare la futura sezione di related work della tesi.

### Temi da esplorare

#### HPC job scheduling

- **FCFS**
- **EASY Backfilling**
- **Priority-based scheduling**

#### Power-aware / Energy-aware scheduling

- Obiettivi tipici
- Metriche tipiche
- Trade-off tra energia e prestazioni

#### Carbon-aware computing

- Differenza tra **energy-aware** e **carbon-aware**
- Uso dei segnali di **carbon intensity** della rete elettrica
- Impatto del momento di esecuzione dei job sulle emissioni

#### Optimization-based scheduling in HPC

- **MILP**
- **Constraint Programming (CP)**

### Domande guida per la letteratura

- La **carbon footprint** dipende solo dall’energia consumata oppure anche dal **momento temporale** in cui il job viene eseguito?
- Serve una **predizione diretta della CO₂ per job**, oppure basta combinare:
  - potenza/energia prevista,
  - durata prevista,
  - carbon intensity del sistema elettrico nel tempo?
- La differenza tra **scheduling carbon-aware** e **energy-aware** è probabilmente cruciale e potrebbe diventare un cardine della tesi.

## 2. Formalizzazione del modello di carbon footprint

Definire un modello sufficientemente semplice ma utile per lo scheduling e la simulazione.

### Possibili modelli iniziali

- Modello a **potenza media costante**
- Modello a **energia totale + carbon intensity media**
- Eventuali semplificazioni progressive da raffinarsi in seguito

### Questioni aperte

- Come gestire i **job multi-nodo**?
  - Somma sui nodi?
  - Somma su partizioni?
- Consideriamo differenze di **efficienza hardware** tra nodi/partizioni?
- È davvero necessario prevedere direttamente la CO₂ come target ML?
  - Probabilmente **no**
  - Più plausibile combinare predizioni di:
    - energia,
    - potenza,
    - tempo/durata

### Domande guida

- La **carbon intensity** va presa:
  - come dato storico reale,
  - come forecast,
  - come serie sintetica?
- Quale **granularità temporale** usare?
  - 5 minuti
  - 15 minuti
  - 1 ora
- Il simulatore opererà in **tempo continuo** o **tempo discreto**?
- Consideriamo solo **operational carbon** oppure anche **embodied carbon**?
  - Suggerimento: considerare **solo operational carbon**

## 3. Modellazione della carbon intensity nel simulatore

Studiare come rappresentare nel simulatore il consumo di CO₂ e i segnali esterni di carbon intensity.

### Opzioni possibili

1. **Tracce storiche reali** di carbon intensity
2. **Serie temporali pubbliche** e dati reali di grid intensity
3. **Profili sintetici**, nel caso in cui i dati reali non siano disponibili

### Componente da introdurre nel simulatore

- `CarbonIntensityProvider`
  - Input: timestamp
  - Output: carbon intensity corrispondente

## 4. Integrazione dei modelli di consumo nel simulatore

Integrare nel simulatore i modelli necessari alla valutazione delle strategie di scheduling.

### Componenti da definire

- Modello di **potenza media** oppure di **energia**
- Modello di **durata** del job
- Eventuale modello per predire la **carbon intensity** (se non disponibile come serie esterna)

### Vincolo importante

- Usare solo le **feature note al submission time**

## 5. Baseline scheduler da implementare

Definire e implementare alcuni scheduler di riferimento.

### Possibili baseline

- **FCFS**
- **EASY backfilling**
- **Power-aware / Energy-aware greedy scheduler**

## 6. Metriche di valutazione

### Metriche carbon / energia

- **CO₂ totale emessa**
- **CO₂ media per job**
- **Energia totale consumata**
- **Picchi di potenza** o di energia nel tempo

### Metriche prestazionali HPC

- **Average waiting time**
- **Average turnaround time**
- **Bounded slowdown**

## 7. Progettazione delle strategie carbon-aware

Fase successiva, da sviluppare dopo aver definito:

- modello,
- simulatori,
- baseline,
- metriche.

Questa parte costituirà probabilmente il cuore sperimentale della tesi.

# Ipotesi di lavoro iniziali

## Ipotesi principali

- La differenza tra **energy-aware** e **carbon-aware scheduling** è centrale.
- Non è necessario prevedere direttamente la CO₂ per job.
- È probabilmente sufficiente combinare:
  - energia/potenza prevista,
  - durata prevista,
  - carbon intensity nel tempo.
- Conviene limitarsi inizialmente a **operational carbon**.
- Un modello semplificato può essere adeguato nella fase iniziale.

# Questioni aperte da risolvere progressivamente

- Come modellare in modo semplice ma realistico la carbon footprint?
- Quale sorgente usare per la carbon intensity?
- Quale granularità temporale adottare?
- Come gestire job multi-nodo e differenze hardware?
- Come integrare tutto nel simulatore senza introdurre complessità eccessiva?
- Quali trade-off emergeranno tra:
  - emissioni,
  - consumo energetico,
  - prestazioni,
  - QoS?

---

# Prossimi passi concreti

1. Fare una **scansione iniziale della letteratura**
2. Chiarire la distinzione tra:
   - **energy-aware scheduling**
   - **carbon-aware scheduling**
3. Scegliere un primo **modello semplificato di carbon footprint**
4. Definire come rappresentare la **carbon intensity** nel simulatore
5. Implementare le **baseline**
6. Definire le **metriche**
7. Passare poi alla progettazione delle strategie **carbon-aware**

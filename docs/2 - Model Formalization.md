## 2. Formalizzazione del modello di carbon footprint

> **Stato: completata.** Il modello operativo, le formule con unità esplicite,
> le assunzioni PM100 e l'API implementata sono documentati in
> [`src/README.md`](../src/README.md). Le domande sotto restano come traccia
> delle decisioni considerate durante la progettazione.

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

---

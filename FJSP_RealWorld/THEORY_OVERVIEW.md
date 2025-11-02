# Fondamenti teorici per l'FJSP con DRL

Questo documento raccoglie gli elementi chiave da conoscere per spiegare e estendere il progetto FJSP_RealWorld.

---

## 1. Flexible Job-Shop Scheduling Problem (FJSP)
- **Obiettivo**: minimizzare il `makespan` (tempo totale di completamento) assegnando ogni operazione a una tra più macchine/attrezzature compatibili.
- **Vincoli**:
  - Sequenza di operazioni per ogni job (precedenza).
  - Ogni macchina esegue una sola operazione alla volta.
  - Tempi di lavorazione dipendono da job, operazione e macchina (compatibilità).
- **Particolarità**: la flessibilità nella scelta della macchina rende il problema combinatorio e NP-hard.

---

## 2. Rappresentazione a grafo disgiuntivo
- **Nodi**: operazioni identificate da job e indice operazione.
- **Archi congiuntivi**: collegano operazioni successive dello stesso job (ordine fisso).
- **Archi disgiuntivi**: collegano operazioni che competono per la stessa macchina (ordine da definire).
- **Grafo dinamico**: a ogni assegnazione si aggiornano archi/matrici per riflettere l'ultima sequenza decisa.
- Questo grafo alimenta la **Graph Isomorphism Network (GIN)** che produce gli embedding degli stati.

---

## 3. Reinforcement Learning
- **Stato** `s_t`: feature dei nodi, maschere di disponibilità e grafo delle operazioni non ancora schedulate.
- **Azione** `a_t`: coppia (operazione da eseguire, macchina/attrezzatura scelta).
- **Transizione**: l'ambiente aggiorna orari, maschere e grafo dopo l'azione.
- **Reward** `r_t`: differenza (in negativo) tra il nuovo lower bound del makespan e il precedente. Reward positivo se il makespan stimato diminuisce.
- **Episodio**: termina quando tutte le operazioni sono pianificate.

---

## 4. Multi-actor Proximal Policy Optimization (PPO)
- **Motivazione**: decisione in due fasi (operazione + macchina) → due attori coordinati.
- **PPO**: stabilizza il gradiente policy tramite clipping del rapporto tra probabilità nuove e vecchie.
- **Critico condiviso**: stima del valore dello stato `V(s)` per calcolare il vantaggio `A = R - V`.
- **Ciclo di training**:
  1. Raccolta dei rollouts nelle memorie.
  2. Calcolo dei reward scontati e normalizzazione dei vantaggi.
  3. Ottimizzazione con clipping, entropia per favorire esplorazione e loss del critico.
- **Scheduler** opzionale per il learning rate (StepLR).

---

## 5. Embedding dell'ambiente
- **Feature nodali**: lower bound cumulativo (LBM) e flag "operazione completata".
- **Graph pooling**: media sugli embedding per ottenere una rappresentazione globale dello stato.
- **Maschere**:
  - `mask`: job finiti o operazioni già scelte.
  - `mask_mch`: macchine/attrezzature non compatibili (1 = vietato).
- I nodi vengono aggiornati a ogni step tramite `permissibleLeftShift` e `getActionNbghs`.

---

## 6. Gestione attrezzature multiple
- Rappresenta ogni attrezzatura come una "macchina virtuale".
- Compatibilità: tempi non nulli per combinazioni valide, 0 altrimenti.
- Estensioni possibili:
  - Tempo di setup → aggiungi penalità o vincoli nell'ambiente.
  - Priorità dei micro-lotti → integra feature aggiuntive o modifica il reward.

---

## 7. Workflow del sistema
1. **Import istanza** (`DataRead.getdata`) → costruisce il tensore delle durate.
2. **Reset ambiente** (`FJSP.reset`) → inizializza maschere, grafo e feature.
3. **Rollout**:
   - `Job_Actor` sceglie l'operazione.
   - `Mch_Actor` sceglie la macchina/attrezzatura.
   - `FJSP.step` aggiorna orari, grafi e reward.
4. **Update PPO** dopo ogni batch di episodi.

---

## 8. Metriche di valutazione
- **Makespan**: massimo tempo di completamento tra macchine/attrezzature.
- **Reward cumulativo**: misura qualitativa dell'episodio.
- Usa `validation_realWorld.py` per confrontare policy allenate contro baseline euristiche.

---

## 9. Terminologia importante
- **Micro-lotto**: piccolo gruppo di pezzi dello stesso modello; può corrispondere a un job o a più operazioni.
- **Attrezzatura**: macchinario o setup diverso; è trattata come macchina virtuale con tempi dedicati.
- **Lower Bound (LB)**: stima ottimista del completamento; ridurlo aiuta a migliorare il makespan.
- **Policy / Critico**: rispettivamente il modello che decide e quello che valuta.

---

Con questi concetti puoi spiegare come l'algoritmo esplora combinazioni di job e attrezzature, apprende strategie per ridurre il makespan e come adattare il framework al tuo contesto produttivo.

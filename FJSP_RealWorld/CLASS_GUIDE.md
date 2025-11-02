# Guida rapida alle classi (FJSP_RealWorld)

Questo documento riassume le classi principali utili per adattare l’algoritmo di scheduling a casi industriali con una sola macchina dotata di attrezzature differenti. Le descrizioni sono volutamente semplici per aiutarti a capire dove intervenire.

---

## Ambiente e parsing

- **`FJSP_Env1.FJSP`**
  - Modello Gym dell’officina: tiene traccia delle operazioni già pianificate, di quelle disponibili e delle macchine/attrezzature compatibili.
  - Metodi chiave:
    - `reset(data, rule=None)`: inizializza lo stato partendo dal tensore delle durate (batch × job × operazione × macchina). Qui puoi cambiare come vengono calcolate le feature o le maschere per riflettere vincoli specifici (p.es. una sola macchina con 10 attrezzature).
    - `step(action, mch_a)`: applica la scelta dell’agente (operazione + macchina/attrezzatura), aggiorna la pianificazione e calcola il reward. Modifica qui se vuoi un reward diverso dal makespan.

- **`DataRead.getdata`**
  - Converte un file `.fjs` in dizionari Python (`n`, `m`, `operations_times`, ecc.).
  - Utile se vuoi importare i tuoi micro-lotti da un formato compatibile.

---

## Algoritmo PPO

- **`PPOwithValue.PPO`**
  - Contiene due attori (`policy_job`, `policy_mch`) e un critico condiviso. Gestisce training, update e salvataggio dei parametri.
  - Metodi principali:
    - `update(memories, epoch)`: implementa il passo PPO (clipping dei rapporti, calcolo dei vantaggi, entropia).
    - `save`/`load` (se aggiunti): punto perfetto per salvare checkpoint personalizzati.
  - Per un caso monomacchina con diverse attrezzature puoi mantenere la doppia policy: considera un “job actor” che decide quale micro-lotto processare e un “machine actor” che sceglie l’attrezzatura.

- **`PPOwithValue.Memory`**
  - Struttura di supporto che accumula stati, azioni, reward durante il rollout. Se vuoi loggare variabili extra (es. tipo attrezzatura montata) aggiungi i campi qui.

---

## Reti neurali

- **`models/PPO_Actor.Job_Actor`**
  - Encoder grafico + MLP che producono la distribuzione sulle operazioni fattibili.
  - Ingressi: feature nodali (LBM, stato operazioni), maschere di disponibilità, grafo disgiuntivo.
  - Uscite: indice dell’operazione scelta, embedding associato, log-probabilità.
  - Per personalizzare le feature aggiungi dimensioni in `FJSP_Env1.reset()` e aggiorna `MLPActor`.

- **`models/PPO_Actor.Mch_Actor`**
  - Decide la macchina/attrezzatura più adatta per l’operazione selezionata, tenendo conto dei tempi residui.
  - In caso di singola macchina con 10 attrezzature, tratta ciascuna attrezzatura come “macchina virtuale” modificando la maschera `mask_mch`.

- **`models/graphcnn_congForSJSSP.GraphCNN`**
  - GIN (Graph Isomorphism Network): propaga le informazioni lungo il grafo disgiuntivo (precedenze + conflitti macchina).
  - Se introduci nuove tipologie di vincoli puoi cambiare come viene costruito l’adjacency matrix (`FJSP_Env1.reset` + `updateAdjMat`).

- **`models/mlp.MLPActor`, `models/mlp.MLPCritic`**
  - Blocchi MLP standard per attori e critico. Per reti più leggere o più profonde modifica questi file.

- **`models/Pointer.Pointer`, `Mhattention.ProbAttention`**
  - Moduli di attenzione usati nelle varianti dell’attore. Puoi ignorarli se rimani sulla configurazione PPO standard.

---

## Supporto alla logica di scheduling

- **`permissibleLS.permissibleLeftShift`**
  - Calcola l’inserimento più anticipato possibile di un’operazione rispettando precedenze e conflitti sulle macchine.
  - Modifica qui se servono regole specifiche sulle attrezzature (setup time, tempo di cambio attrezzo).

- **`min_job_machine_time.min_job_mch`**
  - Costruisce le maschere per il “machine actor”, individuando macchine compatibili e priorità basate sui tempi.
  - Per gestire una singola macchina con strumenti multipli, assicurati che ogni attrezzatura corrisponda a una colonna valida nelle durate (`dur[:,:,tool]` > 0).

- **`updateAdjMat.getActionNbghs`**
  - Aggiorna gli archi del grafo quando scheduli una nuova operazione.
  - Da toccare se aggiungi nuove relazioni di precedenza oltre a quelle standard job/machine.

- **`dispatichRule.DRs`**
  - Implementa regole euristiche opzionali (FIFO, SPT ecc.). Puoi usarle come baseline o come vincolo ibrido.

---

## Script di esecuzione

- **`validation_realWorld.validate`**
  - Carica policy allenate e valuta il makespan su un set di istanze. Ottimo per confrontare “prima / dopo” le modifiche.

- **`run_example.py`** (nella root)
  - Esegue un rollout analitico mostrando le scelte step-by-step. Ideale per capire come l’agente elabora micro-lotti e attrezzature differenti.

---

## Consigli per il tuo scenario

- Mappa ogni attrezzatura disponibile sulla macchina come una “macchina virtuale” separata, mantenendo `configs.n_m = numero_attrezzature`.
- Assicurati che il tensore delle durate (`durations[job][operazione][attrezzatura]`) contenga 0 per attrezzature incompatibili.
- Se vuoi penalizzare cambi frequenti di attrezzatura, aggiungi un costo in `FJSP_Env1.step()` usando `self.mchMat` per sapere quale attrezzatura era montata prima.
- Per micro-lotti, considera di estendere le feature con il numero di pezzi rimanenti o il tempo medio richiesto, aggiornando `input_dim` e `Job_Actor`.

Con queste informazioni dovresti riuscire a identificare rapidamente i punti del codice dove intervenire per adattare lo scheduler alle tue attrezzature e ottimizzare il makespan dei micro-lotti.

---

## È fattibile nel tuo scenario?

- **Un’unica macchina con 10 attrezzature**: sì. Rappresenta ogni attrezzatura come una “macchina virtuale” diversa compilando il tensore delle durate con valori > 0 solo per le attrezzature compatibili; il `Mch_Actor` sceglierà tra 10 opzioni.
- **Micro-lotti/modelli differenti**: puoi modellare ogni combinazione lotto/modello come una sequenza di operazioni di un job. Se servono feature aggiuntive (quantità, priorità), aggiungile in `FJSP_Env1.reset()` e alza `input_dim`.
- **Setup tra attrezzature**: introduci un costo di cambio in `FJSP_Env1.step()` o modifica `permissibleLeftShift` per bloccare certe sequenze. Il framework è flessibile a vincoli addizionali.
- **Spiegabilità**: la pipeline ha una chiara separazione (ambiente → policy → reward). Il file `THEORY_OVERVIEW.md` sintetizza i concetti teorici necessari da comunicare.

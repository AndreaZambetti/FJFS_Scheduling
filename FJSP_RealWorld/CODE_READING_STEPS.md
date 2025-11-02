# Piano di analisi del codice (FJSP_RealWorld)

Segui questi step per comprendere il progetto prima di modificarlo. Ogni fase indica file e domande guida.

---

## 1. Visione d’insieme
1. Leggi `README.md` (sezione FJSP_RealWorld) per capire scopo e workflow.
2. Consulta `THEORY_OVERVIEW.md` per fissare i concetti teorici.
3. Apri `CLASS_GUIDE.md` per avere la mappa delle classi.

**Obiettivo**: sapere cosa fa il sistema, quali sono gli attori principali e quali file userai più spesso.

---

## 2. Ambiente di simulazione
1. Apri `FJSP_Env1.py`:
   - `reset`: come costruisce feature, maschere, grafi?
   - `step`: come aggiorna la schedulazione, calcola reward e done?
2. Esamina `permissibleLS.py` (funzione `permissibleLeftShift`) per capire l’inserimento anticipato.
3. Controlla `min_job_machine_time.py` per la costruzione delle maschere macchina.

**Domande**:
- Dove posso aggiungere la logica di setup tra attrezzature?
- Come rappresento nuove feature di stato?

---

## 3. Parser e dati
1. Leggi `DataRead.py`: come viene convertito il file `.fjs`?
2. Apri una istanza in `FJSP_RealWorld/FJSSPinstances` per capire il formato reale.

**Domanda**: come trasformare i micro-lotti/attrezzature aziendali in questo formato?

---

## 4. Modelli di rete
1. Studia `models/PPO_Actor.py`:
   - `Job_Actor`: ingessi/uscite, uso di GraphCNN.
   - `Mch_Actor`: come combina embedding e maschere macchina.
2. Esamina `models/graphcnn_congForSJSSP.py` per capire l’encoder GIN.
3. Dai un’occhiata a `models/mlp.py` per i moduli MLP.

**Domande**:
- Se aggiungo feature, dove devo aumentare le dimensioni?
- Come produco embedding per attrezzature speciali?

---

## 5. Algoritmo PPO
1. Apri `PPOwithValue.py`:
   - Classe `Memory`.
   - Metodo `update`: come usa le memorie, calcola vantaggi, aggiorna job/machine actor?
   - Metodo `train` (parte finale del file): come viene orchestrato il ciclo episodio → update?
2. Controlla `agent_utils.py` per la logica di campionamento e valutazione delle azioni.

**Domande**:
- Come integrare nuove penalità nel reward nel calcolo dei vantaggi?
- Dove salvare/caricare modelli personalizzati?

---

## 6. Script di validazione e demo
1. `validation_realWorld.py`: osserva come esegue rollouts greedy su file `.fjs`.
2. `run_example.py` (root): un esempio passo-passo utile per debug.
3. `toy_ppo_demo.py` (root): micro training REINFORCE per capire le primitive.

**Domanda**: come automatizzare benchmark sulla tua suite di micro-lotti?

---

## 7. Collegare i pezzi
- Traccia un flusso completo: `DataRead` → `FJSP.reset` → `Job_Actor/Mch_Actor` → `FJSP.step` → `Memory` → `PPO.update`.
- Identifica esattamente i punti dove dovrai intervenire per gestire:
  - Una sola macchina con 10 attrezzature (modifica maschere/feature).
  - Setup costosi (reward o vincoli).
  - Metriche aggiuntive (log, validazione).

---

## 8. Prepararsi alla modifica
- Crea una copia di un file `.fjs` rappresentando il tuo caso.
- Esegui `run_example.py` o `validation_realWorld.py` per avere un baseline.
- Annota in `CLASS_GUIDE.md` eventuali estensioni da implementare (es. nuova feature, penalità).
- Definisci un piano di test (metriche, istanze, baseline euristiche).

---

Seguendo questi step avrai una comprensione solida del codice e saprai precisamente dove intervenire per adattare l'algoritmo al tuo scenario industriale.

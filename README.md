# Flexible Job-Shop Scheduling con DRL (FJSP_RealWorld)

Questa repository contiene un’implementazione end-to-end di un algoritmo di *Deep Reinforcement Learning* pensato per risolvere problemi reali di **Flexible Job-Shop Scheduling (FJSP)**.  
Il focus della documentazione è sulla cartella `FJSP_RealWorld`, che offre:
- un ambiente Gym personalizzato per trasformare istanze industriali (`*.fjs`) in uno scenario RL;
- due policy neurali (selezione operazione e selezione macchina) addestrate con PPO multi-attore;
- strumenti per validare modelli pre-addestrati e per avviare nuovi processi di training.

---

## 1. Panoramica dell’Algoritmo

### Stato
Ogni stato codifica l’avanzamento del programma produttivo tramite:
- grafo disgiuntivo dell’istanza (`adj`);
- feature dei nodi: *Lower Bound* cumulativi e flag “operazione completata” (`fea`);
- maschere delle operazioni disponibili (`omega`, `mask`);
- maschere delle macchine ammissibili per ogni operazione (`mask_mch`).

Queste informazioni sono generate dall’ambiente `FJSP_RealWorld/FJSP_Env1.py`.

### Azioni
L’agente sceglie due azioni per step:
1. **Operazione da processare**: policy `Job_Actor` sfrutta un encoder GIN per produrre le probabilità sulle operazioni ammissibili.
2. **Macchina da assegnare**: policy `Mch_Actor` valuta le macchine compatibili considerando tempi residui e embedding dell’operazione scelta.

### Reward
Il reward è il miglioramento (negativo se peggiorativo) del *Lower Bound* massimo del makespan. In questo modo la policy è incentivata a ridurre il tempo di completamento globale.

### Algoritmo di Training
La classe `PPO` in `FJSP_RealWorld/PPOwithValue.py` addestra congiuntamente i due attori e condivide un critico. Il training avviene su batch multipli di istanze, con:
- rollout memorizzato in `Memory`;
- normalizzazione dei vantaggi;
- aggiornamento PPO (clipping) su politiche e critico;
- scheduler di learning rate opzionale.

---

## 2. Requisiti

| Componente | Versione consigliata |
|------------|----------------------|
| Python     | 3.8 – 3.11           |
| PyTorch    | ≥ 1.10 (CUDA opzionale) |
| Gymnasium  | ≥ 0.27               |
| NumPy      | ≥ 1.21               |
| Matplotlib | per grafici Gantt (opzionale) |

Installa le dipendenze principali:

```bash
python -m venv .venv
source .venv/bin/activate  # su Windows: .venv\Scripts\activate
pip install torch gymnasium numpy matplotlib
```

---

## 3. Struttura della cartella `FJSP_RealWorld`

```
FJSP_RealWorld/
├─ FJSP_Env1.py              # Ambiente RL per istanze reali
├─ PPOwithValue.py           # Implementazione PPO multi-attore
├─ Params.py                 # Iperparametri globali (n° job/machine, LR, ecc.)
├─ validation_realWorld.py   # Script di validazione su benchmark .fjs
├─ models/                   # Moduli di rete (GIN, MLP attori, attenzione)
├─ DataRead.py               # Parser per file benchmark .fjs
├─ utils/                    # Funzioni di supporto (log, tensori, ecc.)
├─ saved_network/            # Checkpoint pre-addestrati (policy_job.pth, policy_mch.pth)
└─ FJSSPinstances/           # Benchmark di esempio (.fjs)
```

---

## 4. Utilizzo dei Modelli Pre-addestrati

### 4.1 Validazione su Benchmark
Per calcolare il makespan utilizzando i checkpoint forniti:

```bash
cd FJSP_RealWorld
python validation_realWorld.py \
  --Pn_j 15 --Pn_m 15 \
  --Nn_j 15 --Nn_m 15 \
  --n_vali 1 \
  --seed 200
```

Lo script:
1. carica `policy_job.pth` e `policy_mch.pth` da `saved_network/FJSP_J15M15/best_value0/`;  
2. costruisce il batch di durate a partire dal file `.fjs`;  
3. esegue il rollout greedy e stampa il makespan risultante.

Puoi sostituire i file `.fjs` modificando il percorso dentro `validation_realWorld.py` (funzione `test`).

### 4.2 Rollout Dettagliato e Debug
Lo script `run_example.py` mostra passo passo la schedulazione di `HurinkVdata39.fjs`. Modifica `instance_path` per puntare ad altre istanze e `checkpoint_root` per usare modelli diversi. L’output include:
- operazione e macchina scelte a ogni step;
- reward ottenuto;
- makespan finale.

---

## 5. Training su Dati Aziendali

1. **Preparazione dati**  
   - Converte il tuo processo produttivo in formato `.fjs` (numero job, operazioni, macchine compatibili e tempi di lavorazione).  
   - Salva i file in `FJSP_RealWorld/FJSSPinstances/<nome>/`.

2. **Configurazione Iperparametri**  
   Aggiorna `Params.py` con il numero di job/machine su cui addestrare (`n_j`, `n_m`) e scegli learning rate, batch size, numero di episodi (`max_updates`).

3. **Dataset di Training**  
   Puoi riutilizzare `validation_realWorld.py` come esempio per caricare le istanze: crea un *DataLoader* che produca tensori `durations` con shape `(batch, n_job, max_op, n_machine)` (usa `DataRead.py` come riferimento).

4. **Avvio Training**  
   Usa la classe `PPO` in `PPOwithValue.py`:  
   - istanzia l’ambiente `FJSP` con `EachJob_num_operation`;  
   - esegui rollouts, memorizza le transizioni in `Memory`;  
   - chiama `ppo.update(memory, epoch)` a fine batch.  

5. **Salvataggio Modelli**  
   Salva periodicamente `policy_job.state_dict()` e `policy_mch.state_dict()` in `FJSP_RealWorld/saved_network/<tuo_run>/`.

---

## 6. Personalizzazioni

| Esigenza | Dove intervenire |
|----------|------------------|
| Feature aggiuntive | Estendi il vettore restituito da `FJSP_Env1.reset()` (ad es. aggiunta di backlog per job). Aggiorna `input_dim` in `Params.py` e l’encoder in `models/PPO_Actor1.py`. |
| Obiettivi diversi | Modifica il calcolo del reward in `FJSP_Env1.step()` per riflettere KPI aziendali (es. tardiness, energy cost). |
| Vincoli macchina | Adatta `mask_mch` e la logica di filtro in `min_job_machine_time.py` o nelle regole `dispatichRule.py`. |
| Euristiche ibride | Passa `rule=<nome>` a `env.reset(...)` per abilitare dispatching rules custom (ad es. `FIFO_SPT`, `MOPNR_EET`). |

---

## 7. Troubleshooting

- **Il modello seleziona macchine non consentite**: verifica che `mask_mch` sia correttamente inizializzato in `FJSP_Env1.reset()` e che la durata sia > 0 solo per macchine ammissibili.
- **Reward fermo a zero**: controlla `configs.rewardscale` e assicurati che `LBm.max()` venga aggiornato (vedi `FJSP_Env1.step()`).
- **Crashes per dimensioni mismatch**: allinea `configs.n_j`, `configs.n_m` ai dati reali prima di istanziare `PPO`.

---

## 8. Citazione

Se utilizzi questo codice in ambito accademico o commerciale ti invitiamo a citare:

> Kun Lei, Peng Guo, Wenchao Zhao, Yi Wang, Linmao Qian, Xiangyin Meng, Liansheng Tang.  
> *A multi-action deep reinforcement learning framework for flexible Job-shop scheduling problem.*  
> Expert Systems with Applications, Volume 205, 2022, 117796.

---

## 9. Contatti e Contributi

- Per adattamenti aziendali puoi creare issue o PR con richieste specifiche (feature, supporto a nuovi formati).
- Per segnalazioni di bug relativi al training su istanze reali allega sempre: file `.fjs`, parametri usati, log di training/validazione.

Buon lavoro con il tuo progetto di scheduling!

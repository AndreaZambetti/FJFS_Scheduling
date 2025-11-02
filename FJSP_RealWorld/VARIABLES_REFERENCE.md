# Riferimento rapido alle variabili

Questo memo spiega in modo semplice le variabili principali che incontrerai lavorando con il progetto. L’obiettivo è farti capire dove intervenire quando devi adattare reward, vincoli o dati.

---

## Loader dei microlotti

| Variabile | Da dove arriva | Significato | Cosa puoi fare |
|-----------|----------------|-------------|----------------|
| `durations` | `custom_loader.load_microlotti_instance` | Tensore con i tempi di processo (shape `(batch, n_job, max_ops, n_machines)`). Valori 0 indicano macchine incompatibili. | Modifica la logica per costruire questo tensore se ogni job ha più operazioni o se i tempi cambiano in base alla macchina. |
| `ops_per_job` | Loader | Array `(1, n_job)` con il numero di operazioni per job. Serve per creare il grafo. | Se un microlotto richiede più step, aggiorna questo vettore e il JSON. |
| `meta` | Loader | Informazioni di supporto (setup, minuti/pezzo, pezzi, nome macchine). | Solo per logging; puoi estenderlo con altri dati utili (priorità, costi, ecc.). |

---

## Ambiente `FJSP_Env1`

| Variabile | Dove la vedi | Significato | Quando toccarla |
|-----------|--------------|-------------|-----------------|
| `adj` | `env.reset` / `env.step` | Matrice di adiacenza del grafo disgiuntivo (Torch sparse). | Se aggiungi vincoli particolari (es. precedenze extra), modifica `updateAdjMat.getActionNbghs`. |
| `features` | `env.reset` / `env.step` | Feature per ogni operazione (di default: lower bound cumulativo + flag completato). | Se vuoi aggiungere altri indicatori (es. tempo di setup previsto), estendi il vettore qui e aggiorna `configs.input_dim` + gli attori. |
| `omega` | `env.reset` / `env.step` | Per ogni job, ID dell’operazione attualmente schedulabile. | Non modificare manualmente; è gestita dall’ambiente. Puoi usarla per bloccare job specifici. |
| `mask_job` | `env.reset` / `env.step` | Booleano: job già completati (`True`). | Usato dalle policy/euristiche per escludere job conclusi. |
| `mask_mch` | `env.reset` / `env.step` | Compatibilità macchina/operazione (`0` = ok). | Ottimo punto per modellare restrizioni (attrezzature dedicate, manutenzioni). |
| `dur` | `env.reset` / `env.step` | Durate pronte per gli attori (shape `(batch, tasks, n_machines)`). | Già derivata da `durations`; modifica solo se cambi la struttura dei dati. |
| `mch_time` | `env.reset` / `env.step` | Tempo corrente di disponibilità di ciascuna macchina. | Utile per reward custom legati al setup: confronta l’attrezzatura precedente (`self.mchMat`). |
| `self.mchsStartTimes`, `self.mchsEndTimes` | interno all’ambiente | Start e fine di tutte le operazioni su ciascuna macchina. | Sono la schedulazione vera e propria: usale per calcolare makespan, lateness, ecc. |
| `reward` | `env.step` | Default: `-(LBm.max() - self.max_endTime)`, cioè punisce l’aumento del lower bound del makespan. | Modifica qui se vuoi premiare sequenze particolari, penalizzare cambi attrezzo, anticipare job urgenti, ecc. |
| `self.mchMat` | interno all’ambiente | Matrice che indica quale macchina ha gestito ogni operazione di ogni job. | Serve per calcolare setup e analizzare la soluzione finale. |

---

## Parametri globali (`Params.py`)

| Variabile | Significato | Nota |
|-----------|-------------|------|
| `configs.n_j`, `configs.n_m` | Numero di job e macchine | Aggiornali quando cambiano i microlotti o il numero di attrezzature virtuali. |
| `configs.batch_size` | Quanti ambienti parallelizzi in training | Di solito 1 per casi piccoli, >1 per training seri. |
| `configs.lr`, `gamma`, `k_epochs`, `eps_clip` | Iperparametri PPO | Tuning fondamentale se avvii un training tuo. |
| `configs.input_dim` | Dimensione del vettore `features` | Aumenta se aggiungi nuove feature. |
| `configs.rewardscale` | Bonus quando il reward è 0 | Puoi resettarlo a un valore positivo per incentivare certe azioni. |
| `configs.device` | `"cuda"`/`"cpu"` | Imposta `'cpu'` se non hai GPU. |

---

## Attori PPO (`policy_job`, `policy_mch`)

| Output | Significato | Cosa guardare |
|--------|-------------|---------------|
| `action` | Operazione scelta (indice flat) | Converti in job/operazione con `first_col`/`last_col`. |
| `mask_mch_action` | Maschera macchine per l’operazione selezionata | Se vuoi forzare attrezzature specifiche, intervieni qui. |
| `env_mch_time` | Tempo attuale per macchina | Passato al `Mch_Actor`. Puoi aggiungere setup time extra prima di chiamare `policy_mch`. |
| `pi_mch` | Distribuzione sulle macchine | `max` per greedy, `Categorical.sample` per training. |

---

## Variabili di confronto in `run_example.py`

| Variabile | Da dove arriva | Significato |
|-----------|----------------|-------------|
| `random_makespan` | Rollout con pesi random | Makespan ottenuto da policy non addestrate (baseline “zero”). |
| `checkpoint_makespan` | Rollout con checkpoint pre-addestrato | Makespan di una policy già allenata. Se i `.pth` non esistono vale `None`. |
| `makespan` | Euristica manuale | Risultato della logica “lotto più lungo + macchina più libera”. |

---

## Quando modificare cosa

- **Setup attrezzature / vincoli custom** → agisci su `mask_mch`, `self.mchMat`, reward in `FJSP_Env1.step`.
- **Nuove feature per l’attore** → aggiungi elementi a `features` in `reset`, alza `input_dim` e aggiorna gli attori (`models/PPO_Actor.py`).
- **Dataset** → aggiorna il JSON o crea un loader personalizzato con più istanze (utile per training).
- **Iperparametri PPO** → agisci su `Params.py` prima del training, poi osserva i log di reward/makespan.
- **Salvataggio modelli** → usa `torch.save` dopo `ppo.update` per avere checkpoint personalizzati in `saved_network`.

Con questa mappa puoi orientarti rapidamente su cosa modificare in base all’obiettivo (reward, vincoli, training, ecc.). Usa il piccolo esempio per testare ogni cambiamento prima di passare ai dati reali.ა

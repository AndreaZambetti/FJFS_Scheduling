Riassunto di `dispatichRule.py`

Obiettivo del file
------------------
`dispatichRule.py` fornisce funzioni di "Dispatching Rules" (regole di smistamento) per il modulo FJSP_RealWorld. Lo scopo è calcolare lo spazio di azione (maschere) e, quando richiesto, suggerire una macchina per ogni operazione in base a regole classiche (FIFO, SPT, LWKR, ...). La funzione principale è `DRs(...)` che implementa molte varianti di regole.

Struttura generale
------------------
- Import: legge `configs` da `Params` e `numpy`.
- Funzione principale: `DRs(mch_time, job_time, mchsEndTimes, number_of_machines, dur, temp, omega, mask_last, done, mask_mch, num_operation, dispatched_num_opera, input_min, job_col, input_max, rule, last_col, first_col)`.

Cosa fa `DRs`
--------------
1. Copia lo stato temporaneo (`temp = np.copy(temp)`)
2. Calcola `mch_time`: per ogni macchina prende l'ultimo tempo di fine lavoro significativo (valori >= 0) dalla matrice `mchsEndTimes`. Se la macchina è vuota, imposta tempo a 0.
3. Calcola `job_time`: per ogni job prende l'ultimo tempo di completamento registrato in `temp` (se nessun task completato, 0).
4. Calcola `remain_num = num_operation - dispatched_num_opera` (numero rimanente di operazioni per job).
5. In base al valore di `rule` entra in blocchi `if/elif` che implementano regole diverse:
   - FIFO_EET: seleziona job con EET minima (earliest end time), poi per ogni task considera le macchine compatibili e sceglie la macchina con `mch_time` minima.
   - FIFO_SPT: come FIFO ma sceglie macchine con tempo di esecuzione minimo (Shortest Processing Time) per l'operazione.
   - MOPNR_SPT / MOPNR_EET: regole che privilegiano job con maggior numero di operazioni rimanenti (MOPNR), poi scegliere macchina con SPT o EET.
   - LWKR_SPT / LWKR_EET: regole basate su "lookahead" dei tempi minimi cumulati (`input_min`) per le rimanenti operazioni del job; scelta di macchina con SPT o EET.
   - MWKR_SPT / MWKR_EET: regole simili ma con criterio diverso (massimo lookahead).

6. In ciascun blocco la funzione costruisce:
   - `mask`: vettore booleano che indica quali job non sono candidati (1 = bloccato/ignorato in quel giro)
   - `mch_mask` o `m_masks`: maschere per macchina per ogni operazione (aggiornano la disponibilità o i macchinari già provati)
   - `min_task`, `mchFor_minTask`: elenchi di operazioni candidate e le macchine abilitate a processarle.

7. Il loop while continua rimuovendo i job/processi già considerati fino a quando si trova un'azione valida o si scade/`done` è True.

Valori in ingresso/utilizzati
------------------------------
- `mch_time`: array 1D con tempi correnti delle macchine (aggiornato all'inizio)
- `job_time`: array 1D con tempo di completamento corrente dei job
- `mchsEndTimes`: matrice (n_machines x timeline) con i tempi di fine per gli eventi sulla macchina
- `number_of_machines`: intero
- `dur`: array dei tempi di durata per job/operazione/macchina (shape: batch? job? op? m)
- `temp`: array temporaneo delle finish times delle operazioni (usato per LB)
- `omega`: vettore con l'ID dell'operazione attualmente disponibile per ciascun job
- `mask_last`: mask dei job già completati
- `done`: boolean che indica se l'episodio è terminato
- `mask_mch`: maschera iniziale delle macchine per ogni op
- `num_operation`: numero di operazioni per job
- `dispatched_num_opera`: numero di operazioni già dispatchate per job
- `input_min`, `input_max`: statistiche sui tempi di processamento (min, max)
- `job_col`, `last_col`, `first_col`: indici per convertire id operazione <-> (job, col)
- `rule`: stringa che identifica la regola di dispatch

Output
------
La funzione restituisce (in forma implicita nelle variabili locali o come tuple — controllare la firma originale) maschere e spazi azione per il job e la macchina. (Nella lettura del codice, nota che alcuni rami restituiscono `mch_action_space`, altri aggiornano `m_masks` e `mask`.)

Punti di attenzione / codifica
-----------------------------
- Il codice usa molte copie di matrici (`np.copy`) e costruisce nuove maschere frequentemente.
- Ci sono molte ripetizioni tra i blocchi `if` — possibilità di refactoring (estrarre funzioni comuni: scelta macchine, calcolo reverse/lookahead, aggiornamento maschere).
- Non è sempre chiaro quale sia esattamente la shape delle matrici `dur`, `mask_mch` e `temp` senza guardare il chiamante (FJSP_Env). Questo è il primo posto da leggere quando si testa: valori attesi e shapes.
- Funzionalità batch/non-batch: la funzione sembra operare su singola istanza (o su array con primo indice batch?), ma spesso assume dimensioni standard. Verificare comportamento con batch size > 1.

Applicabilità a uno schedulatore con una sola tipologia di macchina
-------------------------------------------------------------------
- Se la tua piattaforma ha una sola tipologia di macchina (ossia ogni operazione può essere eseguita su una sola macchina o su macchine identiche), il codice è adattabile con piccole modifiche:
  - Imposta `number_of_machines = 1` o crea `dur` in cui per ogni operazione la sola macchina possibile ha tempo > 0.
  - `mask_mch` e `mch_mask` diventeranno triviali (solo una entry), la logica di scelta macchina si ridurrà a quella unica macchina.
  - Molti rami che scelgono la macchina più veloce diventeranno ridondanti ma funzioneranno correttamente.
- Se per "una tipologia" intendi "più macchine identiche" (es. 5 macchine identiche): il codice già supporta più macchine; converrebbe mantenere `number_of_machines = 5` e riempire `dur` con lo stesso tempo per tutte le macchine compatibili. Alcune regole come SPT restano valide.

Suggerimenti per iniziare (point of beginning)
----------------------------------------------
1. Leggi `FJSP_Env.py` per capire le shapes delle variabili passate a `DRs`.
2. Aggiungi stampe di debug (o usa un test unitario) che costruisca un piccolo instance (es. 3 job x 2 macchine) e invochi `DRs` per vedere output/masks.
3. Refactor consigliato:
   - Estrarre una funzione ``choose_machines_for_tasks(...)`` che ritorna gli spazi azione macchina per un elenco di task.
   - Estrarre funzioni per i criteri SPT/EET/MOPNR/LWKR.
4. Se vuoi usarlo per schedulatore a singola tipologia: prova prima impostando `number_of_machines=1` o replicando dur per macchine identiche.

File e classi correlate nel progetto
------------------------------------
- `FJSP_Env.py`: dove `DRs` è chiamato; importantissimo per capire shapes e flow.
- `min_job_machine_time.py`: alternativa quando `rule` è None.
- `updateEndTimeLB.py`: calcola LB (lower bounds) usati come feature.

Conclusione rapida
------------------
`dispatichRule.py` è un modulo di regole euristiche per dispatching nel contesto di FJSP. È utile, leggibile e relativamente modulare ma può essere semplificato e rifattorizzato per ridurre duplicazioni. Si può applicare senza grandi problemi a scenari con una sola tipologia di macchina adattando opportunamente `number_of_machines` o `dur`.

---
Generazione PDF
---------------
Nella cartella del progetto ho aggiunto uno script Python `generate_dispatichRule_pdf.py` che legge questo file di riassunto (`dispatichRule_summary.md`) e genera un PDF usando `reportlab`.
Istruzioni per creare il PDF:
1. Installa reportlab: `pip install reportlab`
2. Esegui: `python generate_dispatichRule_pdf.py`


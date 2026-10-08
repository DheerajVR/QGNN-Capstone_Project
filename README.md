# QGNN-Capstone_Project
# Predicting per-qubit failure on real IBM chips with a graph neural network

## 1. What was built

The chip is a graph: nodes are qubits, edges are the physical couplings that
two-qubit gates run on. A sample is one **(device, qubit patch, circuit)**
triple — the induced coupling subgraph of the qubits a specific circuit was
compiled onto, with node features fusing *what the hardware is* (T1, T2,
readout error, one-qubit gate error, incident two-qubit gate errors) with
*what the circuit asks of it* (gate counts, busy time, idle time, schedule
span).

## 2. The data is real IBM hardware

Training data comes from recorded calibration of **54 real IBM Quantum
processors** — Falcon, Hummingbird, Eagle, Heron, 5 to 156 qubits. The T1s, T2s, readout
errors and per-pair gate errors are IBM's own measurements; nothing about the
hardware is invented. 

Some challenges we faced:
Thirteen of the 67 available snapshots were dropped: one flagged
non-representative by Qiskit, one whose two-qubit calibration had failed, one
single-qubit device, and ten older devices with no recorded one-qubit gate
error — dropped rather than imputed, since imputing across devices would be
fabricated data.

**7 020 samples · 41 192 labelled qubits · 12 circuit families · widths 4–9.**
The labels span the full useful range — median circuit fidelity 0.966, 5th
percentile 0.436, minimum 0.000; 43 % of qubits exceed 1 % infidelity, the 99th
percentile is 0.565.

## 3. Results

Four baselines: the training mean; a **calibration error budget** (the analytic
per-qubit estimate you would compute today from the calibration table alone,
linearly recalibrated on the training set); gradient-boosted trees on the
identical features; and an MLP with the identical features and no message
passing. The MLP is the ablation that isolates what graph structure buys.


* **On ranking, the GNN wins clearly.** Within-circuit Spearman 0.394 versus
  0.308 for the same features without message passing and 0.116 for the
  calibration heuristic. It names the single worst qubit in a circuit 44 % of
  the time (a random guess on a 4–9 qubit patch is 11–25 %).
* **On absolute magnitude, it does not.** Node MAE 0.426 dex is statistically
  indistinguishable from the MLP's 0.419 (run-to-run spread on repeated GNN
  training is ±0.009 dex). What message passing buys is *ordering qubits within
  a circuit*, which is exactly the decision a user makes.
* **Against today's practice the margin is large.** The calibration budget is
  2.4× worse in MAE and barely better than random at ranking qubits within a
  circuit (ρ = 0.116) — it knows which qubits are bad in general, not which are
  bad *for this circuit*.

The in-distribution (random) split is slightly better across the board — MAE
0.388, within-circuit ρ 0.410, P@m 0.638 — so transfer to unseen chips costs
little.

### Does it actually help you pick qubits?

The end-use test, on the same 10 held-out chips: 33 tasks (6 circuit families ×
device), 20 candidate qubit patches per task, every candidate simulated exactly
so the true best is known, then each strategy picks one **without** seeing those
results.

| strategy | mean true fidelity of its pick | mean percentile in pool | picked the actual best |
|---|---|---|---|
| random | 0.9476 | 60 % | 6 % |
| calibration budget | 0.9627 | 83 % | 27 % |
| **QChipGNN** | **0.9637** | **90 %** | **36 %** |
| oracle (best possible) | 0.9692 | 100 % | 100 % |

The GNN closes about 75 % of the gap between a random layout and the oracle,
versus about 70 % for the calibration heuristic. The absolute fidelity gain over
the heuristic is small (+0.001) because both strategies avoid the obviously bad
patches; the difference shows up in how often each lands on the genuinely best
one (36 % vs 27 %).

## 4. What this does and doesn't establish

**Does:** given a real chip's calibration and a specific compiled circuit, a GNN
over the coupling graph ranks which qubits will carry the most error
substantially better than the calibration table alone, and better than the same
features without message passing. It transfers to processors it has never seen,
including across generations and across a 5→156 qubit size range.

**Doesn't:** ground truth here is exact simulation under IBM's *calibrated*
error model, not queue time on the physical machine. Coherent and
non-Markovian error, crosstalk from spectator qubits outside the patch, and
drift between calibration and execution are not in the labels — so these numbers
are an upper bound on what the same pipeline would achieve against hardware
counts. Each device contributes a single calibration snapshot, so temporal drift
is represented across devices rather than within one. Widths are capped at 9
qubits by the cost of exact density-matrix simulation, not by the architecture.
<div align = "center">
       <img width="1958" height="510" alt="01_dataset" src="https://github.com/user-attachments/assets/3c3370b6-6e2f-4ecb-9f48-8693bc5700ac" />
       <p><em>Dataset</em></p>
   </div>

<div align = "center">
       <img width="2010" height="615" alt="02_methods_device_split" src="https://github.com/user-attachments/assets/7aecef7f-84fb-47ad-93b1-fde2ccc6ac95" />
       <p><em>Device Split</em></p>
   </div>

<div align = "center">
       <img width="2010" height="615" alt="03_methods_family_split" src="https://github.com/user-attachments/assets/b1b02a35-5ccd-42e1-a4cb-b1afa21a240a" />
       <p><em>Family Split</em></p>
   </div>

<div align = "center">
       <img width="1425" height="615" alt="05_circuit_fidelity" src="https://github.com/user-attachments/assets/ad8ed56a-e308-4552-84c5-fab139e6ec6c" />
        <p><em>Circuit Fidelity</em></p>
   </div>

<div align = "center">
       <img width="1244" height="1026" alt="06_per_family" src="https://github.com/user-attachments/assets/cbda4bce-75be-41a1-bcf6-cd034e8116fa" />
       <p><em>Ranking by Family</em></p>
   </div>

<div align = "center">
       <img width="1564" height="1158" alt="08_chip_heatmap" src="https://github.com/user-attachments/assets/706818fb-91fd-446c-bf93-f47d132914e1" />
       <p><em>Chip Heat Map</em></p>
   </div>

# What each config changes, and why

## How a config is read

Every file in `cnn_lstm_multi_loss/` builds the same wrapper, `MultiHeadTaskRegressor`:
one sub-network per entry of `config_list` (each entry is a file in `single/` with
that sub-network's hyper-parameters), all fed the same series, outputs averaged.
The loss is a weighted sum (`w_losses`, index 0 = the averaged output, then one
weight per member) of a regression loss on the average (`main_loss`) and one per
member (`losses`), plus a `MarginRankingLoss` between this series and a second,
randomly paired series from the batch, weighted by `margin_loss_w`.

The ranking term is `max(0, -s * (y1 - y2) + margin)` with `s = ±1` the true
order. Targets are divided by 500, so the inherited `margin: -0.04` is -20 cells:
the term is zero unless a pair is mis-ordered by more than 20 cells, which a
trained model almost never does. Until config 40 the ranking loss was therefore
effectively off, and every `margin_loss_w` value before that was a dead knob.

| key | meaning | default if absent |
|---|---|---|
| `mode` | which sub-network class each `config_list` entry builds (table below) | `cnn_lstm` |
| `config_list` | one `single/*.yml` per member; a one-entry list is a single model | |
| `losses`, `main_loss` | `L1` or `HuberLoss`, per member and on the average | |
| `w_losses` | weights of `[average, member 1, ..., member N]` | |
| `margin_loss`, `margin_loss_w` | ranking loss on, and its weight | on, 3 |
| `margin` | ranking margin in scaled-target units (see above) | `-0.04` |
| `optimizer`, `weight_decay`, `scheduler` | `adam` / `adamw`, L2 decay, `plateau` / `onecycle` / `cosine` | `adam`, 0, `plateau` |
| `hard_pair_frac`, `hard_pair_min_diff`, `hard_pair_max_diff` | fraction of ranking partners mined from a band of target difference (cells) | 0 = uniform pairs |

Everything not in the yml is shared: batch 256, Adam at 1e-3,
`ReduceLROnPlateau` (factor 0.7, patience 5), early stopping on validation loss
with patience 20, 200-epoch cap unless noted. Parameter counts were obtained by
instantiating each config. Results are test MAE / correct-percent on the 200k
split, one seed unless noted. Configs 30 to 36 and 46 were written but never
trained and have been removed.

## The sub-networks (`mode:`)

| mode | class | what it is |
|---|---|---|
| `cnn_lstm` | `CNNAttLSTM` (upstream) | A bidirectional LSTM over the 20 rows, a 1-D U-Net (`CNNContext`) that compresses the whole series into one query vector, then self-attention over the LSTM outputs followed by cross-attention of the query against them. Knobs `n_h`, `n_l`. |
| `gru` | `GRURegressor` | A bare GRU over the rows; prediction is linear in the final hidden state. No U-Net, no attention. Knobs `hidden_size`, `num_layers`, `bidirectional`, `dropout`, optional `readout`. |
| `cnn_att_gru` | `CNNAttGRU` | `CNNAttLSTM` with the LSTM swapped for a GRU of the same size knobs, so a size-for-size test of the U-Net and attention branches. |
| `event_gru` | `EventRecurrentRegressor` | State changes only at real doses: a residual MLP conditioned on the gap evolves it between doses (`flow_hidden`, `flow_steps`), a `GRUCell` applies each dose, then the state is evolved to the end of the 72 000-step dosing window. Padding rows never touch the state. |
| `ct_transformer` | `ContinuousTimeTransformer` | Attention over dose tokens only, with a learned per-head bias computed from the signed time difference between doses instead of positional encodings. |

`readout:` in a single config picks the output head. `linear` (default) is
`Linear(final state)`. `additive_log` makes the prediction
`exp(log y0 + sum_i c(h_i) + c_end(h_T))`: leftover cells as the mean target
`y0` times one multiplicative factor per dose and one for the end of treatment,
which is the form radiobiology suggests. The factor heads start at zero, so
training begins from predicting the mean.

## The inherited model: `1.yml` (392k parameters)

Three `CNNAttLSTM` sub-networks of 32 hidden units and 3 layers, averaged. L1
losses, ranking weight 5 at the default margin. Four complete runs:
4.52 to 4.59 MAE / 94.9 to 95.0 %.

## 19 to 26: a bare GRU, and how big it has to be

A GRU is a recurrent cell with two gates (update and reset) that reads the rows
in order and carries a hidden state forward. These configs use `GRURegressor`,
which is that and nothing else, to find out whether the baseline's accuracy
comes from the recurrence or from the U-Net and attention branches, and how big
the recurrence must be. The knobs: **hidden size** is the width of the state,
i.e. how much about the protocol so far the model can carry; **layers** stack
GRUs so the second reads the first's output sequence and can form features of
features (a dose's effect conditional on what came before); **bidirectional**
adds a second GRU reading the series backwards and concatenates both final
states, so every position also sees the doses still to come, which is legitimate
for a protocol fixed in advance; **dropout** between layers is the only
regularisation. All share the loss block of `1.yml`.

| config | size | params | result | note |
|---|---|---|---|---|
| `19_gru_single` | 16 hidden, 1 layer | 1k | 7.46 / 92.1 % | starting point |
| `20_gru_tiny` | 8 hidden, 1 layer | 0.3k | 7.95 / 91.5 % | halve 19 again: capacity is the bottleneck |
| `21_gru_medium` | 32 hidden, 2 layers, dropout 0.1 | 10k | 5.81 / 93.8 % | depth is the biggest single step |
| `22_gru_bidir` | 16 hidden, 1 layer, bidirectional | 2k | 7.01 / 92.2 % | bidirectionality alone, at the size of 19 |
| `23_gru_large` | 64 hidden, 2 layers, bidir., dropout 0.2 | 101k | 4.63 (s1), 4.65 (s2) / 95.0 % | first GRU in the baseline band |
| `24_gru_xl` | 96 hidden, 3 layers, bidir., dropout 0.2 | 392k | 4.457 (s1), 4.521 (s2) / 95.1 % | reference single model; base of 27 to 43 |
| `25_gru_xxl_single` | 128 hidden, 4 layers, bidir., dropout 0.25 | 992k | 4.62 (s1), 4.51 (s2) / 95.0 % | 2.5x the parameters of 24, no better |
| `26_gru_ensemble_diverse` | medium + large + xl averaged, each with its own L1 head | 503k | 4.46 (s1), 4.59 (s2) / 95.1 % | within noise of 24 alone |

Depth matters most (21 vs 19), bidirectionality helps a little on its own
(22 vs 19), and the error falls monotonically up to 24 and then stops. 24 has the
same parameter count as `1.yml` and beats it, so the U-Net and attention
branches were not what made the baseline work; it became the reference single
model. 26 is the upstream trick of mixing member sizes inside one model and is
within seed noise of 24. A second seed of 23 to 26 with Welch's t-test
(`summarize_gru_seed_sweep.py`) found the four indistinguishable, so the sweep
stopped at xl.

## 27 to 29: the U-Net and attention branches put back on a GRU

`CNNAttGRU` is `CNNAttLSTM` with the LSTM replaced by a GRU of exactly the size
used in 21, 23 and 24, so each pair differs only by the presence of the U-Net
query and the two attention passes. If those branches carry information the
recurrence misses, the member with them should win at every size.

| config | same core as | params | result | against the plain GRU |
|---|---|---|---|---|
| `27_cnn_att_gru_medium` | 21 (32/2) | 44k | 5.72 / 93.7 % | better (5.81) |
| `28_cnn_att_gru_large` | 23 (64/2 bidir.) | 295k | 4.57 / 95.0 % | better (4.63) |
| `29_cnn_att_gru_xl` | 24 (96/3 bidir.) | 775k | 4.565 / 95.0 % | worse (4.457) at twice the parameters |

The branches help a core that is too small to summarise the protocol on its own
and become redundant, then harmful, once the core is large enough. Not carried
forward.

## 37 and 38: optimiser, scheduler, loss shape and weight averaging

Four standard training-process changes stacked on the two best models.
**Huber loss** is quadratic for residuals below a threshold and linear above,
so it has smooth gradients near zero without the outlier sensitivity of a
squared loss. **AdamW** applies weight decay directly to the weights rather
than through the gradient (decoupled L2, here 1e-4). **OneCycle** replaces the
plateau-driven decay with a fixed schedule that ramps the learning rate up and
anneals it to zero over the epoch budget. **Stochastic weight averaging**
(`-swa_lrs 0.0005` at launch) keeps a running average of the weights over the
final epochs, which tends to settle in a flatter minimum. The ranking loss is
untouched.

| config | base | params | result | against the plain run |
|---|---|---|---|---|
| `37_lstm_best_twitched` | `1.yml` + the four changes | 392k | 4.885 / 94.6 % | worse (4.55) |
| `38_gru_best_twitched` | `24_gru_xl` + the four changes | 392k | 4.856 / 94.9 % | worse (4.46) |

Both are clearly worse. The likely reasons are the schedule, which commits to the
full epoch budget where the plain runs stop at the validation optimum, and the
loss, which is quadratic exactly where this problem's residuals live (a few
cells on a target of several hundred) and so flattens the gradient there. The
stack was dropped; the knobs remain in the code as options.

## 39 and 40: making the ranking loss do something

Margin 0 makes the ranking term penalise every mis-ordered pair in proportion to
how far the two predictions are from the right order. 40 is 24 with only that
change and is the control. 39 adds **hard-pair mining**: instead of a uniformly
random partner, half of the partners are drawn from series whose true target is
4 to 15 cells away, pairs that are hard to order but not so close that label
noise decides the order.

| config | change from 24 | params | result | verdict |
|---|---|---|---|---|
| `40_gru_xl_margin_0` | margin 0, random pairs | 392k | 4.427 (s1), 4.550 (s2) / 95.2 % | +0.13 pt correct, -0.03 MAE, inside the CI; kept as the recommended setting |
| `39_gru_xl_hardpairs` | margin 0 + hard pairs (`hard_pair_frac 0.5`, band 4 to 15 cells) | 392k | 4.561 / 95.1 % | 0.10 MAE worse than 40; dropped |

Margin 0 is a small gain inside the bootstrap interval, so it is recommended but
not a result. Hard pairs hurt: forcing the model to order near-ties spends
capacity on distinctions the labels themselves cannot support.

## 41 to 45: models shaped like the simulator

The simulator behind the data evolves a tumour continuously and applies doses at
instants; a GRU stepping over a fixed 20-row grid with zero padding is a poor
match. `EventRecurrentRegressor` (41, 42) mirrors the simulator's structure. A
hidden state starts from a learned initial vector. Between two doses it is
**evolved** by a residual MLP that takes the state and the elapsed gap on a
linear and a log scale, applied in `flow_steps` sub-steps; this is where
repopulation and repair can be represented. At each dose a `GRUCell` applies a
**jump** conditioned on the dose, its square and its time; this is where cell
kill can be represented. After the last dose the state is evolved to the end of
the 72 000-step dosing window (the simulation itself runs to 144 000 steps; the
five untreated days are left to the readout), and padding rows never touch the
state, so the dose count does not matter.

The `additive_log` readout (42, 43, 45) changes only the output head to the
per-dose multiplicative form described above. 43 adds only the readout to 24 so
the two ideas can be separated.

`ContinuousTimeTransformer` (44, 45) is the attention version of the same idea:
tokens are the doses only, each carrying dose, dose squared, absolute time, gap
from the previous dose and time left to the horizon, and every attention head
adds a learned bias computed from the signed time difference between two doses,
so "how long ago" is explicit rather than inferred from a position index. The
prediction is a masked mean over dose tokens, or the additive log readout.

The loss block is identical to `24_gru_xl` in all five, so only the model
differs.

| config | single config | change | params | result (200k) | result (400k, uniform test protocols) |
|---|---|---|---|---|---|
| `41_event_gru` | `event_gru`: 96 hidden, flow MLP 192, 2 sub-steps, dropout 0.1, linear readout | event recurrence replaces the GRU-over-rows | 94k | 4.510 / 94.9 % | |
| `42_event_gru_addlog` | `event_gru_addlog`: same, `readout: additive_log` | 41 + the readout | 99k | 4.399 (s1), 4.383 (s2, 300-epoch cap), 4.289 (s3, 300-epoch cap) / 95.1 to 95.3 % | 4.226 / 4.194 / 4.175 (s1 to s3), 95.65 to 95.72 % |
| `43_gru_xl_addlog` | `gru_xl_addlog`: `gru_xl` with `readout: additive_log` | 24 + the readout only | 411k | 4.440 (s1), 4.500 (s2) / 95.1 % | |
| `44_ct_transformer` | `ct_transformer`: d_model 96, 4 heads, 3 layers, FF 192, linear readout | event transformer | 235k | 5.190 / 94.4 % | |
| `45_ct_transformer_addlog` | same, `readout: additive_log` | 44 + the readout | 240k | 5.064 / 94.4 % | |

Event recurrence alone is slightly worse than the plain GRU and the readout
alone is within noise, but together (42) they give the best single model on 200k
and, repeated on 400k, 4.18 to 4.23 MAE. The combination works because the
readout asks for exactly the per-dose quantity the event recurrence produces.
The continuous-time transformer is 0.6 to 0.7 MAE behind on one seed and was
dropped: with at most 20 events per protocol there is little for attention to
learn that the recurrence does not already capture.

`24_gru_xl` (seeds 1, 2) and `42_event_gru_addlog` (seeds 1 to 3) are the five
runs repeated on the 400k data (`scripts/run_400k_baseline.sh`); on the uniform
half of the 400k test split `24_gru_xl` scores 4.273 / 4.287.

## Lineage in one picture

```
1.yml (3x CNNAttLSTM, 392k)                 4.55
 └─ 37 + Huber/AdamW/OneCycle/SWA           4.89  worse
19 → 20 → 21 → 22 → 23 → 24_gru_xl → 25     size sweep of a bare GRU; 24 (392k) is the knee
                         ├─ 26 diverse ensemble          4.46 / 4.59  = 24
                         ├─ 27/28/29 + U-Net & attention 29: 4.565   worse than 24
                         ├─ 38 + Huber/AdamW/OneCycle/SWA 4.86       worse
                         ├─ 40 margin 0                   4.43 / 4.55 ≈ 24, ranking loss now live
                         │   └─ 39 + hard pairs           4.56       worse
                         ├─ 43 + additive_log             4.44 / 4.50 ≈ 24
                         └─ 41 event recurrence           4.51
                              └─ 42 + additive_log        4.29–4.40  best single; 4.18–4.23 on 400k
44 ct_transformer 5.19 → 45 + additive_log 5.06   dropped
```

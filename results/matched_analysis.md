## Matched study, by model (tests a one run filter keeps)

| Model | Suites | Tests generated | Kept after one run | Flaky kept tests | Flaky % (95% CI) | Suites with a flaky test | Rejected by the filter | Rejected but flaky | Suites that did not import |
|---|---|---|---|---|---|---|---|---|---|
| GPT 4.1 mini | 34 | 746 | 558 | 12 | 2.15 (1.23 to 3.72) | 6 | 188 | 10 | 0 |
| GPT 5.4 mini | 34 | 554 | 429 | 0 | 0.00 (0.00 to 0.89) | 0 | 125 | 0 | 0 |
| Claude Haiku 4.5 | 34 | 810 | 806 | 3 | 0.37 (0.13 to 1.09) | 2 | 4 | 0 | 0 |
| Claude Sonnet | 34 | 792 | 789 | 0 | 0.00 (0.00 to 0.48) | 0 | 3 | 0 | 0 |
| GPT 4.1 mini, guard prompt | 34 | 771 | 550 | 9 | 1.64 (0.86 to 3.08) | 4 | 221 | 3 | 0 |

## Human control per project (the known flaky test, same runner, same number of runs)

| Project | Gruber category | Runs | Failures | Reproduced |
|---|---|---|---|---|
| Intervallum | random | 200 | 12 | yes |
| analytic_shrinkage | random | 200 | 3 | yes |
| cuckoopy | random | 200 | 46 | yes |
| data-genie.git | random | 200 | 0 | no |
| data-partitioner | random | 200 | 4 | yes |
| filtered-intervaltree | random | 200 | 48 | yes |
| fuzzylogic | random | 200 | 18 | yes |
| gamble.git | random | 200 | 3 | yes |
| langsplit | random | 200 | 26 | yes |
| piripherals | async wait | 200 | 2 | yes |
| pylabeledrf | random | 200 | 10 | yes |
| python-matlab-functions | random | 200 | 3 | yes |
| pytsmp | random | 200 | 2 | yes |
| scikit-procrustes | random | 200 | 21 | yes |
| spreading_dye_sampler | random | 200 | 2 | yes |
| torch-layer-normalization | random | 200 | 15 | yes |
| words-to-regular-expression | random | 200 | 169 | yes |

## Flaky generated tests

- cuckoopy / haiku_plain_1 / `zz_haiku_plain_1.test_gen::test_delete_keeps_other_items_findable` pass=197 fail=3
- cuckoopy / haiku_plain_1 / `zz_haiku_plain_1.test_gen::test_len_tracks_mixed_inserts_and_deletes` pass=194 fail=6
- cuckoopy / haiku_plain_2 / `zz_haiku_plain_2.test_gen::test_many_inserts_remain_findable` pass=154 fail=46
- data-partitioner / gpt41mini_plain_1 / `zz_gpt41mini_plain_1.test_gen::test_internal_state_advance_after_yielding` pass=142 fail=58
- pylabeledrf / gpt41mini_plain_1 / `zz_gpt41mini_plain_1.test_gen::test_computeLRF_identical_simple_tree` pass=148 fail=52
- pylabeledrf / gpt41mini_plain_1 / `zz_gpt41mini_plain_1.test_gen::test_computeLRF_simple_flip_label` pass=158 fail=42
- scikit-procrustes / gpt41mini_plain_1 / `zz_gpt41mini_plain_1.test_gen::test_gbb_solver_invH_branch` pass=163 fail=37
- spreading_dye_sampler / gpt41mini_plain_1 / `zz_gpt41mini_plain_1.test_gen::test_make_blot_without_permitted_returns_blot` pass=137 fail=63
- spreading_dye_sampler / gpt41mini_plain_1 / `zz_gpt41mini_plain_1.test_gen::test_make_blot_returns_exact_requested_cells_if_possible` pass=118 fail=82
- spreading_dye_sampler / gpt41mini_plain_1 / `zz_gpt41mini_plain_1.test_gen::test_make_blot_with_squeeze_true_skips_forbidden_cells` pass=193 fail=7
- spreading_dye_sampler / gpt41mini_plain_2 / `zz_gpt41mini_plain_2.test_gen::test_make_blot_allows_squeeze_to_skip_nonpermitted_cells` pass=193 fail=7
- spreading_dye_sampler / gpt41mini_plain_2 / `zz_gpt41mini_plain_2.test_gen::test_make_blot_returns_none_if_nonpermitted_cell_found_and_squeeze_false` pass=194 fail=6
- spreading_dye_sampler / gpt41mini_plain_2 / `zz_gpt41mini_plain_2.test_gen::test_make_blot_with_non_standard_cell_dims` pass=128 fail=72
- spreading_dye_sampler / gpt41mini_plain_2 / `zz_gpt41mini_plain_2.test_gen::test_internal_neighbour_sharing_edges_accumulate` pass=129 fail=71
- torch-layer-normalization / gpt41mini_plain_2 / `zz_gpt41mini_plain_2.test_gen::test_layernorm_with_tuple_normalized_shape` pass=196 fail=4
- cuckoopy / gpt41mini_guard_1 / `zz_gpt41mini_guard_1.test_gen::test_delete_only_affects_existing_items` pass=193 fail=7
- cuckoopy / gpt41mini_guard_1 / `zz_gpt41mini_guard_1.test_gen::test_size_updating_after_insert_and_delete` pass=180 fail=20
- cuckoopy / gpt41mini_guard_2 / `zz_gpt41mini_guard_2.test_gen::test_insert_raises_when_full` pass=110 fail=90
- cuckoopy / gpt41mini_guard_2 / `zz_gpt41mini_guard_2.test_gen::test_insert_eviction_path` pass=107 fail=93
- cuckoopy / gpt41mini_guard_2 / `zz_gpt41mini_guard_2.test_gen::test_delete_does_not_affect_other_fingerprints` pass=127 fail=73
- data-partitioner / gpt41mini_guard_1 / `zz_gpt41mini_guard_1.test_gen::test_training_test_partition_ratios[0.1]` pass=192 fail=8
- data-partitioner / gpt41mini_guard_1 / `zz_gpt41mini_guard_1.test_gen::test_training_test_partition_ratios[0.5]` pass=137 fail=63
- data-partitioner / gpt41mini_guard_1 / `zz_gpt41mini_guard_1.test_gen::test_training_test_partition_ratios[0.8]` pass=165 fail=35
- pylabeledrf / gpt41mini_guard_1 / `zz_gpt41mini_guard_1.test_gen::test_mutateLabeledTree_flips_nodes_correctly` pass=117 fail=83

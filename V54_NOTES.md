# V54 notes

- Fix final selected_models / manifest entries to use a true global rerank across all stage2 models generated from stage1 survivors, rather than one pre-winner per stage1 unit.
- Keep the patched stage2 score rule: ce_est + 1.0 * gain_vs_zero, with gain_vs_myopic report-only.
- Make selected_spec.yaml less confusing by separating stage1 survivor lists from final selected spec/model lists.

*patient_coded_fact_entailment_pipeline.py 
**run_multi_pass_pipeline.py
***run_first_pass_seed_pipeline.py

python patient_coded_fact_entailment_pipeline.py \
  --src-build-root ${SATIR_DATA_ROOT}/patient_build_sigir \
  --side inclusion \
  --passes 3

单个pass 传递性调用 谁依赖谁 谁依赖谁 在run_first_pass
目前不依赖写死路径，暂时还是用的原来的依赖关系
IN_ROOT / OUT_ROOT 全局只从环境变量读；
正常使用时，优先用命令行 --in-root / --out-root，如果两个都没给、环境变量也没设，就直接报错退出；
所有原来依赖 IN_ROOT / OUT_ROOT 的逻辑都改成用传进来的 in_root / out_root。
python enrich_with_isa.py \
  --in-root ../../patient_build_sigir_exclusion/patient_coded_results \
  --out-root ../../patient_build_sigir_exclusion/patient_coded_results_isa \

python apply_schema_implications.py \
  --in-root ../../patient_build_sigir_exclusion/patient_coded_results_isa \
  --out-root ../../patient_build_sigir_exclusion/patient_coded_results_isa_imp \

python enrich_findings_to_procedure.py \
  --in-root ../../patient_build_sigir_exclusion/patient_coded_results_isa_imp \
  --out-root ../../patient_build_sigir_exclusion/patient_coded_results_finding_to_procedure_enrichment \

python enrich_findings_to_procedure_via_other_relations.py \
  --in-root ../../patient_build_sigir_exclusion/patient_coded_results_isa_imp \
  --out-root ../../patient_build_sigir_exclusion/patient_coded_results_finding_to_procedure_enrichment_other \
  --side exclusion

python enrich_findings_to_observable_entity.py \
  --in-root  ../../patient_build_sigir_exclusion/patient_coded_results_finding_to_procedure_enrichment \
  --out-root ../../patient_build_sigir_exclusion/patient_coded_results_finding_to_observable_entity_enrichment \
  --side exclusion

python enrich_procedure_to_finding.py \
  --in-root  ../../patient_build_sigir_exclusion/patient_coded_results_finding_to_procedure_enrichment \
  --out-root ../../patient_build_sigir_exclusion/patient_coded_results_procedure_to_finding_enrichment \
  --side exclusion

python enrich_observable_entity_to_finding.py \
  --in-root  ../../patient_build_sigir_exclusion/patient_coded_results_finding_to_observable_entity_enrichment \
  --out-root ../../patient_build_sigir_exclusion/patient_coded_results_observable_entity_to_finding_enrichment \
  --side exclusion

python enrich_with_isa_desc.py \
  --in-root  ../../patient_build_sigir_exclusion/patient_coded_results \
  --out-root ../../patient_build_sigir_exclusion/patient_coded_results_isa \
  --patient sigir-20141_exclusion



script for run prevent disease entailment
*patient_prevent_disease_entailment_pipeline.py
**enrich_with_isa_prevent_disease.py
python patient_prevent_disease_entailment_pipeline.py \
  --src-build-root ${SATIR_DATA_ROOT}/patient_disease_prevention_build \




sometimes may need rm -rf ~/.trialgpt_hiercache  
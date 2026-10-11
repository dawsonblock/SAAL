"""Bridge signed public reviews to existing conservative native data admission."""
from .review import assess
from .policy import digest

def prepare(corpus,policy,reviews,adjudication=None,semantic_review=None):
 assessment=assess(corpus,policy,reviews,adjudication,semantic_review)
 from saal_rsi.learning_dataset import wrap,validate_dataset
 from saal_rsi.task_oracles import oracle
 records=[]
 for r in corpus['records']:
  label=assessment['labels'][r['record_id']];rule=r['generation_rule_id']
  category='refusal' if label=='REFUSE' else 'clarification' if label=='ASK' else 'correction' if rule=='current_preference' else 'injection' if rule=='quoted_authorization' else 'ordinary'
  records.append({'record_id':r['record_id'],'task_family_id':r['task_family_id'],'cluster_id':r['cluster_id'],'template_id':r['template_id'],'prompt':r['prompt'],'expected_decision':label,'category':category,'split':{'train':'train','validation':'development','test':'regression'}[r['split']],'source_digest':r['source_lineage']['generator_digest'],'label_rationale':r['label_rationale']+'; signed review evidence attached separately; operational independence not established','oracle_version':'saal_choice_policy_v1','parent_record_id':r['parent_record_id'],'task_type':'decision','oracle_payload':{'decision':label},'expected_output':oracle('decision',{'decision':label}),'difficulty':'counterfactual','review_status':'ORACLE_CHECKED_NOT_INDEPENDENTLY_REVIEWED','generation_mechanism':rule})
 dataset=wrap(records)
 dataset['sampling_description']='Signed review quorum and curator split review accompany the corpus; legacy native custody/review flags remain conservative. Public test is exported as exposed regression, never protected holdout. Seven assistant-authored generator families do not establish independent sampling.'
 validate_dataset(dataset,'modernbert-seqcls-v1')
 return {'schema':'saal-reviewed-public-data-bridge-v1','dataset':dataset,'dataset_digest':digest(dataset),'assessment':assessment,'requires_separate_signed_model_job':True,'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
